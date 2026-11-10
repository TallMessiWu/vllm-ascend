# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in KV QDQ at the existing GQA writer boundary; cache stays BF16.

Both the cache writer and direct prefill consumers receive the transformed K/V.
Past cache entries are never requantized. This is a numerical experiment, not
packed-cache support, a new native operator or a throughput implementation.
"""

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
from vllm_ascend.quantization.kv_fp4 import qdq_fp4
from vllm_ascend.quantization.kv_qdq_config import parse_kv_qdq, validate_kv_qdq_runtime

# Bound temporary FP32 buffers without changing model/scheduler token budgets.
QDQ_TILE_ROWS = 1024
TRACE_DETAIL_LAYER = "model.layers.0.self_attn.attn"


def _qdq_tiled(tensor, recipe, observer=None):
    shape = tensor.shape
    rows = tensor.reshape(-1, shape[-1])
    if rows.shape[0] <= QDQ_TILE_ROWS:
        return qdq_fp4(rows, recipe, observer=observer).reshape(shape)
    if observer is not None:
        raise ValueError("Detailed KV trace only supports a single QDQ tile")
    return torch.cat(
        [qdq_fp4(rows[start : start + QDQ_TILE_ROWS], recipe) for start in range(0, rows.shape[0], QDQ_TILE_ROWS)]
    ).reshape(shape)


class AscendKVFP4QDQBackend(AscendAttentionBackend):
    forward_includes_kv_cache_update = True
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_impl_cls():
        return AscendKVFP4QDQImpl


class AscendKVFP4QDQImpl(AscendAttentionBackendImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        sinks=None,
        **kwargs,
    ):
        vc = get_current_vllm_config()
        validate_kv_qdq_runtime(vc)
        settings = parse_kv_qdq((vc.additional_config or {}).get("kv_fp4_qdq"))
        if settings is None:
            raise ValueError("KV QDQ backend requires explicit kv_fp4_qdq configuration")
        if (
            alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap not in (None, 0.0)
            or attn_type != "decoder"
            or kv_sharing_target_layer_name is not None
            or sinks is not None
        ):
            raise ValueError("KV QDQ supports plain causal GQA without sharing, sliding window, ALiBi, caps or sinks")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("KV QDQ must not change the native cache storage dtype")
        self.kv_qdq_settings = settings  # immutable recipe, fixed before warmup/capture
        self.kv_trace = None  # optional, per-layer device buffers retained across replay
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        logger.info_once(
            "Experimental KV QDQ: format=%s target=%s operation=%s, BF16 cache/attention; no packed-cache claim",
            settings.format,
            settings.target,
            "off"
            if settings.target == "off"
            else ("copy-only (no quantization)" if settings.diagnostic_copy else "qdq"),
        )

    def _transform_kv(self, key, value, trace=None):
        if key is None or value is None:
            if key is not None or value is not None:
                raise ValueError("K and V must be present or absent together")
            return key, value
        for tensor in (key, value):
            if (
                tensor.ndim != 3
                or tensor.shape[1:] != (self.num_kv_heads, self.head_size)
                or tensor.dtype != torch.bfloat16
            ):
                raise ValueError("KV QDQ requires BF16 [tokens, KV heads, head dimension]")
        if key.shape != value.shape or key.device != value.device:
            raise ValueError("K/V shapes and devices must match")
        settings = self.kv_qdq_settings
        # Explicit value-preserving control for the same writer/prefill seam.
        # No cached values or host reads; device copies are recorded for replay.
        # This does not reproduce all QDQ workspaces or timing.
        if settings.target in ("k", "kv"):
            key = (
                key.clone(memory_format=torch.contiguous_format)
                if settings.diagnostic_copy
                else _qdq_tiled(key, settings.k_recipe, observer=trace.observer("k") if trace is not None else None)
            )
        if settings.target in ("v", "kv"):
            value = (
                value.clone(memory_format=torch.contiguous_format)
                if settings.diagnostic_copy
                else _qdq_tiled(value, settings.v_recipe, observer=trace.observer("v") if trace is not None else None)
            )
        return key, value

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        if not getattr(attn_metadata, "causal", True):
            raise ValueError("KV QDQ reference requires causal attention metadata")
        recording = False
        if self.kv_qdq_settings.diagnostic_trace and key is not None and key.shape[0] == 1 and len(kv_cache) > 1:
            # Lazy import keeps the normal experiment independent of diagnostics.
            from vllm_ascend.attention.kv_fp4_trace import KVReplayTrace

            capturing = torch.npu.is_current_stream_capturing()
            if self.kv_trace is None:
                if capturing:
                    raise RuntimeError("KV trace requires a one-token warmup before capture")
                self.kv_trace = KVReplayTrace(
                    key,
                    query,
                    attn_metadata.slot_mapping,
                    detailed=getattr(self, "_layer_name", None) == TRACE_DETAIL_LAYER,
                )
            recording = capturing
            if recording:
                self.kv_trace.before(key, value, attn_metadata.slot_mapping)
        source = (key, value) if recording and self.kv_trace.raw_after is not None else None
        key, value = self._transform_kv(key, value, trace=self.kv_trace if recording else None)
        if recording:
            self.kv_trace.transformed(key, value, source=source)
        # The parent returns these K/V to its direct prefill path as well as
        # writing them to the persistent BF16 cache. Returning original K/V
        # here would leave PrefillNoCache mathematically unquantized.
        result = super().reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
        if recording:
            self.kv_trace.after_writer(self.key_cache, self.value_cache, self.use_bnsd_kv_cache)
        return result

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        result = super().forward_impl(query, key, value, kv_cache, attn_metadata, output)
        if self.kv_trace is not None and query.shape[0] == 1 and torch.npu.is_current_stream_capturing():
            self.kv_trace.attention.copy_(result.reshape(1, self.num_heads, self.head_size))
            self.kv_trace.attention_captured = True
        return result

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        key, value = self._transform_kv(key, value)
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
