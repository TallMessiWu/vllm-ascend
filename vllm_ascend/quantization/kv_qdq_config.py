# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in, fixed-recipe KV numerical experiment configuration."""

from dataclasses import dataclass

import torch

from .kv_fp4 import FP4Recipe

FORMATS = (
    ("g16-b", 16, "bf16"),
    ("g32-b", 32, "bf16"),
    ("g16-e", 16, "e4m3"),
    ("g32-e", 32, "e4m3"),
    ("mx4", 32, "e8m0"),
)


@dataclass(frozen=True)
class KVQDQSettings:
    format: str
    target: str
    k_recipe: FP4Recipe
    v_recipe: FP4Recipe
    diagnostic_copy: bool = False


def parse_kv_qdq(value: dict | None) -> KVQDQSettings | None:
    if value is None:
        return None
    keys = {"format", "target", "k_outer_scale", "v_outer_scale", "diagnostic_copy"}
    if not isinstance(value, dict) or set(value) - keys or not {"format", "target"} <= set(value):
        raise ValueError("kv_fp4_qdq requires format/target; optional K/V outer scales and diagnostic_copy only")
    if value["target"] not in ("off", "k", "v", "kv"):
        raise ValueError("kv_fp4_qdq.target must be off/k/v/kv")
    diagnostic_copy = value.get("diagnostic_copy", False)
    if type(diagnostic_copy) is not bool:
        raise ValueError("kv_fp4_qdq.diagnostic_copy must be a boolean")
    for name, group, scale in FORMATS:
        if value["format"] == name:
            return KVQDQSettings(
                name,
                value["target"],
                FP4Recipe(group, scale, value.get("k_outer_scale")),
                FP4Recipe(group, scale, value.get("v_outer_scale")),
                diagnostic_copy=diagnostic_copy,
            )
    raise ValueError("Unknown kv_fp4_qdq format")


def validate_kv_qdq_runtime(vllm_config):
    """Reject unsupported combinations without rewriting any runtime settings."""
    model, cache, parallel = vllm_config.model_config, vllm_config.cache_config, vllm_config.parallel_config
    if model.hf_config.model_type != "qwen3_moe" or model.dtype != torch.bfloat16:
        raise ValueError("KV QDQ reference currently requires plain BF16 Qwen3 MoE")
    if getattr(model, "runner_type", "generate") != "generate":
        raise ValueError("KV QDQ reference supports the generation runner only")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("KV QDQ reference must keep BF16 cache storage")
    for name in (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("KV QDQ reference currently requires single-rank execution")
    for name in ("lora_config", "speculative_config", "kv_transfer_config", "quant_config"):
        if getattr(vllm_config, name, None) is not None:
            raise ValueError("KV QDQ reference does not support LoRA, speculation, KV transfer or W/A quantization")
    if getattr(cache, "enable_prefix_caching", False):
        raise ValueError("KV QDQ prefix-cache sharing needs separate validation")
