# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Temporary, opt-in real-input replay snapshots. No host reads in capture.

Each backend owns one last-step snapshot, allocated during one-token warmup.
Only captured one-token calls write it. An explicit idle-worker RPC resets the
device counter after startup, so warmup/capture cannot masquerade as replay.
Reads and eager recomputation happen only in the reporting RPC after requests.
This instruments timing and allocation; passing is not a model repair.
"""

import subprocess
from functools import partial
from pathlib import Path

import torch

from vllm_ascend.quantization.kv_fp4 import qdq_fp4

MX4_TRACE_GROUP = 32
MX4_STAGES = (
    "blocks",
    "amax",
    "raw_scale",
    "scale_bits",
    "exponent",
    "e8_scale",
    "scale",
    "normalized",
    "code",
    "dq_fp32",
)


def read_slots(cache, slots, bnsd):
    block_size = cache.shape[2] if bnsd else cache.shape[1]
    safe = slots.to(torch.int64).clamp(0, cache.shape[0] * block_size - 1)
    blocks, offsets = safe // block_size, safe % block_size
    return cache[blocks, :, offsets, :] if bnsd else cache[blocks, offsets]


def differences(actual, expected):
    """CPU only, exact BF16/FP32/int32 element bits; never called in capture."""
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("Diagnostic comparisons require matching shapes and dtypes")
    view_dtype = torch.int32 if actual.element_size() == 4 else torch.int16
    return int((actual.contiguous().view(view_dtype) != expected.contiguous().view(view_dtype)).sum())


class KVReplayTrace:
    def __init__(self, key, query, slots, detailed=False):
        self.raw = [torch.empty_like(key, memory_format=torch.contiguous_format) for _ in range(2)]
        self.qdq = [torch.empty_like(t) for t in self.raw]
        self.written = [torch.empty_like(t) for t in self.raw]
        self.slot = torch.empty_like(slots[:1])
        self.count = torch.zeros(1, dtype=torch.int32, device=key.device)
        self.attention = torch.empty_like(query, memory_format=torch.contiguous_format)
        self.attention_captured = False
        self.capture_calls = 0
        self.capacity = 0
        self.layout = {}
        self.raw_after = [torch.empty_like(t) for t in self.raw] if detailed else None
        self.stages = None
        self.captured_stages = [set(), set()]
        if detailed:
            rows, width = key.numel() // key.shape[-1], key.shape[-1]
            group_shape = (rows, (width + MX4_TRACE_GROUP - 1) // MX4_TRACE_GROUP)
            self.stages = [
                {
                    name: torch.empty(
                        (*group_shape, MX4_TRACE_GROUP)
                        if name in ("blocks", "normalized", "code")
                        else (rows, width)
                        if name == "dq_fp32"
                        else group_shape,
                        dtype=torch.int32 if name in ("scale_bits", "exponent", "code") else torch.float32,
                        device=key.device,
                    )
                    for name in MX4_STAGES
                }
                for _ in range(2)
            ]

    def observer(self, side):
        return partial(self._save_stage, 0 if side == "k" else 1) if self.stages is not None else None

    def _save_stage(self, side, name, tensor):
        self.stages[side][name].copy_(tensor)
        self.captured_stages[side].add(name)  # capture coverage, not a replay counter

    def before(self, key, value, slots):
        self.capture_calls += 1  # Python capture events, not a replay counter
        self.slot.copy_(slots[:1])
        for destination, tensor in zip(self.raw, (key, value)):
            destination.copy_(tensor)
        self.layout = {"shape": list(key.shape), "k_stride": list(key.stride()), "v_stride": list(value.stride())}

    def transformed(self, key, value, source=None):
        for destination, tensor in zip(self.qdq, (key, value)):
            destination.copy_(tensor)
        if self.raw_after is not None:
            if source is None:
                raise ValueError("Detailed trace requires original input tensors after QDQ")
            for destination, tensor in zip(self.raw_after, source):
                destination.copy_(tensor)

    def _stage_report(self, side, host_raw, qdq, eager_stages, cpu_stages):
        after = self.raw_after[side].cpu()
        result = {"raw_before_after": differences(host_raw, after), "stages": {}}
        first_replay = "source_changed_during_qdq" if result["raw_before_after"] else None
        first_oracle = None
        missing = set(MX4_STAGES) - self.captured_stages[side]
        if missing:
            result["missing_stages"] = sorted(missing)
            result["first_divergence"] = first_replay or "UNVERIFIED: missing stages"
            result["first_replay_divergence"] = result["first_divergence"]
            result["first_oracle_divergence"] = "UNVERIFIED: missing stages"
            return result
        for name in MX4_STAGES:
            recorded, eager, cpu = self.stages[side][name].cpu(), eager_stages[name].cpu(), cpu_stages[name]
            counts = {
                "replay_vs_npu_eager": differences(recorded, eager),
                "replay_vs_cpu": differences(recorded, cpu),
                "npu_eager_vs_cpu": differences(eager, cpu),
            }
            result["stages"][name] = counts
            if first_replay is None and counts["replay_vs_npu_eager"]:
                first_replay = name
            if first_oracle is None and counts["npu_eager_vs_cpu"]:
                first_oracle = name
            if name == "exponent":
                deltas, sizes = torch.unique(recorded.to(torch.int64) - eager.to(torch.int64), return_counts=True)
                result["exponent_delta_histogram"] = {
                    str(delta): size for delta, size in zip(deltas.tolist(), sizes.tolist())
                }
        result["first_replay_divergence"] = first_replay or ("output_bf16" if qdq["replay_vs_npu_eager"] else "none")
        result["first_oracle_divergence"] = first_oracle or ("output_bf16" if qdq["npu_eager_vs_cpu"] else "none")
        # Keep the legacy name as an alias for the graph-versus-eager question.
        # Cross-device FP32 differences are evidence, not automatically a graph failure.
        result["first_divergence"] = result["first_replay_divergence"]
        return result

    def after_writer(self, key_cache, value_cache, bnsd):
        self.capacity = key_cache.shape[0] * key_cache.shape[2 if bnsd else 1]
        for destination, cache in zip(self.written, (key_cache, value_cache)):
            # Negative padding slots are clamped for memory safety and rejected
            # by report(); they never count as a valid writer comparison.
            destination.copy_(read_slots(cache, self.slot, bnsd))
        self.count.add_(1)

    def report(self, settings):
        count = int(self.count.cpu()[0])
        if count == 0:
            return {"status": "UNVERIFIED", "reason": "no one-token replay after arm", "replays": 0}
        slot = int(self.slot.cpu()[0])
        if not 0 <= slot < self.capacity:
            return {"status": "UNVERIFIED", "reason": "padding or invalid captured slot", "replays": count}
        row = {"status": "GREEN", "replays": count, "capture_calls": self.capture_calls, **self.layout}
        for side_index, (side, raw, qdq, written, recipe) in enumerate(
            zip(("k", "v"), self.raw, self.qdq, self.written, (settings.k_recipe, settings.v_recipe))
        ):
            # All transfers, recomputation and synchronization are outside the
            # graph, after the client has finished its normal requests.
            host_raw, host_qdq, host_written = raw.cpu(), qdq.cpu(), written.cpu()
            eager_stages, cpu_stages = {}, {}
            eager = (
                qdq_fp4(
                    raw.reshape(-1, raw.shape[-1]),
                    recipe,
                    observer=eager_stages.__setitem__ if self.stages is not None else None,
                )
                .reshape(raw.shape)
                .cpu()
            )
            cpu = qdq_fp4(
                host_raw.reshape(-1, raw.shape[-1]),
                recipe,
                observer=cpu_stages.__setitem__ if self.stages is not None else None,
            ).reshape(host_raw.shape)
            row[side] = {
                "finite": bool(torch.isfinite(host_raw).all() and torch.isfinite(host_qdq).all()),
                "raw_nonzero": int(torch.count_nonzero(host_raw)),
                "replay_vs_npu_eager": differences(host_qdq, eager),
                "replay_vs_cpu": differences(host_qdq, cpu),
                "npu_eager_vs_cpu": differences(eager, cpu),
                "writer_vs_qdq": differences(host_written, host_qdq),
            }
            if not row[side]["finite"] or any(
                row[side][name]
                for name in ("replay_vs_npu_eager", "replay_vs_cpu", "npu_eager_vs_cpu", "writer_vs_qdq")
            ):
                row["status"] = "RED"
            if self.stages is not None:
                detail = self._stage_report(
                    side_index,
                    host_raw,
                    {
                        "replay_vs_npu_eager": row[side]["replay_vs_npu_eager"],
                        "replay_vs_cpu": row[side]["replay_vs_cpu"],
                        "npu_eager_vs_cpu": row[side]["npu_eager_vs_cpu"],
                    },
                    eager_stages,
                    cpu_stages,
                )
                row[side]["detail"] = detail
                if detail.get("missing_stages") and row["status"] != "RED":
                    row["status"] = "UNVERIFIED"
                elif detail["first_divergence"] not in ("none", "UNVERIFIED: missing stages"):
                    row["status"] = "RED"
        row["attention_finite"] = bool(torch.isfinite(self.attention.cpu()).all()) if self.attention_captured else None
        if row["attention_finite"] is False:
            row["status"] = "RED"
        return row


class KVTraceWorkerExtension:
    """Use vLLM's worker-extension RPC seam; do not modify either model runner."""

    # The RPC dispatcher does not inherit model execution's inference mode.
    # Mutate the retained buffers in that mode, preserving captured addresses.
    @torch.inference_mode()
    def kv_trace_control(self, action):
        if action not in ("arm", "report"):
            raise ValueError("KV trace action must be arm/report")
        runner = self.model_runner
        armed = getattr(self, "_kv_trace_armed", False)
        layers = self.vllm_config.compilation_config.static_forward_context
        selected = []
        for name, layer in sorted(layers.items()):
            impl = getattr(layer, "impl", None)
            settings = getattr(impl, "kv_qdq_settings", None)
            if settings is not None and settings.diagnostic_trace:
                selected.append((name, impl))
        root = Path(__file__).resolve().parents[2]
        result = {
            "runner": f"{type(runner).__module__}.{type(runner).__name__}",
            "module": str(Path(__file__).resolve()),
            "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            # Some torch_npu releases accept assignment here without exposing
            # a getter. Optional metadata must not prevent arm/report.
            "allow_internal_format": getattr(torch.npu.config, "allow_internal_format", "UNVERIFIED: no getter"),
            "operation": "mx4 K+V qdq; diagnostic_trace=true",
            "trace_version": 3,
            "exponent_input": "materialized_int32",
            "detail_layer": next(
                (name for name, impl in selected if impl.kv_trace is not None and impl.kv_trace.stages is not None),
                None,
            ),
            "compilation": str(self.vllm_config.compilation_config.mode)
            + "/"
            + str(self.vllm_config.compilation_config.cudagraph_mode),
            "layers": {},
            "attention_comparison": "UNVERIFIED: output finiteness only; no eager attention oracle",
        }
        torch.npu.synchronize()  # explicit idle-worker diagnostic boundary
        for name, impl in selected:
            trace = impl.kv_trace
            if trace is None or trace.capture_calls == 0:
                result["layers"][name] = {"status": "UNVERIFIED", "reason": "no captured one-token writer"}
            elif action == "arm":
                trace.count.zero_()
                result["layers"][name] = {"status": "ARMED"}
            elif armed:
                result["layers"][name] = trace.report(impl.kv_qdq_settings)
            else:
                result["layers"][name] = {"status": "UNVERIFIED", "reason": "arm must precede requests"}
        torch.npu.synchronize()
        statuses = {row["status"] for row in result["layers"].values()}
        expected = self.vllm_config.model_config.hf_config.num_hidden_layers
        result["expected_layers"] = expected
        result["status"] = (
            "RED"
            if "RED" in statuses
            else "UNVERIFIED"
            if len(selected) != expected or "UNVERIFIED" in statuses
            else "ARMED"
            if action == "arm"
            else "GREEN"
        )
        self._kv_trace_armed = action == "arm" and result["status"] == "ARMED"
        return result
