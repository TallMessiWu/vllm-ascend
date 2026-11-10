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
from pathlib import Path

import torch

from vllm_ascend.quantization.kv_fp4 import qdq_fp4


def read_slots(cache, slots, bnsd):
    block_size = cache.shape[2] if bnsd else cache.shape[1]
    safe = slots.to(torch.int64).clamp(0, cache.shape[0] * block_size - 1)
    blocks, offsets = safe // block_size, safe % block_size
    return cache[blocks, :, offsets, :] if bnsd else cache[blocks, offsets]


def differences(actual, expected):
    """CPU only, exact BF16 bits including signed zero; never called in capture."""
    return int((actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).sum())


class KVReplayTrace:
    def __init__(self, key, query, slots):
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

    def before(self, key, value, slots):
        self.capture_calls += 1  # Python capture events, not a replay counter
        self.slot.copy_(slots[:1])
        for destination, tensor in zip(self.raw, (key, value)):
            destination.copy_(tensor)
        self.layout = {"shape": list(key.shape), "k_stride": list(key.stride()), "v_stride": list(value.stride())}

    def transformed(self, key, value):
        for destination, tensor in zip(self.qdq, (key, value)):
            destination.copy_(tensor)

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
        for side, raw, qdq, written, recipe in zip(
            ("k", "v"), self.raw, self.qdq, self.written, (settings.k_recipe, settings.v_recipe)
        ):
            # All transfers, recomputation and synchronization are outside the
            # graph, after the client has finished its normal requests.
            host_raw, host_qdq, host_written = raw.cpu(), qdq.cpu(), written.cpu()
            eager = qdq_fp4(raw.reshape(-1, raw.shape[-1]), recipe).reshape(raw.shape).cpu()
            cpu = qdq_fp4(host_raw.reshape(-1, raw.shape[-1]), recipe).reshape(host_raw.shape)
            row[side] = {
                "finite": bool(torch.isfinite(host_raw).all() and torch.isfinite(host_qdq).all()),
                "raw_nonzero": int(torch.count_nonzero(host_raw)),
                "replay_vs_npu_eager": differences(host_qdq, eager),
                "replay_vs_cpu": differences(host_qdq, cpu),
                "writer_vs_qdq": differences(host_written, host_qdq),
            }
            if not row[side]["finite"] or any(
                row[side][name] for name in ("replay_vs_npu_eager", "replay_vs_cpu", "writer_vs_qdq")
            ):
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
