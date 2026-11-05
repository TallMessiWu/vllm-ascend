# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NVFP4 activation QDQ reference; BF16 storage/GEMMs, not a packed FP4 kernel.

Recipe: ModelOpt e47c2538cdbf6cbc1ab758bc4909450caae17d29,
NVFP4QTensor.get_weights_scaling_factor / _cast_fp4. Exported input_scale is
amax / (6 * 448), a dequantization scale, NOT its reciprocal.
"""

import json
import math
from pathlib import Path

import torch

GROUP_SIZE = 16
QDQ_TILE_ROWS = 1024
FP4_MAX = 6.0
FP8_MAX = 448.0
FP8_MIN = 2.0**-9
# Midpoints, upper magnitudes and whether the upper code has even LSB.
FP4_ROUNDING = (
    (0.25, 0.5, False),
    (0.75, 1.0, True),
    (1.25, 1.5, False),
    (1.75, 2.0, True),
    (2.5, 3.0, False),
    (3.5, 4.0, True),
    (5.0, 6.0, False),
)


def round_e4m3_positive(value: torch.Tensor) -> torch.Tensor:
    """Finite positive E4M3FN rounding without requiring a device FP8 cast."""
    value = value.clamp(min=FP8_MIN, max=FP8_MAX)
    step = torch.exp2((torch.floor(torch.log2(value)) - 3).clamp(min=-9))
    return (torch.round(value / step) * step).clamp(max=FP8_MAX)


def nvfp4_activation_qdq(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Group16 QDQ with scalar or per-row [rows, 1, 1] FP32 global scales.

    Scales are validated/allocated before warmup. Only shape-dependent Python
    tiling is used, never .item()/CPU copies or cached activation values. Tiling
    bounds scratch space for top-k-expanded prefill without changing the server
    token budget. Non-finite inputs propagate instead of becoming valid zeros.
    """
    if x.ndim != 2 or x.shape[-1] % GROUP_SIZE:
        raise ValueError("NVFP4 A4 requires a 2D tensor with K divisible by 16")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("NVFP4 A4 reference supports BF16/FP32 input only")
    if global_scale.dtype != torch.float32 or global_scale.device != x.device:
        raise ValueError("NVFP4 A4 scales must be FP32 on the input device")
    if global_scale.numel() != 1 and global_scale.shape != (x.shape[0], 1, 1):
        raise ValueError("Expected a scalar or [rows, 1, 1] activation scale")
    if x.shape[0] == 0:
        return x
    if x.shape[0] <= QDQ_TILE_ROWS:
        return _qdq_tile(x, global_scale)
    return torch.cat(
        [
            _qdq_tile(
                x[start : start + QDQ_TILE_ROWS],
                global_scale if global_scale.numel() == 1 else global_scale[start : start + QDQ_TILE_ROWS],
            )
            for start in range(0, x.shape[0], QDQ_TILE_ROWS)
        ]
    )


def _qdq_tile(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / (FP4_MAX * global_scale)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    combined = round_e4m3_positive(scale) * global_scale
    normalized = blocks / combined
    magnitude = normalized.abs()
    rounded = torch.zeros_like(magnitude)
    for midpoint, upper, upper_even in FP4_ROUNDING:
        choose_upper = magnitude >= midpoint if upper_even else magnitude > midpoint
        rounded = torch.where(choose_upper, upper, rounded)
    # ModelOpt's E2M1 decode table maps sign-code 8 to positive zero.
    result = torch.where((normalized < 0) & (rounded != 0), -rounded, rounded) * combined
    result = torch.where(torch.isfinite(blocks), result, blocks)
    return result.reshape(x.shape).to(x.dtype)


def routed_global_scales(scales: torch.Tensor, groups: torch.Tensor, group_type: int, rows: int) -> torch.Tensor:
    """Map contiguous dispatched rows to logical experts (single rank, no EPLB).

    Empty experts and padded trailing rows are handled without host inspection.
    The comparison matrix is deliberate reference-path overhead, not an optimized
    kernel. GMM consumes only rows covered by group counts.
    """
    if scales.ndim != 1 or scales.numel() == 0 or group_type not in (0, 1) or groups.numel() != scales.numel():
        raise ValueError("Unexpected MoE group layout for NVFP4 A4 QDQ")
    if rows == 0:
        return scales.new_empty((0, 1, 1))
    ends = groups.reshape(-1) if group_type == 0 else groups.reshape(-1).cumsum(0)
    row_ids = torch.arange(rows, device=groups.device, dtype=ends.dtype)
    expert_ids = (row_ids[:, None] >= ends[None, :]).sum(dim=1).clamp(max=scales.numel() - 1)
    return scales.index_select(0, expert_ids).reshape(rows, 1, 1)


def read_json(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("NVFP4 metadata exceeds 16 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


class ActivationScales:
    """CPU-only, read-only calibration loading before model warmup/capture."""

    def __init__(self, source: str):
        from safetensors import safe_open

        self.root = Path(source).resolve()
        self.config = read_json(self.root / "config.json")
        cfg = self.config
        quant = cfg.get("quantization_config", {})
        if (
            cfg.get("model_type") != "qwen3_moe"
            or quant.get("quant_method") != "modelopt"
            or quant.get("quant_algo") != "NVFP4"
        ):
            raise ValueError("Expected a ModelOpt NVFP4 Qwen3 MoE calibration source")
        groups = quant.get("config_groups", {})
        if len(groups) != 1:
            raise ValueError("Expected one NVFP4 group16 configuration")
        for field in ("weights", "input_activations"):
            args = next(iter(groups.values())).get(field, {})
            if (args.get("num_bits"), args.get("type"), args.get("group_size")) != (4, "float", 16):
                raise ValueError("Expected float4 group16 weights and activations")
        self.num_experts = cfg["num_experts"]
        layers = cfg["num_hidden_layers"]
        if (
            not isinstance(layers, int)
            or not 0 < layers <= 128
            or not isinstance(self.num_experts, int)
            or not 0 < self.num_experts <= 512
        ):
            raise ValueError("Invalid Qwen3 MoE layer/expert counts")
        expected = set()
        for layer in range(layers):
            expected.update(
                f"model.layers.{layer}.self_attn.{p}.input_scale" for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            expected.update(
                f"model.layers.{layer}.mlp.experts.{e}.{p}.input_scale"
                for e in range(self.num_experts)
                for p in ("gate_proj", "up_proj", "down_proj")
            )
        mapping = read_json(self.root / "model.safetensors.index.json")["weight_map"]
        if {key for key in mapping if key.endswith(".input_scale")} != expected:
            raise ValueError("Missing or unexpected activation scale tensors")
        by_file: dict[Path, list[str]] = {}
        for key in sorted(expected):
            path = (self.root / mapping[key]).resolve()
            if not path.is_relative_to(self.root) or path.suffix != ".safetensors":
                raise ValueError("Scale shard escapes the original checkpoint")
            by_file.setdefault(path, []).append(key)
        self.values: dict[str, float] = {}
        for path, keys in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as tensors:
                for key in keys:
                    view = tensors.get_slice(key)
                    if view.get_dtype() != "F32" or view.get_shape() != []:
                        raise ValueError("input_scale must be a scalar FP32 tensor")
                    value = tensors.get_tensor(key).item()  # CPU, before capture, never activation data.
                    if (
                        not math.isfinite(value)
                        or not torch.finfo(torch.float32).tiny / FP8_MIN
                        <= value
                        <= torch.finfo(torch.float32).max / FP8_MAX
                    ):
                        raise ValueError("input_scale must give finite, normal FP32 block/global products")
                    self.values[key.removesuffix(".input_scale")] = value
        # Fused projections must share the exact exported calibration, not an invented max.
        for layer in range(layers):
            self.equal([f"model.layers.{layer}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")])
            for expert in range(self.num_experts):
                self.equal([f"model.layers.{layer}.mlp.experts.{expert}.{p}" for p in ("gate_proj", "up_proj")])

    def equal(self, prefixes: list[str]) -> float:
        values = [self.values[p] for p in prefixes]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(
                "Unequal input_scale in a fused projection is not supported; refusing silent recalibration"
            )
        return values[0]

    def linear(self, prefix: str) -> float:
        if prefix.endswith(".qkv_proj"):
            base = prefix.removesuffix("qkv_proj")
            return self.equal([base + p for p in ("q_proj", "k_proj", "v_proj")])
        return self.values[prefix]

    def experts(self, prefix: str, projection: str) -> list[float]:
        prefix = prefix.removesuffix(".routed_experts")
        return [self.values[f"{prefix}.{expert}.{projection}"] for expert in range(self.num_experts)]
