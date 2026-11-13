# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Experimental FP4 KV numerical reference, not a cache backend or native kernel.

All formats use E2M1 round-to-nearest-even, saturation and signed zero. Packing
places the even element in the low nibble. Groups run along the last dimension;
tail groups are zero padded and accounted for in payload_bytes.

pack_fp4/unpack_fp4 validate tensor values and may synchronize. They are offline
reference APIs, NOT capture-safe model hooks. qdq_fp4 uses device arithmetic only
and poisons invalid groups instead of silently converting NaN/Inf to finite FP4.
"""

import math
import struct
from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP4_MAX = 6.0
FP32_MIN_NORMAL = 2.0**-126
FP32_EXPONENT_SHIFT = 23
FP8_MIN = 2.0**-9
FP8_MAX = 448.0
# midpoint, upper code, upper code has even least-significant bit
FP4_BOUNDARIES = (
    (0.25, 1, False),
    (0.75, 2, True),
    (1.25, 3, False),
    (1.75, 4, True),
    (2.5, 5, False),
    (3.5, 6, True),
    (5.0, 7, False),
)


@dataclass(frozen=True)
class FP4Recipe:
    group_size: int
    scale_type: str
    outer_scale: float | None = None
    scale_selection: str = "ceil"

    def __post_init__(self):
        if type(self.group_size) is not int or self.group_size not in (16, 32):
            raise ValueError("Expected group16 or group32")
        if self.scale_type not in ("bf16", "e4m3", "e8m0"):
            raise ValueError("Unknown scale type")
        if self.scale_type == "e8m0" and self.group_size != 32:
            raise ValueError("MXFP4 requires group32")
        if self.scale_selection not in ("ceil", "mse") or (
            self.scale_selection != "ceil" and self.scale_type != "e8m0"
        ):
            raise ValueError("scale_selection must be ceil, or mse for MXFP4 only")
        if self.scale_type == "e4m3":
            scale = self.outer_scale
            if (
                type(scale) not in (int, float)
                or not math.isfinite(scale)
                or not FP32_MIN_NORMAL / FP8_MIN <= scale <= torch.finfo(torch.float32).max / (FP8_MAX * FP4_MAX)
            ):
                raise ValueError("E4M3 requires an explicit positive, finite FP32 outer scale in range")
            object.__setattr__(self, "outer_scale", struct.unpack("f", struct.pack("f", scale))[0])
        elif self.outer_scale is not None:
            raise ValueError("Outer scale is only defined for E4M3 candidates")


@dataclass(frozen=True)
class PackedFP4:
    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    recipe: FP4Recipe

    @property
    def payload_bytes(self) -> int:
        """Data + scales including tail padding; NOT allocator/workspace/HBM usage."""
        outer_bytes = 4 if self.recipe.outer_scale is not None else 0
        return (
            self.data.numel() * self.data.element_size()
            + self.scales.numel() * self.scales.element_size()
            + outer_bytes
        )


def _check_shape(x: torch.Tensor):
    if x.ndim < 1 or x.shape[-1] < 1 or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16/FP32 tensor with a nonempty last dimension")


def _codes(x: torch.Tensor, observer=None) -> torch.Tensor:
    if observer is not None:
        observer("normalized", x)
    magnitude = x.abs()
    code = torch.zeros_like(x, dtype=torch.int32)
    for midpoint, upper, upper_even in FP4_BOUNDARIES:
        choose = magnitude >= midpoint if upper_even else magnitude > midpoint
        code = torch.where(choose, upper, code)
    return code | (torch.signbit(x).to(torch.int32) << 3)


def _values(code: torch.Tensor) -> torch.Tensor:
    magnitude = code & 7
    value = torch.where(
        magnitude < 4, magnitude * 0.5, torch.where(magnitude < 6, magnitude - 2.0, (magnitude - 4.0) * 2)
    )
    return torch.where((code & 8) != 0, -value, value)


def _e4m3_scale_input(amax: torch.Tensor, recipe: FP4Recipe) -> torch.Tensor:
    # Keep the divisor a same-shape FP32 tensor. Scalar division can lower to
    # multiplication by a rounded reciprocal, moving an exact E4M3 midpoint
    # by one ULP (e.g. 6 / float32(6 * float32(0.02)) must round to 50).
    # full_like is device-only and replayed; no host reads or cached inputs.
    divisor = torch.full_like(amax, FP4_MAX * recipe.outer_scale)
    return amax / divisor


def _e4m3_round(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=FP8_MIN, max=FP8_MAX)
    step = torch.exp2((torch.floor(torch.log2(value)) - 3).clamp(min=-9))
    return (torch.round(value / step) * step).clamp(max=FP8_MAX)


def _e8m0_scale_values(exponent: torch.Tensor) -> torch.Tensor:
    # This reference accepts biased exponents [1, 254]: positive normal FP32
    # powers of two have exactly these exponent bits and a zero mantissa.
    # Avoid exp2 approximations (including values just below a power of two,
    # whose stored exponent would be one too small). The shift owns new storage;
    # its equal-width dtype view keeps it alive and is recomputed on replay.
    return (exponent.to(torch.int32) << 23).view(torch.float32)


def _scale_storage(scale: torch.Tensor, recipe: FP4Recipe) -> torch.Tensor:
    if recipe.scale_type == "bf16":
        return scale.to(torch.bfloat16)
    if recipe.scale_type == "e8m0":
        return ((scale.view(torch.int32) >> 23) & 255).to(torch.uint8)
    exponent = torch.floor(torch.log2(scale))
    mantissa = torch.round((scale / torch.exp2(exponent) - 1) * 8)
    normal = (exponent + 7) * 8 + mantissa
    return torch.where(scale < 2.0**-6, torch.round(scale * 512), normal).to(torch.uint8)


def _scale_values(scales: torch.Tensor, recipe: FP4Recipe) -> torch.Tensor:
    if recipe.scale_type == "bf16":
        return scales.float()
    if recipe.scale_type == "e8m0":
        return _e8m0_scale_values(scales)
    exponent = scales.to(torch.int32) >> 3
    mantissa = (scales.to(torch.int32) & 7).float()
    value = torch.where(exponent == 0, mantissa * FP8_MIN, (1 + mantissa / 8) * torch.exp2(exponent.float() - 7))
    return value * recipe.outer_scale


def _mx4_candidate_error(blocks, scale, amax, output_dtype):
    """Per-group squared error of the actual output carrier, entirely on device.

    Divide errors by the same amax for both candidates before squaring to avoid
    FP32 overflow/underflow at ordinary large/small power-of-two input scales.
    This preserves the mathematical ordering of group MSE. FP32 reductions can
    still tie; the caller retains ceil on ties or unordered (invalid) comparisons.
    """
    expanded = scale.unsqueeze(-1)
    reconstructed = (_values(_codes(blocks / expanded)) * expanded).to(output_dtype).float()
    error = (reconstructed - blocks) / amax.clamp(min=FP32_MIN_NORMAL).unsqueeze(-1)
    return error.square().sum(dim=-1)


def _quantize_groups(x: torch.Tensor, recipe: FP4Recipe, observer=None, exponent_shift=None):
    _check_shape(x)
    group = recipe.group_size
    padded = F.pad(x.float(), (0, (-x.shape[-1]) % group))
    blocks = padded.reshape(*x.shape[:-1], padded.shape[-1] // group, group)
    if observer is not None:
        observer("blocks", blocks)
    finite = torch.isfinite(blocks).all(dim=-1)
    amax = torch.where(torch.isfinite(blocks), blocks.abs(), 0.0).amax(dim=-1)
    if observer is not None:
        observer("amax", amax)
    if recipe.scale_type == "bf16":
        scale = (amax / FP4_MAX).clamp(min=FP32_MIN_NORMAL).to(torch.bfloat16).float()
    elif recipe.scale_type == "e4m3":
        scale = _e4m3_round(_e4m3_scale_input(amax, recipe))
    else:
        # Ceiling power-of-two recipe. Integer exponent extraction avoids log2
        # rounding a value immediately above a power of two down to that power.
        raw = (amax / FP4_MAX).clamp(min=FP32_MIN_NORMAL)
        # Give integer kernels owning storage instead of a dtype view of the
        # FP32 allocation. The exponent formula is unchanged; graph validation
        # is still required for the materialized input on the target device.
        bits = raw.view(torch.int32).clone(memory_format=torch.contiguous_format)
        if exponent_shift is not None and (
            not isinstance(exponent_shift, torch.Tensor)
            or exponent_shift.dtype != torch.int32
            or exponent_shift.shape != (1,)
            or exponent_shift.device != bits.device
        ):
            raise ValueError("MX4 exponent_shift must be an initialized int32 [1] tensor on the input device")
        shift = FP32_EXPONENT_SHIFT if exponent_shift is None else exponent_shift
        exponent = ((bits >> shift) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        if recipe.scale_selection == "mse":
            ceil_exponent = exponent.clamp(1, 254)
            lower_exponent = (ceil_exponent - 1).clamp(min=1)
            ceil_scale = _e8m0_scale_values(ceil_exponent)
            lower_scale = _e8m0_scale_values(lower_exponent)
            # Do not change the E2M1 codebook, group axis, or storage format.
            # Select independently for each new K/V group, never past cache.
            use_lower = _mx4_candidate_error(blocks, lower_scale, amax, x.dtype) < _mx4_candidate_error(
                blocks, ceil_scale, amax, x.dtype
            )
            exponent = torch.where(use_lower, lower_exponent, ceil_exponent)
        if observer is not None:
            observer("raw_scale", raw)
            observer("scale_bits", bits)
            observer("exponent", exponent)
        scale = _e8m0_scale_values(exponent.clamp(1, 254))
        if observer is not None:
            observer("e8_scale", scale)
    scale = torch.where(amax == 0, 1.0, scale)
    scale = torch.where(finite, scale, float("nan"))
    if observer is not None:
        observer("scale", scale)
    combined = scale * recipe.outer_scale if recipe.outer_scale is not None else scale
    code = _codes(blocks / combined.unsqueeze(-1), observer=observer)
    if observer is not None:
        observer("code", code)
    return code, scale


def qdq_fp4(x: torch.Tensor, recipe: FP4Recipe, observer=None, *, exponent_shift=None) -> torch.Tensor:
    """Device-only numerical QDQ; does not allocate a packed cache or save memory.

    The optional diagnostic observer may copy stage values on device. It must
    not read device values on the host during model capture/replay.
    MX4 model graph callers supply an immutable device int32 [1] containing 23,
    initialized outside capture. None retains the scalar offline/eager reference.
    """
    code, scale = _quantize_groups(x, recipe, observer=observer, exponent_shift=exponent_shift)
    combined = scale * recipe.outer_scale if recipe.outer_scale is not None else scale
    result = (_values(code) * combined.unsqueeze(-1)).flatten(-2)[..., : x.shape[-1]]
    if observer is not None:
        observer("dq_fp32", result)
    return result.to(x.dtype)


def pack_fp4(x: torch.Tensor, recipe: FP4Recipe) -> PackedFP4:
    """Offline/diagnostic packing; validates data before encoding (may synchronize)."""
    _check_shape(x)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Cannot pack non-finite FP4 input")
    code, scale = _quantize_groups(x, recipe)
    code = code.flatten(-2)
    data = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return PackedFP4(data, _scale_storage(scale, recipe), tuple(x.shape), recipe)


def unpack_fp4(packed: PackedFP4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Offline/diagnostic reader with strict descriptor checks (may synchronize)."""
    shape, recipe = packed.shape, packed.recipe
    if not shape or any(type(d) is not int or d < 0 for d in shape) or shape[-1] == 0:
        raise ValueError("Invalid logical shape")
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16/FP32 output")
    groups = math.ceil(shape[-1] / recipe.group_size)
    if packed.data.shape != (*shape[:-1], groups * recipe.group_size // 2) or packed.data.dtype != torch.uint8:
        raise ValueError("Invalid packed data shape/dtype")
    scale_dtype = torch.bfloat16 if recipe.scale_type == "bf16" else torch.uint8
    if packed.scales.shape != (*shape[:-1], groups) or packed.scales.dtype != scale_dtype:
        raise ValueError("Invalid scale shape/dtype")
    if packed.data.device != packed.scales.device:
        raise ValueError("Data and scales must be on the same device")
    scale_limit = 126 if recipe.scale_type == "e4m3" else 254
    if recipe.scale_type != "bf16" and not bool(((packed.scales > 0) & (packed.scales <= scale_limit)).all()):
        raise ValueError("Invalid scale bytes")
    scale = _scale_values(packed.scales, recipe)
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError("Invalid scale values")
    data = packed.data.to(torch.int32)
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(*shape[:-1], groups, recipe.group_size)
    result = (_values(codes) * scale.unsqueeze(-1)).flatten(-2)[..., : shape[-1]].to(dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("FP4 dequantization overflow in the requested dtype")
    return result
