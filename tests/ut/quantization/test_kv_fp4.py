# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only numerical contracts, without importing the framework/NPU runtime."""

import importlib.util
import math
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/kv_fp4.py"
spec = importlib.util.spec_from_file_location("kv_fp4_reference", SOURCE)
codec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec
spec.loader.exec_module(codec)


LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4_VALUES = tuple((c / 512 if c < 8 else (1 + (c % 8) / 8) * 2.0 ** (c // 8 - 7)) for c in range(127))


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def bf16(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", rounded))[0]


def scalar_reference(row, recipe):
    """Independent table search/struct rounding, not vector threshold arithmetic."""
    result = []
    for start in range(0, len(row), recipe.group_size):
        block = row[start : start + recipe.group_size]
        maximum = max(map(abs, block))
        if maximum == 0:
            scale = 1.0
        elif recipe.scale_type == "bf16":
            scale = bf16(max(f32(maximum / 6), 2.0**-126))
        elif recipe.scale_type == "e4m3":
            raw = min(max(f32(maximum / f32(6 * recipe.outer_scale)), 2.0**-9), 448.0)
            code = min(range(1, 127), key=lambda c: (abs(raw - E4_VALUES[c]), c % 2))
            scale = E4_VALUES[code]
        else:
            scale = 2.0 ** math.ceil(math.log2(max(f32(maximum / 6), 2.0**-126)))
        if recipe.outer_scale is not None:
            scale = f32(scale * recipe.outer_scale)
        for value in block:
            normalized = f32(value / scale)
            magnitude = min(abs(normalized), 6.0)
            code = min(range(8), key=lambda c: (abs(magnitude - LEVELS[c]), c % 2))
            result.append(f32(math.copysign(LEVELS[code], value) * scale))
    return result


def recipes():
    return [
        codec.FP4Recipe(g, kind, outer_scale=0.02 if kind == "e4m3" else None)
        for g, kind in ((16, "bf16"), (32, "bf16"), (16, "e4m3"), (32, "e4m3"), (32, "e8m0"))
    ]


class KVFP4Tests(unittest.TestCase):
    def test_literal_codes_pack_low_nibble_first_and_keep_signed_zero(self):
        x = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]])
        recipe = codec.FP4Recipe(16, "bf16")
        packed = codec.pack_fp4(x, recipe)
        self.assertEqual(packed.data.tolist(), [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]])
        self.assertEqual(packed.scales.tolist(), [[1.0]])
        self.assertEqual(packed.payload_bytes, 10)
        decoded = codec.unpack_fp4(packed)
        torch.testing.assert_close(decoded, x, rtol=0, atol=0)
        self.assertEqual(decoded.view(torch.int32).tolist(), x.view(torch.int32).tolist())
        torch.testing.assert_close(codec.qdq_fp4(x, recipe), decoded, rtol=0, atol=0)

    def test_scale_formats_have_real_byte_storage_and_explicit_outer_scale(self):
        for group in (16, 32):
            recipe = codec.FP4Recipe(group, "e4m3", outer_scale=0.5)
            x = torch.full((1, group), 3.0)
            packed = codec.pack_fp4(x, recipe)
            self.assertEqual(packed.scales.dtype, torch.uint8)
            self.assertEqual(packed.scales.tolist(), [[0x38]])  # E4M3FN 1.0
            self.assertEqual(packed.data.tolist(), [[0x77] * (group // 2)])
            self.assertEqual(packed.payload_bytes, group // 2 + 1 + 4)
            torch.testing.assert_close(codec.unpack_fp4(packed), x, rtol=0, atol=0)
        mx = codec.FP4Recipe(32, "e8m0")
        packed = codec.pack_fp4(torch.full((1, 32), 7.0), mx)
        self.assertEqual(packed.scales.tolist(), [[128]])  # ceil(log2(7/6)) -> 2
        self.assertEqual(packed.data.tolist(), [[0x66] * 16])  # 3.5 -> 4 (RNE)
        self.assertEqual(packed.payload_bytes, 17)
        self.assertEqual(codec.unpack_fp4(packed).tolist(), [[8.0] * 32])
        for args in ((16, "e8m0"), (16, "e4m3"), (32, "bad")):
            with self.assertRaises(ValueError):
                codec.FP4Recipe(*args)

    def test_all_formats_against_independent_scalar_oracle(self):
        generator = torch.Generator().manual_seed(123)
        for dtype in (torch.float32, torch.bfloat16):
            x = (torch.randn(11, 37, generator=generator) * 3.1).to(dtype)
            before = x.clone()
            for recipe in recipes():
                with self.subTest(dtype=dtype, recipe=recipe):
                    expected = torch.tensor([scalar_reference(row, recipe) for row in x.tolist()]).to(dtype)
                    packed = codec.pack_fp4(x, recipe)
                    decoded = codec.unpack_fp4(packed, dtype)
                    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
                    torch.testing.assert_close(codec.qdq_fp4(x, recipe), expected, rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_all_e2m1_midpoints_round_to_even_including_signed_zero(self):
        row = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        x = torch.tensor([row + [-v for v in row]])
        expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6, -0.0, -1, -1, -2, -2, -4, -4, -6]])
        for kind in ("bf16", "e4m3"):
            recipe = codec.FP4Recipe(16, kind, outer_scale=1.0 if kind == "e4m3" else None)
            result = codec.qdq_fp4(x, recipe)
            self.assertEqual(result.view(torch.int32).tolist(), expected.view(torch.int32).tolist())

    def test_every_positive_e4m3_midpoint_matches_native_cpu_cast(self):
        # Construct amax/6 at the desired midpoint, then compare the scale BYTE.
        midpoints = torch.tensor([(a + b) / 2 for a, b in zip(E4_VALUES[1:-1], E4_VALUES[2:])])
        x = (midpoints * 6)[:, None].expand(-1, 16)
        packed = codec.pack_fp4(x, codec.FP4Recipe(16, "e4m3", outer_scale=1.0))
        expected = (x[:, 0] / 6).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(packed.scales[:, 0], expected, rtol=0, atol=0)

    def test_e8m0_scale_changes_immediately_above_power_of_two(self):
        x = torch.tensor([[6.0] * 32, [torch.nextafter(torch.tensor(6.0), torch.tensor(float("inf"))).item()] * 32])
        packed = codec.pack_fp4(x, codec.FP4Recipe(32, "e8m0"))
        self.assertEqual(packed.scales.tolist(), [[127], [128]])

    def test_zero_groups_scale_clipping_and_wide_dynamic_range(self):
        x = torch.tensor([[0.0] * 32, [2.0**-100] * 32, [2.0**100] * 32])
        for recipe in recipes():
            packed = codec.pack_fp4(x, recipe)
            expected = torch.tensor([scalar_reference(row, recipe) for row in x.tolist()])
            torch.testing.assert_close(codec.unpack_fp4(packed), expected, rtol=0, atol=0)
            torch.testing.assert_close(codec.qdq_fp4(x, recipe), expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(expected).all())
        clipped = codec.pack_fp4(torch.full((1, 16), 3000.0), codec.FP4Recipe(16, "e4m3", 1.0))
        self.assertEqual(clipped.scales.tolist(), [[126]])
        self.assertEqual(codec.unpack_fp4(clipped).tolist(), [[2688.0] * 16])
        for args in ((True, "bf16"), (16.0, "bf16"), (32, "bf16", 1.0)):
            with self.assertRaises(ValueError):
                codec.FP4Recipe(*args)

    def test_padding_empty_and_noncontiguous_shapes(self):
        x = torch.arange(2 * 3 * 17, dtype=torch.float32).reshape(2, 17, 3).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        for recipe in recipes():
            packed = codec.pack_fp4(x, recipe)
            expected = codec.qdq_fp4(x, recipe)
            torch.testing.assert_close(codec.unpack_fp4(packed), expected, rtol=0, atol=0)
            groups = math.ceil(17 / recipe.group_size)
            self.assertEqual(packed.data.numel(), 6 * groups * recipe.group_size // 2)
            empty = x[:0]
            self.assertEqual(codec.unpack_fp4(codec.pack_fp4(empty, recipe)).shape, empty.shape)
            self.assertEqual(codec.qdq_fp4(empty, recipe).shape, empty.shape)

    def test_invalid_data_descriptor_and_recipe_fail_closed(self):
        for kind in recipes():
            for value in (float("nan"), float("inf"), -float("inf")):
                x = torch.ones(2, 32)
                x[0, 3] = value
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    codec.pack_fp4(x, kind)
                y = codec.qdq_fp4(x, kind)
                self.assertFalse(torch.isfinite(y[0, : kind.group_size]).any())
                self.assertTrue(torch.isfinite(y[1]).all())
        for value in (None, 0, -1, float("nan"), float("inf"), 1e-100, True):
            with self.assertRaises(ValueError):
                codec.FP4Recipe(16, "e4m3", value)
        for shape in ((), (2, 0)):
            with self.assertRaises(ValueError):
                codec.qdq_fp4(torch.empty(shape), codec.FP4Recipe(16, "bf16"))
        for recipe in recipes():
            good = codec.pack_fp4(torch.ones(1, 32), recipe)
            for bad in (
                replace(good, data=good.data.float()),
                replace(good, shape=(1, 33)),
                replace(good, scales=torch.zeros_like(good.scales)),
                replace(good, scales=good.scales.float()),
            ):
                with self.assertRaises(ValueError):
                    codec.unpack_fp4(bad)
        mx = codec.pack_fp4(torch.ones(1, 32), codec.FP4Recipe(32, "e8m0"))
        with self.assertRaises(ValueError):
            codec.unpack_fp4(replace(mx, scales=torch.full_like(mx.scales, 255)))
        with self.assertRaisesRegex(ValueError, "overflow"):
            codec.unpack_fp4(codec.pack_fp4(torch.full((1, 32), 3.0e38), codec.FP4Recipe(32, "e8m0")))

    def test_cpu_fullgraph_uses_changed_values_and_shapes(self):
        for recipe in recipes():
            compiled = torch.compile(
                lambda x, recipe=recipe: codec.qdq_fp4(x, recipe), backend="eager", fullgraph=True, dynamic=True
            )
            for rows in (2, 5, 2):
                x = torch.linspace(-5, 5, rows * 37).reshape(rows, 37).to(torch.bfloat16)
                actual = compiled(x)
                expected = codec.unpack_fp4(codec.pack_fp4(x, recipe), x.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
