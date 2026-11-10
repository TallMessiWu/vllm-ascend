# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only numerical contracts, without importing the framework/NPU runtime."""

import importlib.util
import math
import struct
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

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
    def test_mx4_tensor_shift_matches_scalar_oracle_and_rejects_invalid_metadata(self):
        recipe = codec.FP4Recipe(32, "e8m0")
        shift = torch.tensor([23], dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.linspace(-9.1, 6.7, 259).reshape(7, 37).to(dtype)
            actual = codec.qdq_fp4(x, recipe, exponent_shift=shift)
            expected = torch.tensor([scalar_reference(row, recipe) for row in x.float().tolist()]).to(dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for invalid in (
            torch.tensor(23, dtype=torch.int32),
            torch.tensor([23.0]),
            torch.tensor([23, 23], dtype=torch.int32),
            torch.empty(1, dtype=torch.int32, device="meta"),
        ):
            with self.assertRaisesRegex(ValueError, "exponent_shift"):
                codec.qdq_fp4(x, recipe, exponent_shift=invalid)

    def test_mx4_exponent_input_has_independent_integer_storage(self):
        recipe = codec.FP4Recipe(32, "e8m0")
        x = torch.linspace(-6.0, 6.0, 128).reshape(4, 32)
        stages = {}
        actual = codec.qdq_fp4(x, recipe, observer=stages.__setitem__)
        expected = torch.tensor([scalar_reference(row, recipe) for row in x.tolist()])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        raw, bits = stages["raw_scale"], stages["scale_bits"]
        self.assertEqual(bits.dtype, torch.int32)
        self.assertNotEqual(raw.data_ptr(), bits.data_ptr())
        torch.testing.assert_close(bits, raw.view(torch.int32), rtol=0, atol=0)
        saved = bits.clone()
        raw.zero_()
        torch.testing.assert_close(bits, saved, rtol=0, atol=0)

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

    def test_e4m3_scale_avoids_scalar_reciprocal_lowering(self):
        # Model the observed scalar division path using a rounded FP32
        # reciprocal. This is CPU fault injection, not an NPU implementation.
        original_div = torch.Tensor.__truediv__
        denominator = f32(6 * f32(0.02))

        def reciprocal_scalar_div(x, other):
            if isinstance(other, (int, float)) and f32(other) == denominator:
                return x * f32(1.0 / denominator)
            return original_div(x, other)

        for group in (16, 32):
            recipe = codec.FP4Recipe(group, "e4m3", 0.02)
            x = torch.tensor([[6.0] * group, [1.5] * group])
            expected = codec.qdq_fp4(x, recipe)
            with patch.object(torch.Tensor, "__truediv__", reciprocal_scalar_div):
                packed = codec.pack_fp4(x, recipe)
                actual = codec.qdq_fp4(x, recipe)
            self.assertEqual(packed.scales.tolist(), [[100], [84]])  # scales 48 and 12, not 52 and 13
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_e4m3_scale_midpoint_neighbors_are_not_snapped(self):
        centers = torch.tensor([6.0, 1.5])
        maxima = torch.stack(
            (
                torch.nextafter(centers, torch.zeros_like(centers)),
                centers,
                torch.nextafter(centers, torch.full_like(centers, float("inf"))),
            ),
            dim=-1,
        ).flatten()
        for group in (16, 32):
            recipe = codec.FP4Recipe(group, "e4m3", 0.02)
            x = maxima[:, None].expand(-1, group)
            packed = codec.pack_fp4(x, recipe)
            self.assertEqual(packed.scales[:, 0].tolist(), [100, 100, 101, 84, 84, 85])
            expected = torch.tensor([scalar_reference(row, recipe) for row in x.tolist()])
            torch.testing.assert_close(codec.qdq_fp4(x, recipe), expected, rtol=0, atol=0)

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


class KVFP4CompileContractTests(unittest.TestCase):
    """CPU tests for test-vector math and compiler option plumbing only."""

    @classmethod
    def setUpClass(cls):
        source = SOURCE.parents[2] / "tests/e2e/pull_request/one_card/test_kv_fp4_codec.py"
        spec = importlib.util.spec_from_file_location("kv_fp4_compile_contract", source)
        cls.device_tests = importlib.util.module_from_spec(spec)
        quantization = types.ModuleType("vllm_ascend.quantization")
        quantization.kv_fp4 = codec
        with patch.dict(
            sys.modules,
            {
                "torch_npu": types.ModuleType("torch_npu"),
                "vllm_ascend": types.ModuleType("vllm_ascend"),
                "vllm_ascend.quantization": quantization,
            },
        ):
            spec.loader.exec_module(cls.device_tests)

    def test_scale_trace_has_literal_bits_and_preserves_power_neighbors(self):
        x = torch.tensor([0.5, 1.0, 2.0])
        bits, exponent, scale = self.device_tests.mx4_scale_trace(x)
        self.assertEqual(bits.tolist(), [0x3F000000, 0x3F800000, 0x40000000])
        self.assertEqual(exponent.tolist(), [126, 127, 128])
        torch.testing.assert_close(scale, x, rtol=0, atol=0)
        above = torch.nextafter(x, torch.full_like(x, float("inf")))
        torch.testing.assert_close(self.device_tests.mx4_scale_trace(above)[2], 2 * x, rtol=0, atol=0)
        zeros = torch.zeros_like(x)
        torch.testing.assert_close(
            self.device_tests.mx4_scale_trace(zeros)[2], torch.full_like(x, 2.0**-126), rtol=0, atol=0
        )

    def test_e8m0_floor_survives_exp2_underflow_in_qdq_pack_and_unpack(self):
        recipe = codec.FP4Recipe(32, "e8m0")
        original_exp2 = torch.exp2

        def flush_floor(x):
            return torch.where(x == -126, 0.0, original_exp2(x))

        for dtype in (torch.float32, torch.bfloat16):
            values = torch.tensor([0.0, 6 * codec.FP32_MIN_NORMAL, -6 * codec.FP32_MIN_NORMAL], dtype=dtype)
            pattern = (torch.tensor([0, 1, 2, 3, 4, 6, 2, 4]) * codec.FP32_MIN_NORMAL).to(dtype).repeat(4)[None]
            host = torch.cat((values[:, None].expand(-1, 32), pattern, -pattern))
            # Mixed normal-valued levels catch bad normalization as well as a
            # zero scale. Literal payloads, not the active codec, are the oracle.
            packed = codec.PackedFP4(
                torch.tensor(
                    [[0x00] * 16, [0x77] * 16, [0xFF] * 16, [0x20, 0x54, 0x76, 0x64] * 4, [0xA8, 0xDC, 0xFE, 0xEC] * 4],
                    dtype=torch.uint8,
                ),
                torch.tensor([[127], [1], [1], [1], [1]], dtype=torch.uint8),
                tuple(host.shape),
                recipe,
            )
            with patch.object(torch, "exp2", flush_floor):
                self.assertEqual(torch.exp2(torch.tensor([-126.0])).item(), 0.0)  # negative control
                torch.testing.assert_close(codec.qdq_fp4(host, recipe), host, rtol=0, atol=0)
                actual = codec.pack_fp4(host, recipe)
                torch.testing.assert_close(actual.scales, packed.scales, rtol=0, atol=0)
                torch.testing.assert_close(actual.data, packed.data, rtol=0, atol=0)
                torch.testing.assert_close(codec.unpack_fp4(packed, dtype), host, rtol=0, atol=0)

    def test_reference_e8m0_scale_bytes_decode_to_literal_fp32_bits(self):
        recipe = codec.FP4Recipe(32, "e8m0")
        # Independent arithmetic oracle, including transposed input and both
        # producer (int32) and reader (uint8) exponent types.
        expected = torch.tensor([math.ldexp(1.0, e - 127) for e in range(1, 255)]).reshape(2, 127).T
        for dtype in (torch.uint8, torch.int32):
            scales = torch.arange(1, 255, dtype=dtype).reshape(2, 127).T
            before = scales.clone()
            actual = codec._scale_values(scales, recipe)
            torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(codec._scale_storage(actual, recipe), scales.to(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(scales, before, rtol=0, atol=0)
            scales.zero_()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)  # owns new storage, not an input alias

    def test_exp2_neighbor_error_cannot_change_scale_bytes_or_fp4_midpoints(self):
        recipe = codec.FP4Recipe(32, "e8m0")
        reported = torch.tensor([18, 19, 20, 24], dtype=torch.uint8)
        expected_scales = torch.tensor([math.ldexp(1.0, int(e) - 127) for e in reported])
        original_exp2 = torch.exp2

        def approximate_exp2(x):
            exact = original_exp2(x)
            affected = (x == -109) | (x == -108) | (x == -107) | (x == -103)
            return torch.where(affected, torch.nextafter(exact, torch.zeros_like(exact)), exact)

        row = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]).repeat(4)
        rounded = torch.tensor([0, 1, 1, 2, 2, 4, 4, 6]).repeat(4)
        x = expected_scales[:, None] * row
        expected_values = expected_scales[:, None] * rounded
        literal = codec.PackedFP4(
            torch.tensor([[0x20, 0x42, 0x64, 0x76] * 4] * 4, dtype=torch.uint8),
            reported[:, None],
            tuple(x.shape),
            recipe,
        )
        with patch.object(torch, "exp2", approximate_exp2):
            legacy = torch.exp2(reported.float() - 127)
            self.assertEqual(codec._scale_storage(legacy, recipe).tolist(), [17, 18, 19, 23])
            torch.testing.assert_close(codec._scale_values(reported, recipe), expected_scales, rtol=0, atol=0)
            packed = codec.pack_fp4(x, recipe)
            torch.testing.assert_close(packed.scales, literal.scales, rtol=0, atol=0)
            torch.testing.assert_close(packed.data, literal.data, rtol=0, atol=0)
            torch.testing.assert_close(codec.qdq_fp4(x, recipe), expected_values, rtol=0, atol=0)
            torch.testing.assert_close(codec.unpack_fp4(literal), expected_values, rtol=0, atol=0)

    def test_opaque_out_buffer_survives_dynamo_and_aot_without_inlining_codec(self):
        from torch._dynamo.backends.registry import lookup_backend

        self.addCleanup(torch._dynamo.reset)
        recipe = codec.FP4Recipe(32, "e8m0")
        opaque = self.device_tests.make_opaque_qdq(lambda x: codec.qdq_fp4(x, recipe), "cpu_contract")
        for backend_name in ("eager", "aot_eager"):
            graphs = []
            backend = lookup_backend(backend_name)

            def checked_backend(graph, inputs, graphs=graphs, backend=backend):
                graphs.append(graph.code)
                return backend(graph, inputs)

            compiled = torch.compile(opaque, backend=checked_backend, fullgraph=True, dynamic=True)
            host = torch.linspace(-5, 5, 3 * 32).reshape(3, 32).bfloat16()
            backing = torch.zeros((3, 3, 32), dtype=host.dtype)
            x, output = backing[:, 1], torch.empty_like(host)
            self.assertFalse(x.is_contiguous())
            for factor in (0.0, 1.0, -0.7):
                changed = (host.float() * factor).to(host.dtype)
                x.copy_(changed)
                actual = compiled(x, output)
                self.assertEqual(actual.data_ptr(), output.data_ptr())
                torch.testing.assert_close(output, codec.qdq_fp4(changed, recipe), rtol=0, atol=0)
                torch.testing.assert_close(x, changed, rtol=0, atol=0)
            self.assertTrue(graphs)
            for graph in graphs:
                self.assertIn("kv_fp4_diagnostic.cpu_contract", graph)
                self.assertNotIn("exp2", graph)
                self.assertNotIn("bitwise", graph)
            torch._dynamo.reset()

    def test_compiler_options_match_service_without_real_npu_import(self):
        for layout in ("compiler_config", "npugraphex_config"):
            nge = types.ModuleType("npugraph_ex")
            nge.__path__ = []
            nge.CompilerConfig = Mock(return_value=object())
            nge.get_npu_backend = Mock(return_value=object())
            configs = types.ModuleType("npugraph_ex.configs")
            configs.__path__ = []
            options = types.ModuleType(f"npugraph_ex.configs.{layout}")
            options._process_kwargs_options = Mock()
            modules = {
                "npugraph_ex": nge,
                "npugraph_ex.configs": configs,
                "npugraph_ex.configs.compiler_config": None,
                "npugraph_ex.configs.npugraphex_config": None,
                f"npugraph_ex.configs.{layout}": options,
            }
            npu = types.SimpleNamespace(set_compile_mode=Mock())
            with patch.dict(sys.modules, modules), patch.object(torch, "npu", npu, create=True):
                actual = self.device_tests.compiler_backend()
            self.assertIs(actual, nge.get_npu_backend.return_value)
            npu.set_compile_mode.assert_called_once_with(jit_compile=False)
            options._process_kwargs_options.assert_called_once_with(
                nge.CompilerConfig.return_value,
                {"options": dict(force_eager=True, inplace_pass=False, clone_input=False, clone_output=False)},
            )


if __name__ == "__main__":
    unittest.main()
