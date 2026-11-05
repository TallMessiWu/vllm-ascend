# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU numerical/reference tests, deliberately independent of the NPU runtime.

Run with --confcutdir=tests/ut/quantization to avoid framework-wide UT fixtures.
This is not an ACL graph or NPU kernel test.
"""

import importlib.util
import json
import math
import struct
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("nvfp4_a4_reference", ROOT / "vllm_ascend/quantization/nvfp4_a4.py")
qdq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qdq  # Dynamo resolves the numerical module's globals by module name.
spec.loader.exec_module(qdq)


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp8_value(code):
    exponent, mantissa = code >> 3, code & 7
    return mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)


def oracle(values, global_scale):
    levels = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    outputs = []
    for start in range(0, len(values), 16):
        block = values[start : start + 16]
        scale = fp32(max(map(abs, block)) / fp32(6 * global_scale))
        scale = 1.0 if scale == 0 else min(448, max(2**-9, scale))
        code = min(range(1, 127), key=lambda c: (abs(fp8_value(c) - scale), c & 1))
        combined = fp32(fp8_value(code) * global_scale)
        for value in block:
            normalized = fp32(value / combined)
            code4 = min(range(8), key=lambda c: (abs(levels[c] - abs(normalized)), c & 1))
            signed = -levels[code4] if value < 0 and code4 else levels[code4]
            outputs.append(fp32(signed * combined))
    return outputs


class NVFP4A4Tests(unittest.TestCase):
    def test_e4m3_rounding_all_midpoints_matches_independent_fp8_codebook(self):
        values = [0, 1e-20, 448, 1000]
        for code in range(1, 126):
            middle = (fp8_value(code) + fp8_value(code + 1)) / 2
            values.extend((middle, middle * (1 - 1e-6), middle * (1 + 1e-6)))
        x = torch.tensor(values, dtype=torch.float32)
        actual = qdq.round_e4m3_positive(x)
        expected = [
            fp8_value(min(range(1, 127), key=lambda c: (abs(fp8_value(c) - min(448, max(2**-9, v))), c & 1)))
            for v in x.tolist()
        ]
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)

    def test_qdq_values_and_ties_match_scalar_oracle_and_keep_input(self):
        generator = torch.Generator().manual_seed(123)
        x = (torch.randn((9, 64), generator=generator) * 4).to(torch.bfloat16)
        x[0, :16] = torch.tensor(
            [0, -0.1, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.25, -1.75, -2.5, -6]
        )
        x[1].zero_()
        before = x.clone()
        scales = torch.tensor([1.0, 0.5, 0.01, 1e-3, 2.0, 0.7, 0.01, 1.0, 0.1])
        actual = qdq.nvfp4_activation_qdq(x, scales.reshape(-1, 1, 1))
        expected = torch.tensor([oracle(row.tolist(), scale.item()) for row, scale in zip(x, scales)], dtype=x.dtype)
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
        self.assertTrue(torch.equal(x, before))
        self.assertFalse(torch.equal(actual, x))

    def test_expert_scale_mapping_empty_experts_tail_padding_and_live_groups(self):
        scales = torch.tensor([0.5, 1.0, 2.0, 4.0])
        counts = torch.tensor([2, 0, 3, 0])
        expected = torch.tensor([0.5, 0.5, 2, 2, 2, 4, 4]).reshape(-1, 1, 1)
        torch.testing.assert_close(qdq.routed_global_scales(scales, counts, 1, 7), expected)
        torch.testing.assert_close(qdq.routed_global_scales(scales, counts.cumsum(0), 0, 7), expected)
        changed = qdq.routed_global_scales(scales, torch.tensor([0, 2, 1, 2]), 1, 5)
        torch.testing.assert_close(changed.flatten(), torch.tensor([1.0, 1.0, 2.0, 4.0, 4.0]))
        self.assertEqual(qdq.routed_global_scales(scales, counts, 1, 0).shape, (0, 1, 1))
        with self.assertRaises(ValueError):
            qdq.routed_global_scales(scales, counts, 2, 7)

    def test_cpu_dynamo_fullgraph_uses_current_inputs_groups_and_shapes(self):
        scales = torch.tensor([0.03, 0.07])

        def apply(x, groups):
            row_scales = qdq.routed_global_scales(scales, groups, 1, x.shape[0])
            return qdq.nvfp4_activation_qdq(x, row_scales)

        compiled = torch.compile(apply, backend="eager", fullgraph=True, dynamic=True)
        generator = torch.Generator().manual_seed(39)
        for rows, first_count in ((32, 7), (1041, 0), (32, 32)):
            x = torch.randn((rows, 32), generator=generator).to(torch.bfloat16)
            groups = torch.tensor([first_count, rows - first_count])
            torch.testing.assert_close(compiled(x, groups), apply(x, groups), rtol=0, atol=0)

    def test_tiling_preserves_values_and_per_row_scales(self):
        generator = torch.Generator().manual_seed(17)
        x = torch.randn((2051, 32), generator=generator).to(torch.bfloat16)
        scales = torch.linspace(0.01, 0.1, x.shape[0]).reshape(-1, 1, 1)
        actual = qdq.nvfp4_activation_qdq(x, scales)
        for row in (0, 1023, 1024, 2048, 2050):
            expected = torch.tensor(oracle(x[row].tolist(), scales[row].item()), dtype=x.dtype)
            torch.testing.assert_close(actual[row], expected, rtol=0, atol=0)
        scalar = torch.tensor(0.05)
        whole = qdq.nvfp4_activation_qdq(x, scalar)
        pieces = torch.cat([qdq.nvfp4_activation_qdq(part, scalar) for part in x.split(400)])
        torch.testing.assert_close(whole, pieces, rtol=0, atol=0)

    def test_empty_invalid_shape_nonfinite_and_repeated_tensor_inputs(self):
        scale = torch.tensor(1.0)
        self.assertEqual(qdq.nvfp4_activation_qdq(torch.zeros(0, 32), scale).shape, (0, 32))
        with self.assertRaises(ValueError):
            qdq.nvfp4_activation_qdq(torch.zeros(2, 17), scale)
        with self.assertRaises(ValueError):
            qdq.nvfp4_activation_qdq(torch.zeros(2, 16), scale.double())
        with self.assertRaises(ValueError):
            qdq.nvfp4_activation_qdq(torch.zeros(2, 16), torch.ones(3, 1, 1))
        x = torch.zeros(2, 16)
        x[0, 0], x[1, 0] = math.nan, math.inf
        out = qdq.nvfp4_activation_qdq(x, scale)
        self.assertTrue(torch.isnan(out[0, 0]))
        self.assertTrue(torch.isinf(out[1, 0]))
        # Device tensor math must use the new values on each call, never a cached Python result.
        x = torch.full((2, 32), 0.7)
        first = qdq.nvfp4_activation_qdq(x, scale)
        x.mul_(4)
        second = qdq.nvfp4_activation_qdq(x, scale)
        self.assertFalse(torch.equal(first, second))

    def test_dense_and_expert_methods_quantize_both_inputs_before_router_scaling(self):
        class LinearParent:
            def process_weights_after_loading(self, layer):
                pass

            def apply(self, layer, x, bias=None):
                return x @ layer.weight.T

        class MoEParent:
            def __init__(self, moe=None):
                self.moe = moe

            def process_weights_after_loading(self, layer):
                pass

            def apply_gmm1(self, inputs):
                return inputs.hidden_states

            def apply_act_quant(self, inputs, x):
                return x * inputs.topk_scales, None

        modules = {}
        for name, members in {
            "vllm.logger": {"logger": MagicMock()},
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": LinearParent},
            "vllm_ascend.ops.fused_moe.routed_experts": {"AscendUnquantizedFusedMoEMethod": MoEParent},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(members)
            modules[name] = module
        modules["vllm_ascend.quantization.nvfp4_a4"] = qdq
        with patch.dict(sys.modules, modules):
            method_spec = importlib.util.spec_from_file_location(
                "a4_methods", ROOT / "vllm_ascend/quantization/methods/nvfp4_a4.py"
            )
            methods = importlib.util.module_from_spec(method_spec)
            method_spec.loader.exec_module(methods)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.eye(16, dtype=torch.bfloat16))
        linear = methods.NVFP4A4LinearMethod(0.07)
        x = torch.linspace(-2.3, 2.4, 32).reshape(2, 16).to(torch.bfloat16)
        with self.assertRaises(AttributeError):
            linear.apply(layer, x)  # No lazy initialization inside capture.
        linear.process_weights_after_loading(layer)
        expected = torch.tensor([oracle(row.tolist(), fp32(0.07)) for row in x], dtype=x.dtype)
        torch.testing.assert_close(linear.apply(layer, x), expected, rtol=0, atol=0)
        moe_layer = torch.nn.Module()
        moe_layer.w13_weight = torch.nn.Parameter(torch.ones(2, 16, 32, dtype=x.dtype))
        before = moe_layer.w13_weight.detach().clone()
        moe = methods.NVFP4A4MoEMethod(
            types.SimpleNamespace(num_experts=2, num_local_experts=2), [0.03, 0.07], [0.05, 0.11]
        )
        moe.process_weights_after_loading(moe_layer)

        @dataclass(frozen=True)
        class Inputs:
            hidden_states: torch.Tensor
            layer: torch.nn.Module
            group_list: torch.Tensor
            topk_scales: torch.Tensor
            group_list_type: int = 1
            dynamic_scale: object = None
            dynamic_eplb: bool = False
            fusion: bool = False
            lora_context: object = None

        inputs = Inputs(x, moe_layer, torch.tensor([1, 1]), torch.tensor([[0.3], [0.7]], dtype=x.dtype))
        expected_gate = torch.tensor(
            [oracle(row.tolist(), fp32(scale)) for row, scale in zip(x, [0.03, 0.07])], dtype=x.dtype
        )
        torch.testing.assert_close(moe.apply_gmm1(inputs), expected_gate, rtol=0, atol=0)
        expected_down = (
            torch.tensor([oracle(row.tolist(), fp32(scale)) for row, scale in zip(x, [0.05, 0.11])], dtype=x.dtype)
            * inputs.topk_scales
        )
        actual, scale = moe.apply_act_quant(inputs, x)
        torch.testing.assert_close(actual, expected_down, rtol=0, atol=0)
        self.assertIsNone(scale)
        self.assertTrue(torch.equal(moe_layer.w13_weight, before))
        self.assertTrue(torch.equal(inputs.hidden_states, x))

    def test_config_routes_targets_and_rejects_unsupported_runtime_without_mutating_carrier(self):
        registry = {}

        def register(name):
            def decorator(cls):
                registry[name] = cls
                return cls

            return decorator

        class ConfigBase:
            def __init__(self):
                self.packed_modules_mapping = {}

        class DenseMethod:
            def __init__(self, scale):
                self.scale = scale

        class ExpertMethod:
            def __init__(self, *args):
                self.args = args

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            carrier = Path(tmp)
            (carrier / ".nvfp4_source_metadata").mkdir()
            source_config = {"model_type": "qwen3_moe", "source_identity": "fixture"}
            (carrier / ".nvfp4_source_metadata/source-config.json").write_text(json.dumps(source_config))
            (carrier / "config.json").write_text('{"model_type":"qwen3_moe"}')
            config = types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(
                    tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1
                ),
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(model_type="qwen3_moe"), dtype=torch.bfloat16, model=str(carrier)
                ),
                cache_config=types.SimpleNamespace(cache_dtype="bfloat16"),
                lora_config=None,
                speculative_config=None,
            )
            ascend = types.SimpleNamespace(enable_fused_mc2=0, eplb_config=types.SimpleNamespace(dynamic_eplb=False))
            modules = {"vllm_ascend.quantization.nvfp4_a4": qdq}
            for name, members in {
                "vllm.config": {"get_current_vllm_config": lambda: config},
                "vllm.model_executor.layers.linear": {"LinearBase": torch.nn.Linear},
                "vllm.model_executor.layers.quantization": {"register_quantization_config": register},
                "vllm.model_executor.layers.quantization.base_config": {"QuantizationConfig": ConfigBase},
                "vllm_ascend.quantization.utils": {"is_fused_moe_layer": lambda layer: hasattr(layer, "moe_config")},
                "vllm_ascend.ascend_config": {"get_ascend_config": lambda: ascend},
                "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": ConfigBase},
                "vllm_ascend.quantization.methods.nvfp4_a4": {
                    "NVFP4A4LinearMethod": DenseMethod,
                    "NVFP4A4MoEMethod": ExpertMethod,
                },
            }.items():
                module = types.ModuleType(name)
                module.__dict__.update(members)
                modules[name] = module
            with patch.dict(sys.modules, modules):
                cfg_spec = importlib.util.spec_from_file_location(
                    "a4_config", ROOT / "vllm_ascend/quantization/configs/nvfp4_a4_config.py"
                )
                cfg_module = importlib.util.module_from_spec(cfg_spec)
                cfg_spec.loader.exec_module(cfg_module)
                scales = types.SimpleNamespace(
                    config=source_config, linear=lambda prefix: 0.1, experts=lambda prefix, proj: [0.2, 0.3]
                )
                with patch.object(cfg_module, "ActivationScales", return_value=scales):
                    cls = registry["nvfp4_a4_qdq"]
                    mode = cls.from_config_dict_json('{"source":"fixture"}')
                    linear = torch.nn.Linear(16, 16)
                    self.assertIsInstance(
                        mode.get_quant_method(linear, "model.layers.0.self_attn.qkv_proj"), DenseMethod
                    )
                    self.assertIsInstance(mode.get_quant_method(linear, "model.layers.0.mlp.gate"), ConfigBase)
                    self.assertIsNone(mode.get_quant_method(torch.nn.Embedding(8, 16), "lm_head"))
                    expert = types.SimpleNamespace(moe_config=object())
                    self.assertIsInstance(mode.get_quant_method(expert, "model.layers.0.mlp.experts"), ExpertMethod)
                    with self.assertRaises(ValueError):
                        mode.get_quant_method(linear, "model.unrecognized")
                    for owner, attribute, value in (
                        (config.parallel_config, "tensor_parallel_size", 2),
                        (ascend, "enable_fused_mc2", 1),
                        (config.cache_config, "cache_dtype", "fp8"),
                    ):
                        old = getattr(owner, attribute)
                        setattr(owner, attribute, value)
                        with self.assertRaises(ValueError):
                            cls(source="fixture").get_quant_method(linear, "model.layers.0.self_attn.o_proj")
                        setattr(owner, attribute, old)
                    (carrier / ".nvfp4_to_bf16.incomplete").write_text("INCOMPLETE")
                    with self.assertRaises(ValueError):
                        cls(source="fixture").get_quant_method(linear, "model.layers.0.self_attn.o_proj")
            self.assertEqual(json.loads((carrier / "config.json").read_text()), {"model_type": "qwen3_moe"})

    def test_scale_loader_reads_only_scalar_metadata_and_rejects_unequal_fusions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            quant = {"num_bits": 4, "type": "float", "group_size": 16}
            config = {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 1,
                "num_experts": 2,
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "config_groups": {"g": {"weights": quant, "input_activations": quant}},
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            values = {
                f"model.layers.0.self_attn.{p}.input_scale": torch.tensor(0.1)
                for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
            for e in range(2):
                values.update(
                    {
                        f"model.layers.0.mlp.experts.{e}.{p}.input_scale": torch.tensor((e + 1) * 0.2)
                        for p in ("gate_proj", "up_proj", "down_proj")
                    }
                )
            save_file(values, root / "calibration.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(values, "calibration.safetensors")})
            )
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            source = qdq.ActivationScales(str(root))
            self.assertAlmostEqual(source.linear("model.layers.0.self_attn.qkv_proj"), 0.1)
            torch.testing.assert_close(
                torch.tensor(source.experts("model.layers.0.mlp.experts.routed_experts", "down_proj")),
                torch.tensor([0.2, 0.4]),
            )
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            for bad in (float("nan"), 0.0, -0.1, 0.3):
                values["model.layers.0.self_attn.k_proj.input_scale"] = torch.tensor(bad)
                save_file(values, root / "calibration.safetensors")
                with self.assertRaises(ValueError):
                    qdq.ActivationScales(str(root))


if __name__ == "__main__":
    unittest.main()
