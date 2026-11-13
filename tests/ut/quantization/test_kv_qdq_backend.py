# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Test the backend seam with real QDQ math and a fake cache-writer parent.

Does not import real vLLM/Ascend runtime, torch_npu, or test any hardware kernel.
"""

import ast
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
package = types.ModuleType("kv_qdq_backend_numerics")
package.__path__ = [str(ROOT / "vllm_ascend/quantization")]
sys.modules[package.__name__] = package
codec = importlib.import_module(package.__name__ + ".kv_fp4")
config = importlib.import_module(package.__name__ + ".kv_qdq_config")


class CacheWriterParent:
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
        **kwargs,
    ):
        self.num_heads, self.head_size, self.num_kv_heads = num_heads, head_size, num_kv_heads

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        kv_cache[0].copy_(key)
        kv_cache[1].copy_(value)
        return query, key, value, output

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        kv_cache[0].copy_(key)
        kv_cache[1].copy_(value)


def load_backend(target="kv", name="g16-b", diagnostic_copy=False, diagnostic_trace=False, scale_selection=None):
    runtime = NS(
        model_config=NS(dtype=torch.bfloat16, hf_config=NS(model_type="qwen3_moe")),
        cache_config=NS(cache_dtype="bfloat16", enable_prefix_caching=False),
        parallel_config=NS(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        quant_config=None,
        additional_config={"kv_fp4_qdq": {"format": name, "target": target}},
    )
    if scale_selection is not None:
        runtime.additional_config["kv_fp4_qdq"]["mx4_scale_selection"] = scale_selection
    if diagnostic_copy:
        runtime.additional_config["kv_fp4_qdq"]["diagnostic_copy"] = diagnostic_copy
    if diagnostic_trace:
        runtime.additional_config["kv_fp4_qdq"]["diagnostic_trace"] = diagnostic_trace
    modules = {
        "vllm.config": NS(get_current_vllm_config=lambda: runtime),
        "vllm.logger": NS(logger=NS(info_once=lambda *a, **kw: None)),
        "vllm_ascend.attention.attention_v1": NS(
            AscendAttentionBackend=type("Backend", (), {}), AscendAttentionBackendImpl=CacheWriterParent
        ),
        "vllm_ascend.quantization.kv_fp4": codec,
        "vllm_ascend.quantization.kv_qdq_config": config,
    }
    source = ROOT / "vllm_ascend/attention/kv_fp4_qdq.py"
    spec = importlib.util.spec_from_file_location("kv_qdq_backend_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class KVQDQBackendTests(unittest.TestCase):
    def test_mse_kv_cache_and_direct_prefill_share_selected_values_across_tiles(self):
        module = load_backend(name="mx4", scale_selection="mse")
        backend = module.AscendKVFP4QDQImpl(8, 128, 0.1, 4, None, None, "bfloat16", None, "decoder", None)
        for tokens in (1, 257, 1):
            for small, rounded in ((0.375, 0.5), (4.0, 4.0), (0.0, 0.0)):
                key = torch.full((tokens, 4, 128), small, dtype=torch.bfloat16)
                key[..., ::32] = 6.03125
                value = -key
                before = key.clone()
                expected = torch.full_like(key, rounded)
                expected[..., ::32] = 6.0
                query = torch.zeros(tokens, 8, 128, dtype=key.dtype)
                cache = [torch.zeros_like(key), torch.zeros_like(value)]
                with patch.object(torch, "npu", NS(is_current_stream_capturing=lambda: False), create=True):
                    q, k, v, _ = backend.reshape_and_cache(query, key, value, cache, NS(), query.clone())
                self.assertIs(q, query)
                self.assertEqual(backend._mx4_shift.tolist(), [23])
                for actual, target in ((k, expected), (v, -expected), (cache[0], expected), (cache[1], -expected)):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)
                torch.testing.assert_close(key, before, rtol=0, atol=0)

    def create(self, target):
        module = load_backend(target)
        return module.AscendKVFP4QDQImpl(2, 16, 0.25, 1, None, None, "bfloat16", None, "decoder", None)

    def test_ordinary_mx4_reuses_tensor_shift_through_capture_and_tiled_prefill(self):
        class TensorShiftOnly(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.shifts = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if "right_shift" in str(func) or "rshift" in str(func):
                    if not isinstance(args[1], torch.Tensor):
                        raise AssertionError("scalar right shift reached the model path")
                    self.shifts += 1
                return func(*args, **(kwargs or {}))

        module = load_backend(name="mx4")  # diagnostics remain off
        backend = module.AscendKVFP4QDQImpl(8, 128, 0.1, 4, None, None, "bfloat16", None, "decoder", None)
        capturing = [True]
        key = torch.linspace(-3, 6, 512).reshape(1, 4, 128).bfloat16()
        recipe = codec.FP4Recipe(32, "e8m0")
        with (
            torch.inference_mode(),
            patch.object(torch, "npu", NS(is_current_stream_capturing=lambda: capturing[0]), create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "eager initialization"):
                backend._transform_kv(key, -key)
            capturing[0] = False
            backend._transform_kv(key, -key)
            shift = backend._mx4_shift
            self.assertTrue(shift.is_inference())
            self.assertEqual(shift.tolist(), [23])
            self.assertIsNone(backend.kv_trace)
            address = shift.data_ptr()
            for tokens in (1, 300):
                current = key.repeat(tokens, 1, 1) * 2.3
                expected = codec.qdq_fp4(current, recipe)
                capturing[0] = True
                with (
                    TensorShiftOnly() as dispatch,
                    patch.object(torch, "tensor", side_effect=AssertionError("capture allocated shift")),
                ):
                    actual_key, actual_value = backend._transform_kv(current, -current)
                self.assertGreater(dispatch.shifts, 0)
                torch.testing.assert_close(actual_key, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_value, -expected, rtol=0, atol=0)
                self.assertIs(backend._mx4_shift, shift)
                self.assertEqual(backend._mx4_shift.data_ptr(), address)
            with self.assertRaisesRegex(AssertionError, "scalar right shift"), TensorShiftOnly():
                codec.qdq_fp4(key, recipe)  # negative control for the reference path

    def test_trace_is_preallocated_in_warmup_and_only_records_during_capture(self):
        trace_spec = importlib.util.spec_from_file_location(
            "backend_trace_test", ROOT / "vllm_ascend/attention/kv_fp4_trace.py"
        )
        trace_module = importlib.util.module_from_spec(trace_spec)
        with patch.dict(sys.modules, {"vllm_ascend.quantization.kv_fp4": codec}):
            trace_spec.loader.exec_module(trace_module)
        module = load_backend(name="mx4", diagnostic_trace=True)
        backend = module.AscendKVFP4QDQImpl(2, 16, 0.25, 1, None, None, "bfloat16", None, "decoder", None)
        backend._layer_name = module.TRACE_DETAIL_LAYER
        q = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
        k = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 16)
        cache = [torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16) for _ in range(2)]
        backend.key_cache, backend.value_cache = cache
        backend.use_bnsd_kv_cache = False
        metadata = NS(slot_mapping=torch.tensor([0], dtype=torch.int32))
        capturing = [False]
        with (
            patch.dict(sys.modules, {"vllm_ascend.attention.kv_fp4_trace": trace_module}),
            patch.object(torch, "npu", NS(is_current_stream_capturing=lambda: capturing[0]), create=True),
        ):
            backend.reshape_and_cache(q, k, -k, cache, metadata, q.clone())
            self.assertEqual(int(backend.kv_trace.count[0]), 0)
            retained = backend.kv_trace
            capturing[0] = True
            backend.reshape_and_cache(q, k * 2.3, -k * 2.3, cache, metadata, q.clone())
            self.assertIs(backend.kv_trace, retained)
            self.assertEqual(retained.report(backend.kv_qdq_settings)["status"], "GREEN")
            self.assertEqual(retained.report(backend.kv_qdq_settings)["k"]["detail"]["first_divergence"], "none")
            capturing[0] = False
            backend.reshape_and_cache(q, k * 0.7, -k * 0.7, cache, metadata, q.clone())
            self.assertEqual(int(retained.count[0]), 1)
            backend.kv_trace = None
            capturing[0] = True
            with self.assertRaisesRegex(RuntimeError, "warmup"):
                backend.reshape_and_cache(q, k, -k, cache, metadata, q.clone())

    def test_cache_writer_and_direct_prefill_receive_identical_qdq_values(self):
        q = torch.randn(2, 2, 16).bfloat16()
        k = torch.full((2, 1, 16), 0.75, dtype=torch.bfloat16)
        k[:, :, -1] = 6
        v = -k
        k_before, v_before = k.clone(), v.clone()
        expected_k = codec.qdq_fp4(k, codec.FP4Recipe(16, "bf16"))
        expected_v = codec.qdq_fp4(v, codec.FP4Recipe(16, "bf16"))
        for target in ("off", "k", "v", "kv"):
            backend = self.create(target)
            cache = [torch.zeros_like(k), torch.zeros_like(v)]
            output = torch.empty_like(q)
            out_q, out_k, out_v, out = backend.reshape_and_cache(q, k, v, cache, NS(), output)
            self.assertIs(out_q, q)
            self.assertIs(out, output)
            torch.testing.assert_close(out_k, expected_k if target in ("k", "kv") else k, rtol=0, atol=0)
            torch.testing.assert_close(out_v, expected_v if target in ("v", "kv") else v, rtol=0, atol=0)
            torch.testing.assert_close(cache[0], out_k, rtol=0, atol=0)
            torch.testing.assert_close(cache[1], out_v, rtol=0, atol=0)
            if target == "off":
                self.assertIs(out_k, k)
                self.assertIs(out_v, v)
        torch.testing.assert_close(k, k_before, rtol=0, atol=0)
        torch.testing.assert_close(v, v_before, rtol=0, atol=0)

    def test_diagnostic_copy_preserves_bits_and_replaces_only_selected_sides(self):
        q = torch.zeros(2, 2, 16, dtype=torch.bfloat16)
        k = torch.arange(64, dtype=torch.bfloat16).reshape(2, 1, 32)[..., ::2]
        v = (-torch.arange(64, dtype=torch.bfloat16)).reshape(2, 1, 32)[..., ::2]
        k[0, 0, 0] = -0.0
        k[0, 0, 1] = float("nan")
        v[1, 0, -1] = float("inf")
        before = (k.clone(), v.clone())
        self.assertFalse(k.is_contiguous())
        self.assertFalse(v.is_contiguous())
        for target in ("off", "k", "v", "kv"):
            module = load_backend(target, diagnostic_copy=True)
            backend = module.AscendKVFP4QDQImpl(2, 16, 0.25, 1, None, None, "bfloat16", None, "decoder", None)
            cache = [torch.zeros_like(k), torch.zeros_like(v)]
            with patch.object(module, "_qdq_tiled", side_effect=AssertionError("copy control must not call QDQ")):
                _, out_k, out_v, _ = backend.reshape_and_cache(q, k, v, cache, NS(), q.clone())
                for side, original, output, stored in zip(("k", "v"), (k, v), (out_k, out_v), cache):
                    torch.testing.assert_close(output.view(torch.int16), original.view(torch.int16), rtol=0, atol=0)
                    torch.testing.assert_close(stored.view(torch.int16), original.view(torch.int16), rtol=0, atol=0)
                    if target in (side, "kv"):
                        self.assertNotEqual(output.data_ptr(), original.data_ptr())
                        self.assertTrue(output.is_contiguous())
                        output.fill_(9)
                    else:
                        self.assertIs(output, original)
                for original, saved in zip((k, v), before):
                    torch.testing.assert_close(original.view(torch.int16), saved.view(torch.int16), rtol=0, atol=0)
                changed_k, changed_v = k + 1, v - 1
                backend.do_kv_cache_update(None, changed_k, changed_v, cache, torch.tensor([0, 1]))
                torch.testing.assert_close(cache[0].view(torch.int16), changed_k.view(torch.int16), rtol=0, atol=0)
                torch.testing.assert_close(cache[1].view(torch.int16), changed_v.view(torch.int16), rtol=0, atol=0)

    def test_split_writer_consumes_changed_inputs_instead_of_cached_values(self):
        backend = self.create("kv")
        k = torch.full((1, 1, 16), 0.75, dtype=torch.bfloat16)
        k[:, :, -1] = 6
        cache = [torch.zeros_like(k), torch.zeros_like(k)]
        for factor in (1.0, 2.3, 0.0, -0.7):
            value = k * factor
            backend.do_kv_cache_update(None, value, -value, cache, torch.tensor([0]))
            expected = codec.qdq_fp4(value, codec.FP4Recipe(16, "bf16"))
            torch.testing.assert_close(cache[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(cache[1], -expected, rtol=0, atol=0)

    def test_large_noncontiguous_prefill_keeps_group_boundaries_when_tiled(self):
        backend = self.create("kv")
        k = torch.randn(1033, 1, 32).bfloat16()[..., ::2]
        v = -k
        self.assertFalse(k.is_contiguous())
        q = torch.zeros(1033, 2, 16, dtype=torch.bfloat16)
        cache = [torch.zeros_like(k), torch.zeros_like(v)]
        backend.reshape_and_cache(q, k, v, cache, NS(causal=True), q.clone())
        for original, actual in zip((k, v), cache):
            torch.testing.assert_close(actual, codec.qdq_fp4(original, codec.FP4Recipe(16, "bf16")), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            backend.reshape_and_cache(q, k, v, cache, NS(causal=False), q.clone())

    def test_unsupported_attention_and_wrong_tensor_layout_fail_before_writing(self):
        module = load_backend()
        args = [2, 16, 0.25, 1, None, None, "bfloat16", None, "decoder", None]
        for index, value in ((4, [0.1, 0.2]), (5, 128), (6, "fp8"), (7, 1.0), (8, "encoder"), (9, "shared")):
            bad = list(args)
            bad[index] = value
            with self.assertRaises(ValueError):
                module.AscendKVFP4QDQImpl(*bad)
        with self.assertRaises(ValueError):
            module.AscendKVFP4QDQImpl(*args, sinks=torch.ones(2))
        backend = self.create("kv")
        q = torch.zeros(2, 2, 16, dtype=torch.bfloat16)
        k = torch.ones(2, 1, 16, dtype=torch.bfloat16)
        for bad in (k.float(), k[:, :, :8], k.reshape(2, 16), k.expand(2, 2, 16)):
            cache = [torch.zeros_like(k), torch.zeros_like(k)]
            with self.assertRaises(ValueError):
                backend.reshape_and_cache(q, bad, k, cache, NS(), q.clone())
            self.assertEqual(cache[0].count_nonzero().item(), 0)

    def test_real_platform_selector_is_opt_in_and_rejects_other_attention_families(self):
        # Exercise the actual selector method without importing NPU platform setup.
        tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text())
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_select_attn_backend_cls"
        )
        method.decorator_list = []
        node = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        current = [None]
        patcher = patch.dict(sys.modules, {"vllm.config": NS(get_current_vllm_config_or_none=lambda: current[0])})
        patcher.start()
        self.addCleanup(patcher.stop)
        hardware = NS(attention_backend_family="standard")
        namespace = {
            "get_current_hardware_profile": lambda: hardware,
            "AttentionBackendFamily": NS(COMPATIBILITY="compatibility"),
            "_validate_fa3_backend": lambda *a: False,
            "_validate_flash_attn_backend": lambda *a: False,
        }
        exec(compile(node, "actual_platform_selector", "exec"), namespace)
        select = namespace["_select_attn_backend_cls"]
        selector = NS(use_mla=False, use_sparse=False, use_compress=False, use_pcp=False)
        self.assertEqual(select(None, selector), "vllm_ascend.attention.attention_v1.AscendAttentionBackend")
        # Existing PCP capability lookup must not acquire a new singleton-init requirement.
        pcp = NS(**{**vars(selector), "use_pcp": True})
        self.assertEqual(select(None, pcp), "vllm_ascend.attention.attention_v1.AscendAttentionBackend")
        current[0] = NS(additional_config={"kv_fp4_qdq": {"format": "mx4", "target": "kv"}})
        self.assertEqual(select(None, selector), "vllm_ascend.attention.kv_fp4_qdq.AscendKVFP4QDQBackend")
        for name in ("use_mla", "use_sparse", "use_compress", "use_pcp"):
            changed = NS(**vars(selector))
            setattr(changed, name, True)
            with self.assertRaises(ValueError):
                select(None, changed)
        hardware.attention_backend_family = "compatibility"
        with self.assertRaises(ValueError):
            select(None, selector)


if __name__ == "__main__":
    unittest.main()
