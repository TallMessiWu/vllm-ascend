# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

package = types.ModuleType("kv_qdq_config_tests")
package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/quantization")]
sys.modules[package.__name__] = package
config = importlib.import_module(package.__name__ + ".kv_qdq_config")


def runtime():
    return NS(
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
    )


class KVQDQConfigTests(unittest.TestCase):
    def test_explicit_recipe_target_and_outer_scales(self):
        self.assertIsNone(config.parse_kv_qdq(None))
        for name, group, scale in (("g16-b", 16, "bf16"), ("g32-b", 32, "bf16"), ("mx4", 32, "e8m0")):
            for target in ("off", "k", "v", "kv"):
                cfg = config.parse_kv_qdq({"format": name, "target": target})
                self.assertEqual((cfg.k_recipe.group_size, cfg.k_recipe.scale_type, cfg.target), (group, scale, target))
        cfg = config.parse_kv_qdq({"format": "g16-e", "target": "kv", "k_outer_scale": 0.01, "v_outer_scale": 0.03})
        self.assertNotEqual(cfg.k_recipe.outer_scale, cfg.v_recipe.outer_scale)
        for bad in (
            {},
            False,
            {"format": "mx4"},
            {"format": "g16-e", "target": "kv"},
            {"format": "mx4", "target": "typo"},
            {"format": "mx4", "target": "kv", "typo": 1},
        ):
            with self.assertRaises(ValueError):
                config.parse_kv_qdq(bad)

    def test_diagnostic_copy_is_explicit_boolean_and_default_stays_qdq(self):
        base = {"format": "mx4", "target": "kv"}
        self.assertFalse(config.parse_kv_qdq(base).diagnostic_copy)
        for enabled in (False, True):
            for target in ("off", "k", "v", "kv"):
                setting = config.parse_kv_qdq({**base, "target": target, "diagnostic_copy": enabled})
                self.assertIs(setting.diagnostic_copy, enabled)
                self.assertEqual(setting.target, target)
        for bad in (0, 1, "true", "false", None, [], {}):
            with self.assertRaisesRegex(ValueError, "diagnostic_copy"):
                config.parse_kv_qdq({**base, "diagnostic_copy": bad})

    def test_runtime_restrictions_do_not_silently_change_configuration(self):
        cfg = runtime()
        config.validate_kv_qdq_runtime(cfg)
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            cfg = runtime()
            setattr(cfg.parallel_config, name, 2)
            with self.assertRaises(ValueError):
                config.validate_kv_qdq_runtime(cfg)
        for name in ("lora_config", "speculative_config", "kv_transfer_config", "quant_config"):
            cfg = runtime()
            setattr(cfg, name, object())
            with self.assertRaises(ValueError):
                config.validate_kv_qdq_runtime(cfg)
        for dtype in ("fp8", "mxfp8", "float16"):
            cfg = runtime()
            cfg.cache_config.cache_dtype = dtype
            with self.assertRaises(ValueError):
                config.validate_kv_qdq_runtime(cfg)
        cfg = runtime()
        cfg.model_config.hf_config.model_type = "minimax_m3"
        with self.assertRaises(ValueError):
            config.validate_kv_qdq_runtime(cfg)
        cfg = runtime()
        cfg.model_config.runner_type = "pooling"
        with self.assertRaises(ValueError):
            config.validate_kv_qdq_runtime(cfg)
        cfg = runtime()
        cfg.cache_config.enable_prefix_caching = True
        with self.assertRaises(ValueError):
            config.validate_kv_qdq_runtime(cfg)

    def test_trace_requires_boolean_and_ordinary_mx4_kv(self):
        base = {"format": "mx4", "target": "kv"}
        self.assertFalse(config.parse_kv_qdq(base).diagnostic_trace)
        self.assertTrue(config.parse_kv_qdq({**base, "diagnostic_trace": True}).diagnostic_trace)
        for bad in (1, 0, None, "true"):
            with self.assertRaisesRegex(ValueError, "diagnostic_trace"):
                config.parse_kv_qdq({**base, "diagnostic_trace": bad})
        for overrides in ({"target": "off"}, {"format": "g32-b"}, {"diagnostic_copy": True}):
            with self.assertRaisesRegex(ValueError, "diagnostic_trace"):
                config.parse_kv_qdq({**base, **overrides, "diagnostic_trace": True})


if __name__ == "__main__":
    unittest.main()
