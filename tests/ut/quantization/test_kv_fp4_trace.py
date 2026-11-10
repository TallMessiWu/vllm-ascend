# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU/mock tests for diagnostic truthfulness; no NPU/runtime imports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
codec_spec = importlib.util.spec_from_file_location("trace_codec", ROOT / "vllm_ascend/quantization/kv_fp4.py")
codec = importlib.util.module_from_spec(codec_spec)
sys.modules[codec_spec.name] = codec
codec_spec.loader.exec_module(codec)
trace_spec = importlib.util.spec_from_file_location("trace_test", ROOT / "vllm_ascend/attention/kv_fp4_trace.py")
trace_module = importlib.util.module_from_spec(trace_spec)
with patch.dict(sys.modules, {"vllm_ascend.quantization.kv_fp4": codec}):
    trace_spec.loader.exec_module(trace_module)


class KVTraceTests(unittest.TestCase):
    def setUp(self):
        self.recipe = codec.FP4Recipe(32, "e8m0")
        self.settings = NS(k_recipe=self.recipe, v_recipe=self.recipe, diagnostic_trace=True)
        self.key = torch.linspace(-3, 6, 128).reshape(1, 4, 32).bfloat16()
        self.query = torch.zeros(1, 8, 32, dtype=torch.bfloat16)
        self.slots = torch.tensor([3], dtype=torch.int32)
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots)

    def record(self, key, bnsd=False):
        trace = self.trace
        trace.before(key, -key, self.slots)
        transformed = [codec.qdq_fp4(t, self.recipe) for t in (key, -key)]
        trace.transformed(*transformed)
        caches = [torch.zeros(2, 2, 4, 32, dtype=torch.bfloat16) for _ in range(2)]
        for cache, tensor in zip(caches, transformed):
            cache[1, 1].copy_(tensor[0])
        if bnsd:
            caches = [t.permute(0, 2, 1, 3).contiguous() for t in caches]
        trace.after_writer(*caches, bnsd)

    def test_changed_inputs_and_both_cache_layouts_are_compared_at_valid_slot(self):
        for bnsd in (False, True):
            for factor in (1.0, 2.3, 0.0, -0.7):
                self.record(self.key * factor, bnsd)
                row = self.trace.report(self.settings)
                self.assertEqual(row["status"], "GREEN", row)
                self.assertEqual(row["k"]["writer_vs_qdq"], 0)
                self.assertEqual(row["v"]["replay_vs_cpu"], 0)
                self.assertIsNone(row["attention_finite"])
        self.assertEqual(row["replays"], 8)

    def test_corrupted_qdq_and_writer_produce_distinct_red_signals(self):
        self.record(self.key)
        self.trace.qdq[0].zero_()
        row = self.trace.report(self.settings)
        self.assertEqual(row["status"], "RED")
        self.assertGreater(row["k"]["replay_vs_npu_eager"], 0)
        self.record(self.key)
        self.trace.written[1].zero_()
        row = self.trace.report(self.settings)
        self.assertEqual(row["status"], "RED")
        self.assertEqual(row["v"]["replay_vs_npu_eager"], 0)
        self.assertGreater(row["v"]["writer_vs_qdq"], 0)

    def test_zero_replays_padding_and_nonfinite_output_never_pass(self):
        self.assertEqual(self.trace.report(self.settings)["status"], "UNVERIFIED")
        self.record(self.key)
        self.trace.slot.fill_(-1)
        self.assertEqual(self.trace.report(self.settings)["status"], "UNVERIFIED")
        self.trace.slot.fill_(4)
        self.assertEqual(self.trace.report(self.settings)["status"], "UNVERIFIED")
        self.trace.slot.copy_(self.slots)
        self.trace.attention_captured = True
        self.trace.attention.fill_(float("nan"))
        self.assertEqual(self.trace.report(self.settings)["status"], "RED")

    def test_bit_comparison_includes_signed_zero(self):
        self.assertEqual(trace_module.differences(torch.tensor([0.0]).bfloat16(), torch.tensor([-0.0]).bfloat16()), 1)

    def test_rpc_requires_arm_and_rejects_missing_layers(self):
        extension = trace_module.KVTraceWorkerExtension()
        impl = NS(kv_trace=self.trace, kv_qdq_settings=self.settings)
        extension.model_runner = NS()
        extension.vllm_config = NS(
            compilation_config=NS(
                static_forward_context={"layer.0": NS(impl=impl)}, mode="VLLM_COMPILE", cudagraph_mode="FULL"
            ),
            model_config=NS(hf_config=NS(num_hidden_layers=1)),
        )
        self.record(self.key)
        with (
            patch.object(
                torch, "npu", NS(synchronize=lambda: None, config=NS(allow_internal_format=True)), create=True
            ),
            patch.object(trace_module.subprocess, "check_output", return_value="fake-sha\n"),
        ):
            self.assertEqual(extension.kv_trace_control("report")["status"], "UNVERIFIED")
            self.assertEqual(extension.kv_trace_control("arm")["status"], "ARMED")
            self.assertEqual(extension.kv_trace_control("report")["status"], "UNVERIFIED")
            extension.kv_trace_control("arm")
            self.record(self.key * 2.3)
            self.assertEqual(extension.kv_trace_control("report")["status"], "GREEN")
            extension.vllm_config.model_config.hf_config.num_hidden_layers = 2
            self.assertEqual(extension.kv_trace_control("arm")["status"], "UNVERIFIED")

    def test_write_only_config_does_not_block_arm_or_report(self):
        class WriteOnlyConfig:
            def __setattr__(self, name, value):
                pass  # Matches a setter that does not expose a Python getter.

        npu_config = WriteOnlyConfig()
        npu_config.allow_internal_format = True
        self.assertFalse(hasattr(npu_config, "allow_internal_format"))
        extension = trace_module.KVTraceWorkerExtension()
        extension.model_runner = NS()
        extension.vllm_config = NS(
            compilation_config=NS(
                static_forward_context={"layer.0": NS(impl=NS(kv_trace=self.trace, kv_qdq_settings=self.settings))},
                mode="VLLM_COMPILE",
                cudagraph_mode="FULL",
            ),
            model_config=NS(hf_config=NS(num_hidden_layers=1)),
        )
        self.record(self.key)
        with (
            patch.object(torch, "npu", NS(synchronize=lambda: None, config=npu_config), create=True),
            patch.object(trace_module.subprocess, "check_output", return_value="fake-sha\n"),
        ):
            arm = extension.kv_trace_control("arm")
            self.assertEqual(arm["status"], "ARMED")
            self.assertEqual(arm["allow_internal_format"], "UNVERIFIED: no getter")
            self.record(self.key * 2.3)
            report = extension.kv_trace_control("report")
            self.assertEqual(report["status"], "GREEN")
            self.assertEqual(report["allow_internal_format"], "UNVERIFIED: no getter")


if __name__ == "__main__":
    unittest.main()
