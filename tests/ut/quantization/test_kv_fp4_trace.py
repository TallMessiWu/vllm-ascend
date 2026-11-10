# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU/mock tests for diagnostic truthfulness; no NPU/runtime imports."""

import importlib.util
import json
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        transformed = [
            codec.qdq_fp4(t.reshape(-1, t.shape[-1]), self.recipe, observer=trace.observer(side)).reshape(t.shape)
            for side, t in zip(("k", "v"), (key, -key))
        ]
        trace.transformed(*transformed, source=(key, -key))
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
        self.assertEqual(trace_module.differences(torch.tensor([True]), torch.tensor([False])), 1)

    def test_scalar_and_tensor_shadow_steps_and_constant_ownership(self):
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True, probe_ops=True)
        self.assertEqual(self.trace.exponent_probe.seen, [set(), set()])
        self.assertEqual(int(self.trace.count[0]), 0)
        for factor in (1.0, 2.3, 0.0, -0.7):
            self.record(self.key * factor)
            row = self.trace.report(self.settings)
            self.assertEqual(row["status"], "GREEN", row)
            for side in ("k", "v"):
                probes = row[side]["detail"]["exponent_ops"]
                self.assertFalse(any(probes["constant_errors"].values()))
                for mode in ("scalar", "tensor"):
                    self.assertEqual(probes[mode]["first_replay_divergence"], "none")
                    self.assertEqual(probes[mode]["active_vs_shadow_exponent"], 0)
        self.trace.exponent_probe.outputs[0]["scalar"]["shift"].add_(1)
        row = self.trace.report(self.settings)
        probes = row["k"]["detail"]["exponent_ops"]
        self.assertEqual(row["status"], "RED")
        self.assertEqual(probes["scalar"]["first_replay_divergence"], "shift")
        self.assertEqual(probes["tensor"]["status"], "GREEN")
        self.trace.exponent_probe.constants["shift"].zero_()
        self.assertEqual(
            self.trace.report(self.settings)["k"]["detail"]["exponent_ops"]["tensor"]["status"], "UNVERIFIED"
        )
        addresses = {name: tensor.data_ptr() for name, tensor in self.trace.exponent_probe.constants.items()}
        self.trace.arm()
        self.assertEqual(
            {name: tensor.data_ptr() for name, tensor in self.trace.exponent_probe.constants.items()}, addresses
        )
        self.record(self.key)
        self.assertEqual(self.trace.report(self.settings)["status"], "GREEN")

    def test_shadow_exponent_steps_match_literal_integer_arithmetic(self):
        words = [(exponent << 23) | fraction for exponent in range(1, 255) for fraction in (0, 1, 0x2AAAAA, 0x7FFFFF)]
        bits = torch.tensor(words, dtype=torch.int32).reshape(-1, 4)
        constants = {name: torch.tensor([value], dtype=torch.int32) for name, value in trace_module.EXPONENT_CONSTANTS}
        expected = {
            "shift": [word >> 23 for word in words],
            "biased": [(word >> 23) & 255 for word in words],
            "mantissa": [word & 0x7FFFFF for word in words],
            "nonzero": [(word & 0x7FFFFF) != 0 for word in words],
            "round_up": [int((word & 0x7FFFFF) != 0) for word in words],
            "exponent": [((word >> 23) & 255) + int((word & 0x7FFFFF) != 0) for word in words],
        }
        for operands in (None, constants):
            actual = trace_module.exponent_steps(bits, operands)
            for name, values in expected.items():
                self.assertEqual(actual[name].flatten().tolist(), values, name)

    def test_missing_shadow_steps_are_unverified(self):
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True, probe_ops=True)
        self.record(self.key)
        self.trace.exponent_probe.seen[0].discard(("tensor", "shift"))
        row = self.trace.report(self.settings)
        self.assertEqual(row["status"], "UNVERIFIED")

    def test_detailed_stages_detect_first_corruption_and_input_change(self):
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True)
        for bnsd in (False, True):
            for factor in (1.0, 2.3, 0.0, -0.7):
                self.record(self.key * factor, bnsd)
                row = self.trace.report(self.settings)
                self.assertEqual(row["status"], "GREEN", row)
                self.assertEqual(row["k"]["detail"]["first_divergence"], "none")
                self.assertEqual(row["v"]["npu_eager_vs_cpu"], 0)
        for stage in trace_module.MX4_STAGES:
            self.record(self.key)
            self.trace.stages[0][stage].flatten()[0] += 1
            row = self.trace.report(self.settings)
            self.assertEqual(row["status"], "RED")
            self.assertEqual(row["k"]["detail"]["first_divergence"], stage)
        self.record(self.key)
        self.trace.raw_after[1].zero_()
        self.assertEqual(
            self.trace.report(self.settings)["v"]["detail"]["first_divergence"], "source_changed_during_qdq"
        )

    def test_missing_detail_stages_cannot_pass(self):
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True)
        self.record(self.key)
        self.trace.captured_stages[0].discard("amax")
        row = self.trace.report(self.settings)
        self.assertEqual(row["status"], "UNVERIFIED")
        self.assertEqual(row["k"]["detail"]["missing_stages"], ["amax"])

    def test_default_detailed_trace_runs_no_scalar_shadow(self):
        with patch.object(trace_module, "ExponentOpTrace", side_effect=AssertionError("legacy scalar probe created")):
            self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True)
            self.record(self.key)
            row = self.trace.report(self.settings, exponent_shift=torch.tensor([23], dtype=torch.int32))
        self.assertEqual(row["status"], "GREEN")
        self.assertNotIn("exponent_ops", row["k"]["detail"])

    def test_equal_mismatch_counts_do_not_imply_equal_oracles(self):
        self.record(self.key)
        calls = [0]

        def different_oracles(tensor, recipe, observer=None, exponent_shift=None):
            calls[0] += 1
            return codec.qdq_fp4(tensor, recipe) + (1 if calls[0] % 2 else 2)

        with patch.object(trace_module, "qdq_fp4", side_effect=different_oracles):
            row = self.trace.report(self.settings)
        self.assertEqual(row["k"]["replay_vs_cpu"], row["k"]["replay_vs_npu_eager"])
        self.assertGreater(row["k"]["npu_eager_vs_cpu"], 0)

    def test_oracle_only_fp32_differences_do_not_hide_replay_exponent_failure(self):
        self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True)
        self.record(self.key)
        calls = [0]

        def oracle_with_fp32_difference(tensor, recipe, observer=None, exponent_shift=None):
            calls[0] += 1
            cpu_call = calls[0] % 2 == 0

            def publish(name, value):
                if cpu_call and name == "raw_scale":
                    value = torch.nextafter(value, torch.full_like(value, float("inf")))
                elif cpu_call and name == "scale_bits":
                    value = value + 1
                observer(name, value)

            return codec.qdq_fp4(tensor, recipe, observer=publish if observer is not None else None)

        with patch.object(trace_module, "qdq_fp4", side_effect=oracle_with_fp32_difference):
            row = self.trace.report(self.settings)
            self.assertEqual(row["status"], "GREEN", row)
            detail = row["k"]["detail"]
            self.assertEqual(detail["first_replay_divergence"], "none")
            self.assertEqual(detail["first_oracle_divergence"], "raw_scale")
            self.assertEqual(row["k"]["npu_eager_vs_cpu"], 0)
            self.trace.stages[0]["exponent"].add_(1)
            row = self.trace.report(self.settings)
            self.assertEqual(row["status"], "RED")
            detail = row["k"]["detail"]
            self.assertEqual(detail["first_divergence"], "exponent")
            self.assertEqual(detail["first_replay_divergence"], "exponent")
            self.assertEqual(detail["first_oracle_divergence"], "raw_scale")
            self.assertEqual(detail["exponent_delta_histogram"], {"1": self.trace.stages[0]["exponent"].numel()})

    def test_rpc_requires_arm_and_rejects_missing_layers(self):
        extension = trace_module.KVTraceWorkerExtension()
        impl = NS(kv_trace=self.trace, kv_qdq_settings=self.settings, _mx4_shift=torch.tensor([23], dtype=torch.int32))
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
                static_forward_context={
                    "layer.0": NS(
                        impl=NS(
                            kv_trace=self.trace,
                            kv_qdq_settings=self.settings,
                            _mx4_shift=torch.tensor([23], dtype=torch.int32),
                        )
                    )
                },
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

    def test_arm_validates_serving_shift_without_refreshing_or_replacing_it(self):
        self.record(self.key)
        shift = torch.tensor([22], dtype=torch.int32)
        impl = NS(kv_trace=self.trace, kv_qdq_settings=self.settings, _mx4_shift=shift)
        worker = trace_module.KVTraceWorkerExtension()
        worker.model_runner = NS()
        worker.vllm_config = NS(
            compilation_config=NS(
                static_forward_context={"layer.0": NS(impl=impl)}, mode="VLLM_COMPILE", cudagraph_mode="FULL"
            ),
            model_config=NS(hf_config=NS(num_hidden_layers=1)),
        )
        with (
            patch.object(torch, "npu", NS(synchronize=lambda: None, config=NS()), create=True),
            patch.object(trace_module.subprocess, "check_output", return_value="fake-sha\n"),
        ):
            self.assertEqual(worker.kv_trace_control("arm")["status"], "RED")
            self.assertEqual(shift.tolist(), [22])
            shift.fill_(23)
            self.assertEqual(worker.kv_trace_control("arm")["status"], "ARMED")
            self.assertIs(impl._mx4_shift, shift)
            self.assertEqual(worker.kv_trace_control("arm")["shift_buffers"], 1)

    def test_inference_snapshots_survive_rpc_thread_arm_and_report_without_reallocation(self):
        # Model warmup/capture allocates inference tensors; the executor's RPC
        # thread does not inherit that thread-local mode. Exercise that boundary
        # with real CPU Torch tensors, not a mock of the failing zero_ operation.
        with torch.inference_mode():
            self.trace = trace_module.KVReplayTrace(self.key, self.query, self.slots, detailed=True)
            self.record(self.key)
            self.trace.attention.zero_()
            self.trace.attention_captured = True
        self.assertTrue(self.trace.count.is_inference())

        def buffers():
            return [
                self.trace.count,
                self.trace.slot,
                self.trace.attention,
                *self.trace.raw,
                *self.trace.qdq,
                *self.trace.written,
                *self.trace.raw_after,
                *(tensor for side in self.trace.stages for tensor in side.values()),
            ]

        addresses = [tensor.data_ptr() for tensor in buffers()]
        extension = trace_module.KVTraceWorkerExtension()
        extension.model_runner = NS()
        extension.vllm_config = NS(
            compilation_config=NS(
                static_forward_context={
                    "layer.0": NS(
                        impl=NS(
                            kv_trace=self.trace,
                            kv_qdq_settings=self.settings,
                            _mx4_shift=torch.tensor([23], dtype=torch.int32),
                        )
                    )
                },
                mode="VLLM_COMPILE",
                cudagraph_mode="FULL",
            ),
            model_config=NS(hf_config=NS(num_hidden_layers=1)),
        )

        def rpc(action):
            self.assertFalse(torch.is_inference_mode_enabled())
            with torch.no_grad():
                result = extension.kv_trace_control(action)
            self.assertFalse(torch.is_inference_mode_enabled())
            return json.loads(json.dumps(result))

        with (
            patch.object(torch, "npu", NS(synchronize=lambda: None, config=NS()), create=True),
            patch.object(trace_module.subprocess, "check_output", return_value="fake-sha\n"),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            for bnsd in (False, True):
                arm = executor.submit(rpc, "arm").result(timeout=10)
                self.assertEqual(arm["status"], "ARMED")
                self.assertEqual(arm["allow_internal_format"], "UNVERIFIED: no getter")
                self.assertEqual(int(self.trace.count[0]), 0)
                self.assertEqual(executor.submit(rpc, "report").result(timeout=10)["status"], "UNVERIFIED")
                executor.submit(rpc, "arm").result(timeout=10)
                with torch.inference_mode():
                    for factor in (2.3, -0.7):
                        self.record(self.key * factor, bnsd)
                report = executor.submit(rpc, "report").result(timeout=10)
                self.assertEqual(report["status"], "GREEN", report)
                self.assertEqual(report["layers"]["layer.0"]["replays"], 2)
                self.assertTrue(report["layers"]["layer.0"]["attention_finite"])
                self.assertEqual([tensor.data_ptr() for tensor in buffers()], addresses)
                executor.submit(rpc, "arm").result(timeout=10)
                with torch.inference_mode():
                    self.record(self.key, bnsd)
                    self.trace.written[0].zero_()
                self.assertEqual(executor.submit(rpc, "report").result(timeout=10)["status"], "RED")


if __name__ == "__main__":
    unittest.main()
