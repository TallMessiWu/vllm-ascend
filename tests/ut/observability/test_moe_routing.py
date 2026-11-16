# SPDX-License-Identifier: Apache-2.0
"""CPU contracts only; no vLLM, torch_npu or model import."""

import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("routing_metrics", ROOT / "vllm_ascend/observability/moe_routing.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RoutingMetricsTests(unittest.TestCase):
    def test_order_changes_are_not_expert_changes_and_weights_follow_ids(self):
        ids = torch.tensor([[0, 2]])
        result = MODULE.routing_metrics(ids, ids.flip(1), torch.tensor([[0.75, 0.25]]), torch.tensor([[0.25, 0.75]]), 4)
        self.assertEqual(result["overlap_mean"], 1.0)
        self.assertEqual(result["set_equal_rate"], 1.0)
        self.assertEqual(result["weight_l1_mean"], 0.0)

    def test_replacing_one_expert_has_known_overlap_and_weight_mass(self):
        result = MODULE.routing_metrics(
            torch.tensor([[0, 2]]), torch.tensor([[0, 3]]), torch.tensor([[0.75, 0.25]]), torch.tensor([[0.5, 0.5]]), 4
        )
        self.assertEqual(result["overlap_mean"], 0.5)
        self.assertEqual(result["set_equal_rate"], 0.0)
        self.assertEqual(result["retained_reference_weight_mean"], 0.75)
        self.assertEqual(result["weight_l1_mean"], 1.0)
        self.assertEqual(result["load_total_variation"], 0.5)

    def test_empty_duplicate_invalid_and_nonfinite_routes_are_rejected(self):
        for ids, weights in [
            (torch.empty(0, 2, dtype=torch.int64), torch.empty(0, 2)),
            (torch.tensor([[1, 1]]), torch.ones(1, 2)),
            (torch.tensor([[-1, 2]]), torch.ones(1, 2)),
            (torch.tensor([[0, 4]]), torch.ones(1, 2)),
            (torch.tensor([[0, 2]]), torch.tensor([[1.0, float("nan")]])),
        ]:
            with self.assertRaises(ValueError):
                MODULE.validate_routes(ids, weights, 4)

    def test_cpu_fp64_router_has_literal_softmax_and_reports_boundary_ties(self):
        logits, ids, weights, margin = MODULE.router_reference(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0]]), 2, True
        )
        self.assertEqual(logits.tolist(), [[2.0, 0.0, -2.0]])
        self.assertEqual(ids.tolist(), [[0, 1]])
        self.assertAlmostEqual(weights[0, 0].item(), 0.8807970779778823, places=14)
        self.assertEqual(margin["margin_min"], 2.0)
        _, _, _, ties = MODULE.router_reference(torch.ones(1, 1), torch.ones(3, 1), 1, True)
        self.assertEqual(ties["exact_boundary_ties"], 1)

    def test_arithmetic_mismatch_and_zero_reference_norm_are_explicit(self):
        result = MODULE.tensor_error(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0]))
        self.assertFalse(result["exact"])
        self.assertEqual(result["abs_max"], 2.0)
        self.assertIsNone(result["relative_l2"])
        with self.assertRaises(ValueError):
            MODULE.tensor_error(torch.tensor([float("nan")]), torch.ones(1))


class RoutingTraceTests(unittest.TestCase):
    def test_prefill_then_decode_is_aligned_and_reset_discards_warmup(self):
        trace = MODULE.RoutingTrace(4, 3, 4, 2, "cpu", detailed=True)
        h = torch.arange(9).reshape(3, 3).bfloat16()
        logits = torch.arange(12).reshape(3, 4).float()
        ids = torch.tensor([[0, 2]]).expand(3, -1).int()
        weights = torch.tensor([[0.75, 0.25]]).expand(3, -1)
        trace.record(h, logits, weights, ids)
        trace.record_output(h)
        trace.reset()
        trace.record(h, logits, weights, ids)
        trace.record_output(h)
        trace.record(-h[:1], -logits[:1], weights[:1], ids[:1])
        trace.record_output(-h[:1])
        result = trace.snapshot(3, 1)
        self.assertEqual(result["device_calls"], 2)
        self.assertTrue(torch.equal(result["hidden"][3], -h[0]))
        self.assertTrue(torch.equal(result["output"][3], -h[0]))
        self.assertEqual(result["prefill_rows"], 3)
        self.assertEqual(result["decode_rows"], 1)

    def test_no_samples_overflow_and_missing_output_cannot_pass(self):
        trace = MODULE.RoutingTrace(1, 2, 4, 2, "cpu", detailed=True)
        h, logits = torch.ones(1, 2).bfloat16(), torch.ones(1, 4)
        ids, weights = torch.tensor([[0, 2]]).int(), torch.tensor([[0.75, 0.25]])
        with self.assertRaises(ValueError):
            trace.snapshot(1, 0)
        trace.record(h, logits, weights, ids)
        with self.assertRaises(ValueError):
            trace.snapshot(1, 0)
        trace.record_output(h)
        trace.record(h, logits, weights, ids)
        with self.assertRaises(ValueError):
            trace.snapshot(1, 0)

    def test_inference_allocated_buffers_can_be_reset_in_idle_rpc_thread(self):
        with torch.inference_mode():
            trace = MODULE.RoutingTrace(2, 3, 4, 2, "cpu")
        address = trace.ids.data_ptr()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(trace.reset).result()
        self.assertEqual(trace.ids.data_ptr(), address)
        self.assertEqual(trace.ids.tolist(), [[-1, -1], [-1, -1]])

    def test_changing_fullgraph_inputs_update_the_same_buffers(self):
        trace = MODULE.RoutingTrace(4, 2, 4, 2, "cpu", detailed=True)

        def record(h, logits, weights, ids):
            trace.record(h, logits, weights, ids)
            trace.record_output(h)

        compiled = torch.compile(record, backend="eager", fullgraph=True)
        h, logits = torch.ones(3, 2).bfloat16(), torch.ones(3, 4)
        ids, weights = torch.tensor([[0, 2]]).expand(3, -1).int(), torch.tensor([[0.75, 0.25]]).expand(3, -1)
        for factor in (1, -3):
            trace.reset()
            compiled(h * factor, logits * factor, weights, ids)
            compiled(h[:1] * factor, logits[:1] * factor, weights[:1], ids[:1])
            result = trace.snapshot(3, 1)
            self.assertEqual(result["hidden"][3].tolist(), [factor, factor])


if __name__ == "__main__":
    unittest.main()
