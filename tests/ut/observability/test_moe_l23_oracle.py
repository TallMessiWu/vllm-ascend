# SPDX-License-Identifier: Apache-2.0
"""Exercise route alignment and prediction metrics without a device runtime."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_moe_mxfp4_w_check import driver_module
from test_moe_routing_runtime import external_module, load


class Layer23OracleTests(unittest.TestCase):
    def test_arm_from_snapshot_and_switch_using_the_device_row_cursor(self):
        traces = load("l23_trace_fixture", "vllm_ascend/observability/moe_routing.py")

        class ObservedRouter:
            def __init__(self, base, trace):
                self.base, self.trace = base, trace

        class ProbeModel(torch.nn.Module):
            def compute_logits(self, hidden):
                return hidden

        modules = {
            "vllm_ascend.models.qwen3_moe_routing_probe": external_module(
                ObservedRouter=ObservedRouter, Qwen3MoeRoutingProbeForCausalLM=ProbeModel
            ),
            "vllm_ascend.observability.moe_routing": traces,
        }
        with patch.dict(sys.modules, modules):
            runtime = load("l23_model_fixture", "vllm_ascend/models/qwen3_moe_l23_oracle.py")
            rpc = load("l23_rpc_fixture", "vllm_ascend/observability/moe_routing_rpc.py")
        trace = traces.RoutingTrace(3, 4, 4, 2, "cpu")
        native = SimpleNamespace(
            _compute_routing=lambda hidden, *args, **kwargs: (
                torch.tensor([[0.75, 0.25]]).repeat(len(hidden), 1),
                torch.tensor([[0, 1]], dtype=torch.int32).repeat(len(hidden), 1),
            )
        )
        router = runtime.Layer23OracleRouter(native, trace)
        ids = torch.tensor([[2, 0], [1, 3], [3, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.5], [0.625, 0.375], [0.25, 0.75]])
        model = SimpleNamespace(routing_traces=[trace] * 48, arm_l23_trial=router.arm)
        with tempfile.TemporaryDirectory(dir=".") as folder:
            path = Path(folder) / "reference.pt"
            torch.save({"layers": [{"ids": ids, "weights": weights}] * 48}, path)
            with patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True):
                rpc.reset_probe(model, str(path))
                for span in (slice(0, 2), slice(2, 3)):
                    hidden = torch.zeros((len(ids[span]), 4), dtype=torch.bfloat16)
                    used_weights, used_ids = router._compute_routing(hidden, hidden.float(), None)
                    torch.testing.assert_close(used_ids, ids[span])
                    torch.testing.assert_close(used_weights, weights[span])
                torch.testing.assert_close(trace.ids, ids)
                rpc.reset_probe(model)
            hidden = torch.zeros(1, 4, dtype=torch.bfloat16)
            weights_after_reset, ids_after_reset = router._compute_routing(hidden, hidden.float(), None)
            self.assertEqual(ids_after_reset.tolist(), [[0, 1]])
            self.assertEqual(weights_after_reset.tolist(), [[0.75, 0.25]])

        model_type = runtime.Qwen3MoeL23OracleForCausalLM
        model = model_type.__new__(model_type)
        ProbeModel.__init__(model)
        model.oracle_router = router
        model.arm_l23_trial(None)
        for step in range(traces.DECODE_ROWS + 1):
            logits = torch.tensor([[float(step), 2.0]])
            self.assertIs(model.compute_logits(logits), logits)
            logits.zero_()
        captured = model.take_continuation_logits()
        torch.testing.assert_close(captured[:, 0], torch.arange(traces.DECODE_ROWS + 1, dtype=torch.float32))
        self.assertIsNone(model.continuation_logits)

    def test_full_distribution_kl_and_forced_token_scores(self):
        comparison, _ = driver_module()
        with patch.dict(sys.modules, {"moe_mxfp4_w_check": comparison}):
            driver = load("l23_example_fixture", "examples/moe_l23_oracle.py")
        target = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
        candidate = target.flip(-1)
        scores = driver.compare_predictions(target, candidate, target, [0, 1])
        self.assertEqual(tuple(scores.shape), (5, 2))
        torch.testing.assert_close(scores[0], scores[2])
        torch.testing.assert_close(scores[4], torch.zeros(2, dtype=torch.float64))
        self.assertTrue((scores[1] > scores[2]).all())
        self.assertTrue((scores[3] > 0).all())
