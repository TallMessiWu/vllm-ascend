# SPDX-License-Identifier: Apache-2.0
"""Actual NPU snapshot component tests; does not validate a compiled model."""

import unittest

import torch

from vllm_ascend.observability.moe_routing import RoutingTrace


class RoutingTraceNPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    @torch.inference_mode()
    def test_changed_decode_inputs_replay_into_persistent_buffers(self):
        trace = RoutingTrace(5, 3, 4, 2, "npu", detailed=True)
        h = torch.ones(1, 3, device="npu", dtype=torch.bfloat16)
        logits = torch.ones(1, 4, device="npu")
        ids = torch.tensor([[0, 2]], device="npu", dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]], device="npu")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            trace.record(h, logits, weights, ids)
            trace.record_output(h)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            trace.record(h, logits, weights, ids)
            trace.record_output(h)
        torch.npu.synchronize()
        trace.reset()
        # A real multi-row prefill, followed by three graph decode replays.
        trace.record(h.expand(2, -1), logits.expand(2, -1), weights.expand(2, -1), ids.expand(2, -1))
        trace.record_output(h.expand(2, -1))
        for value in (2, -3, 4):
            h.fill_(value)
            logits.fill_(value)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
        result = trace.snapshot(2, 3)
        self.assertEqual(result["hidden"][:, 0].tolist(), [1, 1, 2, -3, 4])
        self.assertEqual(result["output"][:, 0].tolist(), [1, 1, 2, -3, 4])
        self.assertEqual(result["device_calls"], 4)
        self.assertEqual(result["host_calls"], 1)
