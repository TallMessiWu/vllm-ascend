# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small NPU eager/ACL replay check, no checkpoint/model loading.

Run directly with ASCEND_RT_VISIBLE_DEVICES explicitly selecting one device.
A missing device/runtime is a failure, not a skipped/green acceptance result.
"""

import os
import unittest

import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization.nvfp4_a4 import nvfp4_activation_qdq, routed_global_scales


class NVFP4A4DeviceTests(unittest.TestCase):
    def test_eager_and_replay_use_new_inputs_and_routing_groups(self):
        if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            self.fail("Select the NPU explicitly via ASCEND_RT_VISIBLE_DEVICES")
        torch.npu.set_device(0)
        for rows in (32, 2049):
            with self.subTest(rows=rows):
                self.check_replay(rows)

    def check_replay(self, rows):
        generator = torch.Generator().manual_seed(321)
        host = torch.randn((rows, 64), generator=generator).to(torch.bfloat16)
        scales_cpu = torch.tensor([0.02, 0.04, 0.08, 0.1], dtype=torch.float32)
        initial_counts = [0, rows // 3, rows // 4, rows - rows // 3 - rows // 4]
        counts_cpu = torch.tensor(initial_counts, dtype=torch.int64)
        x, scales, counts = host.npu(), scales_cpu.npu(), counts_cpu.npu()

        def apply(value, group):
            row_scales = routed_global_scales(scales, group, 1, value.shape[0])
            return nvfp4_activation_qdq(value, row_scales)

        expected = nvfp4_activation_qdq(host, routed_global_scales(scales_cpu, counts_cpu, 1, host.shape[0]))
        eager = apply(x, counts)
        torch.npu.synchronize()
        torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected, host), "QDQ must actually affect the values")
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(3):
                apply(x, counts)
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream):
            captured = apply(x, counts)
        for factor, new_counts in (
            (1.0, initial_counts),
            (2.3, [rows // 4, 0, rows // 2, rows - rows // 4 - rows // 2]),
            (-0.7, [0, 0, 0, rows]),
        ):
            changed = (host.float() * factor).to(host.dtype)
            counts_cpu = torch.tensor(new_counts, dtype=torch.int64)
            x.copy_(changed)
            counts.copy_(counts_cpu)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            reference = nvfp4_activation_qdq(changed, routed_global_scales(scales_cpu, counts_cpu, 1, host.shape[0]))
            torch.testing.assert_close(captured.cpu(), reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
