# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-1: small-tensor reference packing/QDQ on NPU, no weights or service.

Direct execution requires one explicitly selected NPU. No skipped tests are
allowed to stand in for device evidence. This is not a native FP4 cache backend.
"""

import os
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization import kv_fp4 as codec


def recipes():
    return (
        codec.FP4Recipe(16, "bf16"),
        codec.FP4Recipe(32, "bf16"),
        codec.FP4Recipe(16, "e4m3", 0.02),
        codec.FP4Recipe(32, "e4m3", 0.02),
        codec.FP4Recipe(32, "e8m0"),
    )


class KVFP4DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if not selected.isdecimal() or str(int(selected)) != selected:
            raise RuntimeError("Select exactly one NPU explicitly with ASCEND_RT_VISIBLE_DEVICES")
        root = Path(__file__).resolve().parents[4]
        expected = root / "vllm_ascend/quantization/kv_fp4.py"
        if Path(codec.__file__).resolve() != expected:
            raise RuntimeError("Editable import does not match this test worktree; refusing mixed-source results")
        print(f"implementation={codec.__file__}; torch={torch.__version__}")
        torch.npu.set_device(0)

    def test_packed_bytes_and_values_match_cpu(self):
        generator = torch.Generator().manual_seed(123)
        for shape in ((7, 128), (3, 4, 37), (0, 17)):
            for dtype in (torch.float32, torch.bfloat16):
                host = (torch.randn(shape, generator=generator) * 3.1).to(dtype)
                for recipe in recipes():
                    with self.subTest(shape=shape, dtype=dtype, recipe=recipe):
                        expected = codec.pack_fp4(host, recipe)
                        actual = codec.pack_fp4(host.npu(), recipe)
                        torch.testing.assert_close(actual.data.cpu(), expected.data, rtol=0, atol=0)
                        torch.testing.assert_close(actual.scales.cpu(), expected.scales, rtol=0, atol=0)
                        torch.testing.assert_close(
                            codec.unpack_fp4(actual, dtype).cpu(), codec.unpack_fp4(expected, dtype), rtol=0, atol=0
                        )
                        self.assertEqual(actual.payload_bytes, expected.payload_bytes)

    def test_rounding_boundaries_and_tail(self):
        values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        host = torch.tensor([values + [-v for v in values] + [0.0]])
        for recipe in recipes():
            with self.subTest(recipe=recipe):
                actual = codec.qdq_fp4(host.npu(), recipe).cpu()
                expected = codec.qdq_fp4(host, recipe)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected), rtol=0, atol=0)

    def test_qdq_acl_replay_consumes_new_values(self):
        host = torch.linspace(-5, 5, 9 * 37).reshape(9, 37).to(torch.bfloat16)
        for recipe in recipes():
            with self.subTest(recipe=recipe):
                x = host.npu()
                stream = torch.npu.Stream()
                graph = torch.npu.NPUGraph()
                with torch.npu.stream(stream):
                    for _ in range(3):
                        codec.qdq_fp4(x, recipe)
                stream.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    result = codec.qdq_fp4(x, recipe)
                for factor in (1.0, 2.3, 0.0, -0.7):
                    changed = (host.float() * factor).to(host.dtype)
                    x.copy_(changed)
                    torch.npu.synchronize()
                    graph.replay()
                    torch.npu.synchronize()
                    expected = codec.unpack_fp4(codec.pack_fp4(changed, recipe), host.dtype)
                    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_nonfinite_inputs_are_not_silently_accepted(self):
        for recipe in recipes():
            x = torch.ones((2, 32), device="npu", dtype=torch.bfloat16)
            x[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                codec.pack_fp4(x, recipe)
            result = codec.qdq_fp4(x, recipe).cpu()
            self.assertFalse(torch.isfinite(result[0, : recipe.group_size]).any())
            self.assertTrue(torch.isfinite(result[1]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
