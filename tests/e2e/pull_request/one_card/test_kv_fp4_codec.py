# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-1: small-tensor reference packing/QDQ on NPU, no weights or service.

Direct execution requires one explicitly selected NPU. No skipped tests are
allowed to stand in for device evidence. This is not a native FP4 cache backend.
"""

import os
import unittest
from importlib.metadata import PackageNotFoundError, version
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


def e4m3_stages(x, recipe):
    """Diagnostic re-execution only; never called inside capture/replay."""
    group = recipe.group_size
    padded = torch.nn.functional.pad(x.float(), (0, (-x.shape[-1]) % group))
    blocks = padded.reshape(*x.shape[:-1], padded.shape[-1] // group, group)
    amax = blocks.abs().amax(dim=-1)
    raw = amax / (codec.FP4_MAX * recipe.outer_scale)
    clipped = raw.clamp(min=codec.FP8_MIN, max=codec.FP8_MAX)
    log2 = torch.log2(clipped)
    step = torch.exp2((torch.floor(log2) - 3).clamp(min=-9))
    round_input = clipped / step
    rounded = torch.round(round_input)
    scale = codec._e4m3_round(raw)
    scale = torch.where(amax == 0, 1.0, scale)
    combined = scale * recipe.outer_scale
    normalized = blocks / combined.unsqueeze(-1)
    return dict(
        amax=amax,
        raw=raw,
        log2=log2,
        step=step,
        round_input=round_input,
        rounded=rounded,
        scale=scale,
        combined=combined,
        normalized=normalized,
        codes=codec._codes(normalized),
        scale_bytes=codec._scale_storage(scale, recipe),
        round_ties=torch.round(torch.tensor([0.5, 1.5, 2.5, 12.5, -12.5], device=x.device)),
    )


def report_stage(name, actual, expected):
    """Bounded output: at most four flat indices, including exact FP32 bits."""
    actual, expected = actual.detach().cpu().flatten(), expected.detach().cpu().flatten()
    actual_bits = actual.view(torch.int32) if actual.dtype == torch.float32 else actual
    expected_bits = expected.view(torch.int32) if expected.dtype == torch.float32 else expected
    indices = (actual_bits != expected_bits).nonzero().flatten()
    selected = indices[:4].tolist() if indices.numel() else list(range(min(1, actual.numel())))
    samples = [(i, actual[i].item(), expected[i].item(), int(actual_bits[i]), int(expected_bits[i])) for i in selected]
    print(
        f"DIAG {name}: bit_mismatches={indices.numel()}/{actual.numel()} "
        f"samples(flat_index,device,cpu,device_bits,cpu_bits)={samples}",
        flush=True,
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
        print(f"implementation={codec.__file__}; torch={torch.__version__}", flush=True)
        for package in ("torch-npu", "vllm", "vllm-ascend", "triton-ascend"):
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "metadata unavailable"
            print(f"runtime {package}={installed}", flush=True)
        torch.npu.set_device(0)
        print(f"device={torch.npu.get_device_name(0)}", flush=True)

    def setUp(self):
        # Limit diagnostic volume to one trace per recipe per test method.
        self.diagnosed_recipes = set()

    def assert_matches(self, actual, expected, host, recipe):
        try:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except AssertionError:
            if recipe.scale_type == "e4m3" and recipe not in self.diagnosed_recipes:
                self.diagnosed_recipes.add(recipe)
                print(f"DIAG recipe={recipe}; outside-capture re-execution, not captured intermediates", flush=True)
                cpu_stages = e4m3_stages(host, recipe)
                device_stages = e4m3_stages(host.npu(), recipe)
                for name, cpu_value in cpu_stages.items():
                    report_stage(name, device_stages[name], cpu_value)
            raise

    def test_packed_bytes_and_values_match_cpu(self):
        generator = torch.Generator().manual_seed(123)
        for shape in ((7, 128), (3, 4, 37), (0, 17)):
            for dtype in (torch.float32, torch.bfloat16):
                host = (torch.randn(shape, generator=generator) * 3.1).to(dtype)
                for recipe in recipes():
                    with self.subTest(shape=shape, dtype=dtype, recipe=recipe):
                        expected = codec.pack_fp4(host, recipe)
                        actual = codec.pack_fp4(host.npu(), recipe)
                        self.assert_matches(actual.data.cpu(), expected.data, host, recipe)
                        self.assert_matches(actual.scales.cpu(), expected.scales, host, recipe)
                        self.assert_matches(
                            codec.unpack_fp4(actual, dtype).cpu(), codec.unpack_fp4(expected, dtype), host, recipe
                        )
                        self.assertEqual(actual.payload_bytes, expected.payload_bytes)

    def test_rounding_boundaries_and_tail(self):
        values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        host = torch.tensor([values + [-v for v in values] + [0.0]])
        for recipe in recipes():
            with self.subTest(recipe=recipe):
                actual = codec.qdq_fp4(host.npu(), recipe).cpu()
                expected = codec.qdq_fp4(host, recipe)
                self.assert_matches(actual, expected, host, recipe)
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
                    replay = result.cpu()
                    # Evaluate eager AFTER replay; do not warm the changed input
                    # before replay or inspect device values inside capture.
                    eager = codec.qdq_fp4(x, recipe).cpu()
                    expected = codec.unpack_fp4(codec.pack_fp4(changed, recipe), host.dtype)
                    for comparison, actual, reference in (
                        ("replay_vs_eager", replay, eager),
                        ("eager_vs_cpu", eager, expected),
                        ("replay_vs_cpu", replay, expected),
                    ):
                        with self.subTest(factor=factor, comparison=comparison):
                            print(f"CHECK recipe={recipe}; factor={factor}; comparison={comparison}", flush=True)
                            self.assert_matches(actual, reference, changed, recipe)

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
