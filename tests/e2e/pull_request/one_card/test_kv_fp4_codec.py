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


def mx4_scale_trace(raw):
    """Isolate the current codec's exponent calculation, not a replacement."""
    raw = raw.clamp(min=codec.FP32_MIN_NORMAL)
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = torch.exp2(exponent.clamp(1, 254).float() - 127)
    return bits, exponent, scale


def compiler_backend():
    # Only the compiled cases need this dependency. Match the service's
    # npugraph_ex options without importing/constructing a model or config.
    import npugraph_ex as nge

    try:
        from npugraph_ex.configs.compiler_config import _process_kwargs_options
    except ImportError:
        from npugraph_ex.configs.npugraphex_config import _process_kwargs_options

    config = nge.CompilerConfig()
    options = dict(force_eager=True, inplace_pass=False, clone_input=False, clone_output=False)
    _process_kwargs_options(config, {"options": options})
    torch.npu.set_compile_mode(jit_compile=False)
    print(f"compile_backend=npugraph_ex; options={options}; static/super kernel not requested", flush=True)
    return nge.get_npu_backend(compiler_config=config)


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

    def assert_outputs_equal(self, actual, expected, phase):
        actual = actual if isinstance(actual, tuple) else (actual,)
        expected = expected if isinstance(expected, tuple) else (expected,)
        self.assertEqual(len(actual), len(expected))
        for index, (value, reference) in enumerate(zip(actual, expected)):
            host_value, host_reference = value.detach().cpu(), reference.detach().cpu()
            if not torch.equal(host_value, host_reference):
                bad = (host_value != host_reference).flatten().nonzero().flatten()[:4].tolist()
                samples = [(i, host_value.flatten()[i].item(), host_reference.flatten()[i].item()) for i in bad]
                print(f"DIFF phase={phase}; output={index}; samples(flat_index,actual,reference)={samples}", flush=True)
            torch.testing.assert_close(host_value, host_reference, rtol=0, atol=0, msg=f"{phase}; output={index}")

    def check_compiled_and_replayed(self, function, host, label):
        # Keep both real and compiled output lifetimes stable. CPU comparisons
        # happen only outside capture, and replay is checked before another call
        # can overwrite a compiler-owned output buffer.
        backend = compiler_backend()
        compilations = []

        def counted_backend(graph, inputs):
            compilations.append(len(list(graph.graph.nodes)))
            print(f"COMPILE {label}; nodes={compilations[-1]}", flush=True)
            return backend(graph, inputs)

        compiled = torch.compile(function, backend=counted_backend, fullgraph=True, dynamic=True)
        device = host.npu()
        factors = (1.0, 2.3, 0.0, -0.7)
        for factor in factors:
            changed = (host.float() * factor).to(host.dtype)
            device.copy_(changed)
            expected = function(changed)
            self.assert_outputs_equal(function(device), expected, f"{label} eager factor={factor}")
            self.assert_outputs_equal(compiled(device), expected, f"{label} compiled factor={factor}")
            print(f"PASS {label} compiled factor={factor}", flush=True)
        device.copy_(host)
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                compiled(device)
        stream.synchronize()
        self.assertTrue(compilations, "NPU compiler backend was not invoked")
        compilation_count = len(compilations)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = compiled(device)
        self.assertEqual(len(compilations), compilation_count, "Unexpected recompilation inside ACL capture")
        for factor in factors:
            changed = (host.float() * factor).to(host.dtype)
            device.copy_(changed)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            self.assert_outputs_equal(result, function(changed), f"{label} compiled+ACL factor={factor}")
            print(f"PASS {label} compiled+ACL factor={factor}", flush=True)
        del result, graph, compiled
        torch._dynamo.reset()

    @torch.inference_mode()
    def test_mx4_compiled_scale_trace(self):
        # Include exact powers and both adjacent FP32 values. Output indices
        # identify bits=0, exponent=1 and scale=2 at the first divergence.
        powers = torch.tensor([2.0**-125, 2.0**-10, 0.5, 1.0, 2.0, 2.0**10, 2.0**124])
        host = torch.stack(
            (
                torch.nextafter(powers, torch.zeros_like(powers)),
                powers,
                torch.nextafter(powers, torch.full_like(powers, float("inf"))),
            ),
            dim=-1,
        )
        self.check_compiled_and_replayed(mx4_scale_trace, host, "MX4 scale: bits/exponent/scale")

    @torch.inference_mode()
    def test_qdq_npugraph_ex_matches_reference(self):
        from vllm_ascend.attention import kv_fp4_qdq as backend

        expected_source = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/kv_fp4_qdq.py"
        self.assertEqual(Path(backend.__file__).resolve(), expected_source)
        print(f"tiled_implementation={backend.__file__}", flush=True)
        for recipe in (codec.FP4Recipe(32, "bf16"), codec.FP4Recipe(32, "e8m0")):
            for tokens in (1, 33, 257):
                # Four KV heads; 257 tokens cross the real 1024-row tile limit.
                host = torch.linspace(-5, 5, tokens * 4 * 128).reshape(tokens, 4, 128).bfloat16()
                label = f"{recipe.scale_type} shape={tuple(host.shape)}"
                print(f"CASE {label}", flush=True)
                self.check_compiled_and_replayed(lambda x, recipe=recipe: backend._qdq_tiled(x, recipe), host, label)

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
