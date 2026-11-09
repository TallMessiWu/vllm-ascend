# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small-tensor codec and opaque QDQ graph checks, no weights or service.

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
    scale = codec._e8m0_scale_values(exponent.clamp(1, 254))
    return bits, exponent, scale


def make_opaque_qdq(function, name):
    # Test-local namespace; match attention's opaque, out-buffer mutation
    # boundary without registering/replacing any production attention op.
    @torch.library.custom_op(f"kv_fp4_diagnostic::{name}", mutates_args={"output"})
    def call(x: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(function(x))

    @call.register_fake
    def fake(x, output):
        return None

    def invoke(x, output):
        call(x, output)
        return output

    return invoke


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

    def check_compiled_and_replayed(self, function, opaque, host, device, label):
        # Unlike direct compilation of the codec, the service keeps QDQ inside
        # an opaque attention op. Check that boundary before handing FX to NGE.
        backend = compiler_backend()
        compilations = []

        def counted_backend(graph, inputs):
            calls = [str(node.target) for node in graph.graph.nodes if "kv_fp4_diagnostic" in str(node.target)]
            self.assertEqual(len(calls), 1, "Expected one opaque QDQ call, not inlined codec arithmetic")
            compilations.append(calls[0])
            print(f"COMPILE {label}; opaque={calls[0]}; nodes={len(list(graph.graph.nodes))}", flush=True)
            return backend(graph, inputs)

        compiled = torch.compile(opaque, backend=counted_backend, fullgraph=True, dynamic=True)
        output = torch.empty_like(device, memory_format=torch.contiguous_format)
        device.zero_()
        compiled(device, output)  # Exercise zero-initialized buffers at cold warmup.
        self.assert_outputs_equal(output, torch.zeros_like(host), f"{label} cold zero")
        factors = (1.0, 2.3, 0.0, -0.7)
        for factor in factors:
            changed = (host.float() * factor).to(host.dtype)
            device.copy_(changed)
            expected = function(changed)
            self.assert_outputs_equal(function(device), expected, f"{label} eager factor={factor}")
            compiled(device, output)
            self.assert_outputs_equal(output, expected, f"{label} compiled opaque factor={factor}")
            print(f"PASS {label} compiled opaque factor={factor}", flush=True)
        self.assertTrue(compilations, "NPU compiler backend was not invoked")
        for capture_factor in (0.0, 1.0):
            device.copy_((host.float() * capture_factor).to(host.dtype))
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    compiled(device, output)
            stream.synchronize()
            compilation_count = len(compilations)
            graph = torch.npu.NPUGraph()
            # Retain GLOBAL capture mode: synchronous host copies must fail.
            with torch.npu.graph(graph, stream=stream):
                compiled(device, output)
            self.assertEqual(len(compilations), compilation_count, "Unexpected recompilation inside ACL capture")
            for factor in factors:
                changed = (host.float() * factor).to(host.dtype)
                device.copy_(changed)
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                phase = f"{label} opaque+ACL captured={capture_factor} replay={factor}"
                self.assert_outputs_equal(output, function(changed), phase)
                print(f"PASS {phase}", flush=True)
            del graph
        del compiled
        torch._dynamo.reset()

    def check_opaque_qdq(self, scale_type):
        from vllm_ascend.attention import kv_fp4_qdq as backend

        self.addCleanup(torch._dynamo.reset)
        expected_source = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/kv_fp4_qdq.py"
        self.assertEqual(Path(backend.__file__).resolve(), expected_source)
        print(f"tiled_implementation={backend.__file__}", flush=True)
        recipe = codec.FP4Recipe(32, scale_type)
        function = lambda x: backend._qdq_tiled(x, recipe)
        opaque = make_opaque_qdq(function, f"qdq_{scale_type}")
        for tokens, sliced in ((1, False), (33, True), (257, True)):
            # Synthetic split-QKV view: nonzero offset and non-contiguous rows.
            # Four KV heads; 257 tokens also cross the 1024-row tile limit.
            host = torch.linspace(-5, 5, tokens * 4 * 128).reshape(tokens, 4, 128).bfloat16()
            backing = torch.zeros((tokens, 3, 4, 128), dtype=host.dtype, device="npu") if sliced else None
            device = backing[:, 1] if sliced else torch.empty_like(host, device="npu")
            label = f"{scale_type} shape={tuple(host.shape)} stride={device.stride()} offset={device.storage_offset()}"
            print(f"CASE {label}", flush=True)
            self.check_compiled_and_replayed(function, opaque, host, device, label)
        if scale_type == "e8m0":
            host = torch.full((1, 4, 128), 6 * codec.FP32_MIN_NORMAL, dtype=torch.bfloat16)
            device = torch.empty_like(host, device="npu")
            self.check_compiled_and_replayed(function, opaque, host, device, "e8m0 nonzero floor")

    @torch.inference_mode()
    def test_mx4_opaque_compile_replay(self):
        self.check_opaque_qdq("e8m0")

    @torch.inference_mode()
    def test_g32_b_opaque_compile_replay(self):
        self.check_opaque_qdq("bf16")

    def check_exact_scale_replay(self):
        # The int32 producer and uint8 reader share the bit-construction helper.
        # Retain both input and viewed output storage until replay completes.
        host = torch.arange(1, 255, dtype=torch.int32)
        for dtype in (torch.int32, torch.uint8):
            device = host.to(dtype).npu()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    codec._e8m0_scale_values(device)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                result = codec._e8m0_scale_values(device)
            for name, changed in (
                ("reverse", host.flip(0)),
                ("minimum", torch.ones_like(host)),
                ("maximum", torch.full_like(host, 254)),
                ("ascending", host),
            ):
                device.copy_(changed.to(dtype))
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                literal = (changed << 23).view(torch.float32)
                self.assert_outputs_equal(result, literal, f"exact scale replay {dtype} {name}")
                print(f"PASS exact scale replay {dtype} {name}", flush=True)
            del result, graph

    @torch.inference_mode()
    def test_mx4_scale_floor_eager(self):
        # Check the active producer and packed reader, not a duplicate of the
        # legacy exp2 expression. Retain zero tolerance and independent oracles.
        torch.npu.set_compile_mode(jit_compile=False)
        recipe = codec.FP4Recipe(32, "e8m0")
        zero = torch.zeros((2, 32), dtype=torch.bfloat16)
        tiny = torch.full_like(zero, 6 * codec.FP32_MIN_NORMAL)

        def effective(x):
            return codec._quantize_groups(x, recipe)[1], codec.qdq_fp4(x, recipe)

        legacy = torch.exp2(torch.tensor([-126.0], device="npu")).cpu().tolist()
        print(f"LEGACY observation exp2(-126)={legacy}; active codec must satisfy the exact oracle", flush=True)
        for label, function, host in (
            ("effective zero-group scale=1/QDQ=0", effective, zero),
            ("active raw floor bits/exponent/scale", mx4_scale_trace, torch.zeros((7, 3))),
            ("effective nonzero floor", effective, tiny),
            ("effective negative nonzero floor", effective, -tiny),
        ):
            with self.subTest(boundary=label):
                print(f"BOUNDARY {label}", flush=True)
                self.assert_outputs_equal(function(host.npu()), function(host), label)
                print(f"PASS boundary {label}", flush=True)
        with self.subTest(boundary="reference scale bytes 1..254"):
            scales = torch.arange(1, 255, dtype=torch.uint8).npu()
            literal = (torch.arange(1, 255, dtype=torch.int32) << 23).view(torch.float32)
            legacy = torch.exp2(scales.float() - 127).clamp(min=codec.FP32_MIN_NORMAL).cpu()
            bad = (legacy != literal).nonzero().flatten() + 1
            print(f"LEGACY exp2+clamp mismatching scale bytes={bad.tolist()}", flush=True)
            actual = codec._scale_values(scales, recipe)
            self.assert_outputs_equal(actual, literal, "reference E8M0 scales 1..254")
            self.assert_outputs_equal(codec._scale_storage(actual, recipe), scales, "E8M0 byte round-trip")
            print("PASS reference E8M0 scales 1..254 and byte round-trip", flush=True)
        with self.subTest(boundary="all reference exponents ACL replay"):
            self.check_exact_scale_replay()
        with self.subTest(boundary="reported exponent FP4 midpoints"):
            codes = torch.tensor([18, 19, 20, 24], dtype=torch.uint8)
            powers = torch.tensor([2.0**e for e in (-109, -108, -107, -103)])
            row = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]).repeat(4)
            rounded = torch.tensor([0, 1, 1, 2, 2, 4, 4, 6]).repeat(4)
            host, expected = powers[:, None] * row, powers[:, None] * rounded
            packed = codec.pack_fp4(host.npu(), recipe)
            literal_data = torch.tensor([[0x20, 0x42, 0x64, 0x76] * 4] * 4, dtype=torch.uint8)
            self.assert_outputs_equal((packed.scales, packed.data), (codes[:, None], literal_data), "midpoint bytes")
            literal = codec.PackedFP4(literal_data.npu(), codes[:, None].npu(), tuple(host.shape), recipe)
            self.assert_outputs_equal(codec.unpack_fp4(literal), expected, "midpoint literal reader")
            self.assert_outputs_equal(codec.qdq_fp4(host.npu(), recipe), expected, "midpoint QDQ")
            print("PASS reported-exponent FP4 midpoints and literal payload", flush=True)
        with self.subTest(boundary="floor packed payload and reader"):
            pattern = (torch.tensor([0, 1, 2, 3, 4, 6, 2, 4]) * codec.FP32_MIN_NORMAL).bfloat16().repeat(4)[None]
            host = torch.cat((zero[:1], tiny[:1], -tiny[:1], pattern, -pattern))
            literal = codec.PackedFP4(
                torch.tensor(
                    [[0x00] * 16, [0x77] * 16, [0xFF] * 16, [0x20, 0x54, 0x76, 0x64] * 4, [0xA8, 0xDC, 0xFE, 0xEC] * 4],
                    dtype=torch.uint8,
                    device="npu",
                ),
                torch.tensor([[127], [1], [1], [1], [1]], dtype=torch.uint8, device="npu"),
                tuple(host.shape),
                recipe,
            )
            packed = codec.pack_fp4(host.npu(), recipe)
            self.assert_outputs_equal((packed.data, packed.scales), (literal.data, literal.scales), "packed floor")
            self.assert_outputs_equal(codec.unpack_fp4(literal, host.dtype), host, "literal floor reader")
            self.assert_outputs_equal(codec.qdq_fp4(host.npu(), recipe), host, "mixed floor QDQ")
            print("PASS floor packed bytes and independent literal payload reader", flush=True)

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
