# SPDX-License-Identifier: Apache-2.0
"""CPU worker-extension/MessagePack boundary; no NPU or insecure fallback."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
from test_moe_routing_runtime import ROOT, external_module, load


def round_trip(worker, method, *args):
    # The actual default encoder rejects callables. No pickle/cloudpickle hook.
    wire = msgspec.msgpack.encode((method, args))
    name, parameters = msgspec.msgpack.decode(wire)
    result = getattr(worker, name)(*parameters)
    return msgspec.msgpack.decode(msgspec.msgpack.encode(result))


class RoutingRPCBoundaryTests(unittest.TestCase):
    def setUp(self):
        metrics = self.metrics = load("rpc_metrics_fixture", "vllm_ascend/observability/moe_routing.py")
        with patch.dict("sys.modules", {"vllm_ascend.observability.moe_routing": metrics}):
            self.rpc = load("rpc_runtime_fixture", "vllm_ascend/observability/moe_routing_rpc.py")
        with torch.inference_mode():
            self.traces = [metrics.RoutingTrace(544, 2, 128, 8, "cpu", detailed=i in (0, 23, 47)) for i in range(48)]
        self.gate_shapes = []
        shapes = self.gate_shapes

        class Gate(torch.nn.Linear):
            def forward(self, hidden):
                shapes.append(tuple(hidden.shape))
                return super().forward(hidden).float(), None

        def routing(hidden, logits, indices_type):
            weights, ids = logits.softmax(-1).topk(8, dim=-1)
            return weights / weights.sum(-1, keepdim=True), ids

        runner = SimpleNamespace(
            gate=Gate(2, 128, bias=False, dtype=torch.bfloat16),
            router=SimpleNamespace(base=SimpleNamespace(_compute_routing=routing)),
        )
        model = SimpleNamespace(
            routing_traces=self.traces,
            routing_config={"scope": "CPU_fixture"},
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=runner))] * 48),
        )

        class Worker(self.rpc.MoeRoutingWorkerExtension):
            def get_model(self):
                return model

        self.worker = Worker()
        npu = patch.object(torch, "npu", external_module(synchronize=lambda: None), create=True)
        npu.start()
        self.addCleanup(npu.stop)

    def test_callable_payload_is_rejected_but_named_arm_round_trips(self):
        with self.assertRaises(TypeError):
            msgspec.msgpack.encode(("apply_model", (self.rpc.reset_probe,)))
        addresses = [trace.ids.data_ptr() for trace in self.traces]
        self.assertEqual(round_trip(self.worker, "moe_routing_arm"), {"armed_layers": 48})
        self.assertEqual(addresses, [trace.ids.data_ptr() for trace in self.traces])
        self.assertTrue(all((trace.ids == -1).all() for trace in self.traces))

    @torch.inference_mode()
    def test_named_save_round_trips_only_metadata_and_requires_fresh_arm(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            dest = str(Path(directory) / "snapshot.pt")
            with self.assertRaisesRegex(ValueError, "arm"):
                round_trip(self.worker, "moe_routing_save", dest)
            round_trip(self.worker, "moe_routing_arm")
            self.record_request()
            reply = round_trip(self.worker, "moe_routing_save", dest)
            self.assertEqual(reply, {"saved": dest, "layers": 48, "rows_per_layer": 544})
            self.assertTrue(Path(dest).is_file())
            payload = torch.load(dest, map_location="cpu", weights_only=True)
            self.assertEqual(len(payload["layers"]), 48)
            self.assertEqual(self.gate_shapes, [(512, 2), *[(1, 2)] * 32] * 3)
            for layer in (0, 23, 47):
                self.assertEqual(payload["layers"][layer]["eager_logits"].shape, (544, 128))
                self.assertEqual(payload["layers"][layer]["eager_ids"].shape, (544, 8))
                self.assertTrue(torch.equal(payload["layers"][layer]["output"], torch.ones(544, 2).bfloat16()))
            with self.assertRaisesRegex(ValueError, "arm"):
                round_trip(self.worker, "moe_routing_save", dest)

    @torch.inference_mode()
    def record_request(self):
        hidden = torch.ones(512, 2, dtype=torch.bfloat16)
        logits = torch.ones(512, 128)
        weights = torch.full((512, 8), 0.125)
        ids = torch.arange(8).expand(512, -1).int()
        for trace in self.traces:
            trace.record(hidden, logits, weights, ids)
            if trace.detailed:
                trace.record_output(hidden)
            for _ in range(32):
                trace.record(hidden[:1], logits[:1], weights[:1], ids[:1])
                if trace.detailed:
                    trace.record_output(hidden[:1])

    def test_actual_driver_and_rpc_save_all_four_requests_without_artifact_collisions(self):
        fixture = self
        instances = []

        class LLM:
            def __init__(self, **kwargs):
                self.closed = False
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        use_v2_model_runner=True,
                        model_config=SimpleNamespace(enable_trace_replay=True),
                        cache_config=SimpleNamespace(enable_prefix_caching=True),
                        compilation_config=SimpleNamespace(mode=3, cudagraph_mode="FULL_DECODE_ONLY"),
                    ),
                    engine_core=SimpleNamespace(shutdown=self.shutdown),
                )
                instances.append(self)

            def shutdown(self):
                self.closed = True

            def reset_prefix_cache(self, **kwargs):
                return True

            def collective_rpc(self, method, args=()):
                return [round_trip(fixture.worker, method, *args)]

            def generate(self, prompts, params, use_tqdm):
                fixture.record_request()
                return [
                    SimpleNamespace(
                        prompt_token_ids=prompts[0]["prompt_token_ids"],
                        num_cached_tokens=0,
                        outputs=[
                            SimpleNamespace(
                                token_ids=params.trace_decode_token_ids,
                                routed_experts=torch.arange(8).expand(544, 48, 8).numpy(),
                            )
                        ],
                    )
                ]

        modules = {
            "vllm": external_module(
                LLM=LLM,
                SamplingParams=SimpleNamespace,
                __file__=str(ROOT.parents[1] / "vllm/vllm/__init__.py"),
                __version__="0.32.0",
            ),
            "vllm_ascend": external_module(__file__=str(ROOT / "vllm_ascend/__init__.py")),
            "vllm_ascend.observability.moe_routing": self.metrics,
        }
        with tempfile.TemporaryDirectory(dir=".") as directory, patch.dict(sys.modules, modules):
            sys.modules["vllm_ascend.observability.moe_forced_tokens"] = load(
                "probe_artifact_force", "vllm_ascend/observability/moe_forced_tokens.py"
            )
            folder = Path(directory)
            manifest = folder / "inputs.json"
            manifest.write_text(json.dumps([{"prompt": [0] * 512, "forced": [1] * 33}] * 4))
            driver = load("probe_artifact_driver", "examples/moe_routing_check.py")
            args = SimpleNamespace(
                phase="probe", source=str(ROOT), inputs=str(manifest), output=str(folder), model="unused"
            )
            with redirect_stdout(io.StringIO()):
                driver.collect(args)
            self.assertEqual(torch.load(folder / "probe-0.pt", weights_only=True).shape, (4, 544, 48, 8))
            for sample in range(4):
                snapshot = torch.load(folder / f"probe-snapshot-{sample}.pt", weights_only=True)
                self.assertEqual(len(snapshot["layers"]), 48)
                self.assertTrue(all(row["device_calls"] == 33 for row in snapshot["layers"]))
            native = torch.load(folder / "probe-0.pt", weights_only=True)
            for repeat in range(2):
                torch.save(native, folder / f"reference-{repeat}.pt")
            report = io.StringIO()
            with (
                patch.object(driver, "validate_model", return_value={"norm_topk_prob": True}),
                patch.object(driver, "load_router_weight", return_value=torch.ones(128, 2).bfloat16()),
                redirect_stdout(report),
            ):
                self.assertEqual(driver.analyze(args), 1)
            events = [json.loads(line.removeprefix("MOE1 ")) for line in report.getvalue().splitlines()]
            self.assertEqual(events[-1]["decision"], "REVIEW_DIFFERENCES")
            self.assertEqual(events[-1]["graph_replay"], "UNVERIFIED")
        self.assertTrue(instances[0].closed)

    def test_invalid_save_path_never_writes_outside_execution_directory(self):
        for dest in ("/not-a-valid-diagnostic.pt", "../escape.pt", None):
            round_trip(self.worker, "moe_routing_arm")
            with self.assertRaises((ValueError, TypeError)):
                round_trip(self.worker, "moe_routing_save", dest)
