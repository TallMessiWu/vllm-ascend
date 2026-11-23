# SPDX-License-Identifier: Apache-2.0
"""CPU comparison-driver/RPC/artifact round trip. Simulated replay is NOT NPU evidence."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_moe_routing_rpc as rpc_fixture
import torch
from test_moe_routing_runtime import ROOT, external_module, load


def driver_module():
    reference = load("moe_routing_check", "examples/moe_routing_check.py")
    return load("mxfp4_weight_driver_fixture", "examples/moe_mxfp4_w_check.py"), reference


class MXFP4WeightDriverTests(unittest.TestCase):
    def setUp(self):
        self.fixture = rpc_fixture.RoutingRPCBoundaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.driver, self.reference = driver_module()
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.inputs = self.folder / "inputs.json"
        self.inputs.write_text(json.dumps([{"prompt": [0] * 512, "forced": [1] * 33}] * 4))
        self.gate = self.fixture.worker.get_model().model.layers[0].mlp.experts.gate
        with torch.no_grad():
            numbers = torch.arange(128, dtype=torch.float32) / 128
            self.gate.weight.copy_(torch.stack([numbers, numbers.flip(0)], dim=-1))
        self.modules = {
            "vllm_ascend": external_module(__file__=str(ROOT / "vllm_ascend/__init__.py")),
            "vllm_ascend.observability.moe_routing": self.fixture.metrics,
        }
        self.args = SimpleNamespace(
            model="reference",
            candidate="candidate",
            source=str(ROOT),
            inputs=str(self.inputs),
            output=str(self.folder),
            phase="reference",
        )
        self.fixture.hidden = torch.tensor([[1, 0]], dtype=torch.bfloat16)
        self.replay = True

    @torch.inference_mode()
    def record(self):
        hidden = self.fixture.hidden.expand(512, -1)
        logits = self.gate(hidden)[0]
        weights, ids = logits.softmax(-1).topk(8, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        for trace in self.fixture.traces:
            trace.record(hidden, logits, weights, ids)
            if trace.detailed:
                trace.record_output(hidden)
            for _ in range(32):
                trace.record(hidden[:1], logits[:1], weights[:1], ids[:1])
                if trace.detailed:
                    trace.record_output(hidden[:1])
            # Deliberately simulated device/host count split for CPU analyzer
            # success coverage, NOT evidence that an NPU graph executed.
            if self.replay:
                trace.host_calls = 1

    def collect_all(self, unstable=False, no_drift=False):
        test = self
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
                return [rpc_fixture.round_trip(test.fixture.worker, method, *args)]

            def generate(self, prompts, params, use_tqdm):
                test.record()
                logits = test.gate(test.fixture.hidden)[0]
                ids = logits.topk(8, dim=-1).indices
                return [
                    SimpleNamespace(
                        prompt_token_ids=prompts[0]["prompt_token_ids"],
                        num_cached_tokens=0,
                        outputs=[
                            SimpleNamespace(
                                token_ids=params.trace_decode_token_ids, routed_experts=ids.expand(544, 48, 8).numpy()
                            )
                        ],
                    )
                ]

        modules = {
            **self.modules,
            "vllm": external_module(
                LLM=LLM,
                SamplingParams=SimpleNamespace,
                __version__="0.32.0",
                __file__=str(ROOT.parents[1] / "vllm/vllm/__init__.py"),
            ),
        }
        with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            sys.modules["vllm_ascend.observability.moe_forced_tokens"] = load(
                "mx_driver_force_fixture", "vllm_ascend/observability/moe_forced_tokens.py"
            )
            for side in self.driver.SIDES:
                self.args.phase = side
                self.fixture.hidden = torch.tensor(
                    [[1, 0]]
                    if no_drift or side == "reference" or (unstable and side == "candidate-repeat")
                    else [[0, 1]],
                    dtype=torch.bfloat16,
                )
                self.driver.collect(self.args)
        self.assertTrue(all(instance.closed for instance in instances))
        self.assertEqual(len(instances), 3)

    def analyze(self):
        stdout = io.StringIO()
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(self.reference, "validate_model", return_value={"norm_topk_prob": True}),
            patch.object(self.reference, "load_router_weight", return_value=self.gate.weight.detach().clone()),
            redirect_stdout(stdout),
        ):
            code = self.driver.analyze(self.args)
        events = [json.loads(line.removeprefix("MOE1 ")) for line in stdout.getvalue().splitlines()]
        return code, events

    def test_actual_collection_rpc_save_and_analysis_measures_drift_without_failing(self):
        self.collect_all()
        code, events = self.analyze()
        self.assertEqual(code, 0)
        self.assertEqual(events[-1]["decision"], "DRIFT_MEASURED")
        self.assertEqual(events[-1]["quantization_drift"], "OBSERVED")
        self.assertEqual(events[-1]["candidate_repeat"], "IDENTICAL")
        self.assertEqual(events[-1]["model_quality"], "UNVERIFIED")
        rows = [event for event in events if event["event"] == "weight_drift"]
        self.assertEqual(len(rows), 4 * 2 * 48)
        self.assertTrue(all(row["reference_vs_candidate"]["overlap_mean"] == 0 for row in rows))
        self.assertLess(len(events), 700)

    def test_no_set_drift_is_a_valid_measurement_not_a_quality_claim(self):
        self.collect_all(no_drift=True)
        code, events = self.analyze()
        self.assertEqual(code, 0)
        self.assertEqual(events[-1]["quantization_drift"], "NO_SET_CHANGE_IN_THIS_FIXTURE")
        self.assertEqual(events[-1]["model_quality"], "UNVERIFIED")

    def test_mismatched_input_receipt_is_rejected(self):
        self.collect_all()
        receipt = self.folder / "candidate/collection.json"
        metadata = json.loads(receipt.read_text())
        metadata["token_sha256"] = "different_tokens"
        receipt.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.analyze()

    def test_candidate_nondeterminism_is_not_attributed_to_quantization(self):
        self.collect_all(unstable=True)
        code, events = self.analyze()
        self.assertEqual(code, 1)
        self.assertEqual(events[-1]["candidate_repeat"], "REVIEW")
        self.assertEqual(events[-1]["decision"], "REVIEW_EXECUTION")

    def test_missing_replay_is_nonzero_even_when_numerical_comparisons_pass(self):
        self.replay = False
        self.collect_all()
        code, events = self.analyze()
        self.assertEqual(code, 1)
        self.assertEqual(events[-1]["graph_replay"], "UNVERIFIED")

    def test_missing_native_alignment_or_coverage_rejected(self):
        self.collect_all()
        path = self.folder / "candidate/probe-snapshot-0.pt"
        snapshot = torch.load(path, weights_only=True)
        snapshot["layers"][0]["ids"][0, 0] = 42
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.analyze()

    def test_wrong_router_or_arithmetic_requires_review(self):
        self.collect_all()
        path = self.folder / "candidate/probe-snapshot-0.pt"
        snapshot = torch.load(path, weights_only=True)
        snapshot["layers"][0]["eager_logits"][0, 0] += 1
        torch.save(snapshot, path)
        code, events = self.analyze()
        self.assertEqual(code, 1)
        self.assertEqual(events[-1]["capture_vs_eager"], "REVIEW")

    def test_conversion_pair_identity_and_mutation_guards(self):
        roots = [self.folder.resolve() / side for side in ("source-weights", "candidate-weights")]
        for root in roots:
            root.mkdir()
            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
                (root / name).write_text("{}\n")
            (root / "model.safetensors").write_bytes(b"not_loaded_CPU_identity_fixture")
        manifest = {
            "status": "COMPLETE",
            "candidate": "R-M16",
            "source": str(roots[0]),
            "output": str(roots[1]),
            "recipe": {
                "format": "MXFP4_E2M1_E8M0",
                "group_size": 32,
                "axis": -1,
                "scale_selection": "floor_log2_amax_minus_2",
                "rounding": "RNE",
                "carrier": "BF16",
                "activations": "BF16",
                "kv": "BF16",
                "router": "unchanged_BF16",
                "scale_exponent_range": [-127, 127],
                "flush_fp32_subnorms": False,
                "reference_sha": "7bc41952de394f5cc5e782baf132e7c7542eb4e4",
            },
            "statistics": {"changed_values": 1},
        }
        for name, key in (
            ("config.json", "source_config_sha256"),
            ("model.safetensors.index.json", "source_index_sha256"),
        ):
            manifest[key] = hashlib.sha256((roots[0] / name).read_bytes()).hexdigest()
        for root, key in zip(roots, ("source_shards", "output_shards")):
            p = root / "model.safetensors"
            manifest[key] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]}
        metadata = roots[1] / ".mxfp4_w_metadata/manifest.json"
        metadata.parent.mkdir()
        metadata.write_text(json.dumps(manifest))
        with patch.object(self.reference, "validate_model", return_value={"norm_topk_prob": True}):
            self.assertEqual(self.driver.validate_pair(*map(str, roots)), manifest)
            alias = self.folder / "weights-mount"
            alias.symlink_to(self.folder.resolve(), target_is_directory=True)
            source, target = [str(alias.absolute() / root.name) for root in roots]
            manifest.update(source=source, output=target)
            metadata.write_text(json.dumps(manifest))
            self.assertEqual(self.driver.validate_pair(source, target), manifest)
            manifest["statistics"]["changed_values"] = 0
            metadata.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.driver.validate_pair(*map(str, roots))
            manifest["statistics"]["changed_values"] = 1
            metadata.write_text(json.dumps(manifest))
            (roots[1] / ".mxfp4_w_to_bf16.incomplete").touch()
            with self.assertRaises(ValueError):
                self.driver.validate_pair(*map(str, roots))
