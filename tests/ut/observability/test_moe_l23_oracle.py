# SPDX-License-Identifier: Apache-2.0
"""Exercise route alignment and prediction metrics without a device runtime."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from test_moe_mxfp4_w_check import driver_module
from test_moe_routing_runtime import external_module, load


class Layer23OracleTests(unittest.TestCase):
    def setUp(self):
        self.attention_loader = load("attention_loader_fixture", "vllm_ascend/observability/moe_attention_fallback.py")

    def test_serving_candidate_only_changes_the_sixteen_selected_projections(self):
        class NativeModel(torch.nn.Module):
            def __init__(self, *, vllm_config, prefix):
                super().__init__()

            def load_weights(self, weights):
                self.received = dict(weights)
                return set(self.received)

        modules = {
            "vllm.model_executor.models.qwen3_moe": external_module(Qwen3MoeForCausalLM=NativeModel),
            "vllm.logger": external_module(logger=SimpleNamespace(info=lambda *args: None)),
            "vllm_ascend.observability.moe_attention_fallback": self.attention_loader,
            "vllm_ascend.observability.moe_routing": load(
                "gpqa_serving_routes_fixture", "vllm_ascend/observability/moe_routing.py"
            ),
        }
        with patch.dict(sys.modules, modules):
            runtime = load("gpqa_serving_fixture", "vllm_ascend/models/qwen3_moe_attention_fallback.py")
        self.assertEqual(len(runtime.ATTENTION_WEIGHTS), 16)
        reference = {name: torch.ones(2, 2, dtype=torch.bfloat16) for name in runtime.ATTENTION_WEIGHTS}
        original = {name: torch.zeros_like(weight) for name, weight in reference.items()}
        outside = "model.layers.11.self_attn.q_proj.weight"
        original[outside] = torch.full((2, 2), 2, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder)
            shard = "model.safetensors"
            save_file(reference, root / shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(reference, shard)})
            )
            for source in (None, root):
                config = SimpleNamespace(l12_15_attention_source=source)
                args = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config))
                model = runtime.Qwen3MoeAttentionFallbackForCausalLM(vllm_config=args)
                self.assertEqual(model.load_weights(iter(original.items())), set(original))
                expected = original if source is None else original | reference
                for name, tensor in expected.items():
                    torch.testing.assert_close(model.received[name], tensor)
                self.assertIs(model.received[outside], original[outside])

    def test_fallback_reuses_the_completed_independent_input_manifest(self):
        comparison, _ = driver_module()
        with patch.dict(sys.modules, {"moe_mxfp4_w_check": comparison}):
            oracle = load("moe_l23_oracle", "examples/moe_l23_oracle.py")
            driver = load("l23_fallback_example_fixture", "examples/moe_l23_attn_fallback.py")
        body = json.dumps([{"prompt": [1], "forced": [2]}], sort_keys=True)
        fixture = {
            "event": "fixed_inputs",
            "input_set": "independent_prose_v1",
            "token_sha256": hashlib.sha256(body.encode()).hexdigest(),
        }
        result = {"event": "result", "decision": "L23_ORACLE_MEASURED"}
        with tempfile.TemporaryDirectory(dir=".") as folder, patch.object(comparison, "validate_pair"):
            run = Path(folder)
            output = run / "fallback"
            output.mkdir()
            (run / "inputs.json").write_text(body)
            report = run / "report.txt"
            report.write_text(
                f"原始产物={run}\nMOE1 {json.dumps(fixture)}\nMOE1 {json.dumps(result)}\n", encoding="utf-8"
            )
            args = SimpleNamespace(reference="reference", candidate="candidate", previous_report=report)
            driver.prepare(args, output)
            self.assertEqual((output / "inputs.json").read_text(), body + "\n")
            self.assertEqual((output / "baseline-run.txt").read_text(), str(run))
            self.assertIs(driver.prediction.report_predictions, oracle.report_predictions)

    def test_fallback_only_replaces_layer23_attention_in_the_original_loader(self):
        traces = load("l23_fallback_trace_fixture", "vllm_ascend/observability/moe_routing.py")

        class ProbeModel(torch.nn.Module):
            def load_weights(self, weights):
                self.received = dict(weights)
                return set(self.received)

        modules = {
            "vllm_ascend.models.qwen3_moe_routing_probe": external_module(
                ObservedRouter=object, Qwen3MoeRoutingProbeForCausalLM=ProbeModel
            ),
            "vllm_ascend.observability.moe_routing": traces,
            "vllm_ascend.observability.moe_attention_fallback": self.attention_loader,
        }
        with patch.dict(sys.modules, modules):
            runtime = load("l23_attn_weight_fixture", "vllm_ascend/models/qwen3_moe_l23_oracle.py")
        model_type = runtime.Qwen3MoeL23OracleForCausalLM
        model = model_type.__new__(model_type)
        ProbeModel.__init__(model)
        model.routing_config = {}
        scopes = {None: runtime.ATTENTION_WEIGHTS}
        scopes.update({band: traces.attention_projection_names(layers) for band, layers in traces.MIDDLE_BANDS.items()})
        self.assertEqual(
            scopes["layers_8_11"] | scopes["layers_12_15"], traces.attention_projection_names(range(8, 16))
        )
        self.assertFalse(scopes["layers_8_11"] & scopes["layers_12_15"])
        all_names = frozenset().union(*scopes.values())
        reference_weights = {name: torch.ones(2, 2, dtype=torch.bfloat16) for name in all_names}
        candidate_weights = {name: torch.zeros_like(tensor) for name, tensor in reference_weights.items()}
        non_target = "model.layers.23.mlp.gate.weight"
        candidate_weights[non_target] = torch.full((2, 2), 2, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder)
            shard = "model.safetensors"
            save_file(reference_weights, root / shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(reference_weights, shard)})
            )
            model.attention_reference = root
            for label, expected in scopes.items():
                model.attention_band = label
                self.assertEqual(model.load_weights(iter(candidate_weights.items())), set(candidate_weights))
                self.assertEqual(set(model.routing_config["attention_fallback_loaded"]), expected)
                for name, tensor in candidate_weights.items():
                    if name in expected:
                        torch.testing.assert_close(model.received[name], reference_weights[name])
                    else:
                        self.assertIs(model.received[name], tensor)
            with self.assertRaisesRegex(ValueError, "native expert selection"):
                model.arm_l23_trial({"ids": []})

    def test_prepare_uses_four_independent_passages_with_the_same_token_budget(self):
        comparison, _ = driver_module()
        with patch.dict(sys.modules, {"moe_mxfp4_w_check": comparison}):
            driver = load("l23_independent_example_fixture", "examples/moe_l23_oracle.py")

        words = {}

        def encode(text, *, add_special_tokens):
            self.assertFalse(add_special_tokens)
            return [words.setdefault(word, len(words)) for word in text.split()]

        tokenizer = SimpleNamespace(encode=encode)
        transformers = external_module(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
        with (
            tempfile.TemporaryDirectory(dir=".") as folder,
            patch.dict(sys.modules, {"transformers": transformers}),
            patch.object(comparison, "validate_pair"),
            patch.object(driver.reference, "emit") as emit,
        ):
            output = Path(folder)
            driver.prepare(SimpleNamespace(reference="reference", candidate="candidate"), output)
            samples = json.loads((output / "inputs.json").read_text())
            self.assertEqual([(len(s["prompt"]), len(s["forced"])) for s in samples], [(512, 33)] * 4)
            self.assertEqual(len({tuple(s["forced"]) for s in samples}), 4)
            self.assertEqual(emit.call_args.kwargs["input_set"], "independent_prose_v1")

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
            "vllm_ascend.observability.moe_attention_fallback": self.attention_loader,
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
        model.attention_reference = None
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

    def test_vectorized_layer_drift_matches_existing_sparse_route_metrics(self):
        comparison, _ = driver_module()
        with patch.dict(sys.modules, {"moe_mxfp4_w_check": comparison}):
            load("moe_l23_oracle", "examples/moe_l23_oracle.py")
            load("moe_l23_attn_fallback", "examples/moe_l23_attn_fallback.py")
            driver = load("upstream_bands_example_fixture", "examples/moe_upstream_bands.py")
        traces = load("upstream_bands_traces_fixture", "vllm_ascend/observability/moe_routing.py")
        ref = torch.tensor([[[[1, 0], [2, 3]], [[0, 3], [1, 2]]]], dtype=torch.int32)
        ids = torch.tensor([[[[0, 1], [2, 0]], [[0, 2], [2, 1]]]], dtype=torch.int32)
        ref_w = torch.tensor([[[[0.75, 0.25], [0.5, 0.5]], [[0.8, 0.2], [0.8, 0.2]]]])
        weights = torch.tensor([[[[0.2, 0.8], [0.4, 0.6]], [[0.7, 0.3], [0.2, 0.8]]]])
        summary = driver.layer_metrics(ref, ids, ref_w, weights)
        for layer in range(ref.shape[2]):
            expected = traces.routing_metrics(
                ref[0, :, layer], ids[0, :, layer], ref_w[0, :, layer], weights[0, :, layer], 4
            )
            self.assertAlmostEqual(summary["top8_overlap"][layer], expected["overlap_mean"])
            self.assertAlmostEqual(summary["top8_set_equal"][layer], expected["set_equal_rate"])
            self.assertAlmostEqual(summary["weight_l1"][layer], expected["weight_l1_mean"])

    def test_split_reuses_only_the_parent_run_on_the_same_controls(self):
        comparison, _ = driver_module()
        with patch.dict(sys.modules, {"moe_mxfp4_w_check": comparison}):
            load("moe_l23_oracle", "examples/moe_l23_oracle.py")
            load("moe_l23_attn_fallback", "examples/moe_l23_attn_fallback.py")
            driver = load("middle_bands_example_fixture", "examples/moe_upstream_bands.py")
        controls = {"baseline_run": "outputs/independent-run", "token_sha256": "same-fixed-tokens"}
        result = {"event": "result", "decision": "UPSTREAM_BANDS_MEASURED", "graph_replay": "GREEN"}
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder)
            report = root / "report.txt"
            report.write_text(
                f"原始产物={root}\nMOE1 {json.dumps({'event': 'fallback_inputs', **controls})}\n"
                f"MOE1 {json.dumps(result)}\n",
                encoding="utf-8",
            )
            with patch.object(driver.reference, "emit"):
                driver.prepare_parent_report(report, root, controls)
            self.assertEqual((root / "parent-run.txt").read_text(), str(root))
            with self.assertRaisesRegex(ValueError, "same archived controls"):
                driver.prepare_parent_report(report, root, {**controls, "token_sha256": "different-fixed-tokens"})
