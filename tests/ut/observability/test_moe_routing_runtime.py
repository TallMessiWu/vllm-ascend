# SPDX-License-Identifier: Apache-2.0
"""CPU tests at the upstream router/logits-processor boundary, not NPU evidence."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def external_module(**attributes):
    module = ModuleType("external_vllm_fixture")
    module.__dict__.update(attributes)
    return module


class RuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.metrics = load("probe_test_metrics", "vllm_ascend/observability/moe_routing.py")

        class BaseRouter:
            def __init__(self, top_k, global_num_experts):
                self.top_k, self.global_num_experts = top_k, global_num_experts

        def updates(state, batch, factory):
            if batch is not None:
                for row, params, prompt, output in batch.added:
                    value = factory(params, prompt, output)
                    if value is not None:
                        state[row] = value

        self.modules = {
            "vllm.model_executor.layers.fused_moe.router.base_router": external_module(BaseRouter=BaseRouter),
            "vllm.model_executor.models.qwen3_moe": external_module(Qwen3MoeForCausalLM=torch.nn.Module),
            "vllm_ascend.observability.moe_routing": self.metrics,
            "vllm.v1.sample.logits_processor": external_module(LogitsProcessor=object),
            "vllm.v1.sample.logits_processor.builtin": external_module(process_dict_updates=updates),
        }

    def test_router_delegates_once_and_preserves_output_tensor_identity(self):
        with patch.dict(sys.modules, self.modules):
            runtime = load("probe_test_runtime", "vllm_ascend/models/qwen3_moe_routing_probe.py")
        ids = torch.tensor([[0, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]])

        class NativeRouter:
            eplb_state = None
            scoring_func = "softmax"
            use_grouped_topk = False
            top_k = 2
            global_num_experts = 4
            calls = 0
            routing_method_type = "native"

            def _compute_routing(self, *args, **kwargs):
                self.calls += 1
                return weights, ids

        base = NativeRouter()
        trace = self.metrics.RoutingTrace(1, 3, 4, 2, "cpu")
        router = runtime.ObservedRouter(base, trace)
        actual_weights, actual_ids = router._compute_routing(torch.ones(1, 3).bfloat16(), torch.ones(1, 4), None)
        self.assertIs(actual_weights, weights)
        self.assertIs(actual_ids, ids)
        self.assertEqual(base.calls, 1)
        self.assertEqual(trace.ids.tolist(), [[0, 2]])

    def test_fixed_continuation_advances_with_forwarded_output_history(self):
        with patch.dict(sys.modules, self.modules):
            runtime = load("probe_test_force", "vllm_ascend/observability/moe_forced_tokens.py")
        history = []
        params = SimpleNamespace(extra_args={"routing_probe_tokens": [2, 1]}, max_tokens=2)
        processor = runtime.ForcedContinuation(None, "cpu", False)
        processor.update_state(SimpleNamespace(added=[(0, params, [0], history)]))
        first = processor.apply(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(first.argmax(dim=-1).tolist(), [2])
        self.assertEqual(first[0, 2].item(), 3.0)
        history.append(2)
        processor.update_state(None)
        second = processor.apply(torch.tensor([[1.0, 4.0, 3.0]]))
        self.assertEqual(second.argmax(dim=-1).tolist(), [1])
        history.append(1)
        with self.assertRaises(ValueError):
            processor.apply(torch.ones(1, 3))

    def test_diagnostic_architecture_preserves_checkpoint_parameter_names_and_output(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(2048, 128, bias=False, dtype=torch.bfloat16)
                self.shared_expert = None
                self.experts = torch.nn.Module()
                self.experts.gate = self.gate
                self.experts.routed_experts = torch.nn.Module()
                self.experts.router = SimpleNamespace(
                    eplb_state=None, scoring_func="softmax", use_grouped_topk=False, top_k=8, global_num_experts=128
                )

            def forward(self, hidden):
                return hidden + 1

        class ReferenceModel(torch.nn.Module):
            def __init__(self, *, vllm_config, prefix=""):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList()
                for _ in range(48):
                    layer = torch.nn.Module()
                    layer.mlp = Block()
                    self.model.layers.append(layer)

        self.modules["vllm.model_executor.models.qwen3_moe"] = external_module(Qwen3MoeForCausalLM=ReferenceModel)
        pc = SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
        )
        config = SimpleNamespace(
            num_hidden_layers=48, hidden_size=2048, num_experts=128, num_experts_per_tok=8, norm_topk_prob=True
        )
        vc = SimpleNamespace(
            parallel_config=pc,
            model_config=SimpleNamespace(hf_text_config=config),
            quant_config=None,
            speculative_config=None,
            lora_config=None,
            use_v2_model_runner=True,
            compilation_config=SimpleNamespace(mode=3, cudagraph_mode="FULL_DECODE_ONLY"),
        )
        with patch.dict(sys.modules, self.modules):
            runtime = load("probe_model_fixture", "vllm_ascend/models/qwen3_moe_routing_probe.py")
            baseline = ReferenceModel(vllm_config=vc)
            model = runtime.Qwen3MoeRoutingProbeForCausalLM(vllm_config=vc)
        self.assertEqual(list(dict(baseline.named_parameters())), list(dict(model.named_parameters())))
        hidden = torch.zeros(2, 2048, dtype=torch.bfloat16)
        output = model.model.layers[0].mlp(hidden)
        self.assertTrue(torch.equal(output, hidden + 1))
        self.assertTrue(torch.equal(model.routing_traces[0].output[:2], output))
        pc.tensor_parallel_size = 2
        with self.assertRaises(ValueError):
            runtime.Qwen3MoeRoutingProbeForCausalLM(vllm_config=vc)

    def test_reference_driver_requires_cold_prefix_and_closes_engine_on_failure(self):
        """Native routing may include cache hits, so complete IDs alone are insufficient."""
        cases = ((False, True, 0), (True, True, 0), (False, False, 0), (False, True, 128), (False, True, None))
        for fail, reset_result, cached_tokens in cases:
            instances = []

            class LLM:
                def __init__(
                    self,
                    _instances=instances,
                    _failure=fail,
                    _reset_result=reset_result,
                    _cached_tokens=cached_tokens,
                    **kwargs,
                ):
                    if kwargs["enable_return_routed_experts"] and not kwargs["enable_prefix_caching"]:
                        raise ValueError("AuxOutput Connector requires prefix caching.")
                    self.kwargs, self.closed, self.failure = kwargs, False, _failure
                    self.reset_result, self.cached_tokens = _reset_result, _cached_tokens
                    self.cache_resets, self.generated, self.cache_used = 0, 0, False
                    self.llm_engine = SimpleNamespace(
                        vllm_config=SimpleNamespace(
                            use_v2_model_runner=True,
                            cache_config=SimpleNamespace(enable_prefix_caching=kwargs["enable_prefix_caching"]),
                            compilation_config=SimpleNamespace(mode=3, cudagraph_mode="FULL_DECODE_ONLY"),
                        ),
                        engine_core=SimpleNamespace(shutdown=self.shutdown),
                    )
                    _instances.append(self)

                def shutdown(self):
                    self.closed = True

                def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
                    if reset_running_requests or reset_connector:
                        raise AssertionError("Only idle local cache reset is permitted")
                    self.cache_resets += 1
                    self.cache_used = False
                    return self.reset_result

                def generate(self, prompts, params, use_tqdm):
                    self.generated += 1
                    if self.failure:
                        raise RuntimeError("injected external inference failure")
                    result = SimpleNamespace(
                        token_ids=params.extra_args["routing_probe_tokens"],
                        routed_experts=torch.arange(8).expand(544, 48, 8).numpy(),
                    )
                    hit = 512 if self.cache_used else self.cached_tokens
                    self.cache_used = True
                    return [
                        SimpleNamespace(
                            prompt_token_ids=prompts[0]["prompt_token_ids"], outputs=[result], num_cached_tokens=hit
                        )
                    ]

            modules = {
                **self.modules,
                "vllm": external_module(
                    LLM=LLM,
                    SamplingParams=SimpleNamespace,
                    __file__=str(ROOT.parents[1] / "vllm/vllm/__init__.py"),
                    __version__="0.32.0",
                ),
                "vllm_ascend": external_module(__file__=str(ROOT / "vllm_ascend/__init__.py")),
                "vllm_ascend.observability.moe_forced_tokens": external_module(ForcedContinuation=object),
                "vllm_ascend.observability.moe_routing_rpc": external_module(reset_probe=object(), save_probe=object()),
            }
            with tempfile.TemporaryDirectory(dir=".") as directory, patch.dict(sys.modules, modules):
                folder = Path(directory)
                manifest = folder / "inputs.json"
                manifest.write_text(json.dumps([{"prompt": [0] * 512, "forced": [1] * 33}] * 4))
                driver = load("probe_driver_fixture", "examples/moe_routing_check.py")
                args = SimpleNamespace(
                    phase="reference", source=str(ROOT), inputs=str(manifest), output=str(folder), model="unused"
                )
                with redirect_stdout(io.StringIO()):
                    if fail or not reset_result or cached_tokens != 0:
                        error = RuntimeError if fail else ValueError
                        with self.assertRaises(error):
                            driver.collect(args)
                    else:
                        driver.collect(args)
                        self.assertTrue((folder / "reference-1.pt").exists())
            expected_requests = 1 if fail or not reset_result or cached_tokens != 0 else 8
            self.assertEqual(instances[0].cache_resets, expected_requests)
            self.assertEqual(instances[0].generated, expected_requests if reset_result else 0)
            self.assertTrue(instances[0].closed)
            self.assertIs(instances[0].kwargs["enable_prefix_caching"], True)
            self.assertEqual(instances[0].kwargs["max_num_seqs"], 100)
            self.assertEqual(instances[0].kwargs["max_num_batched_tokens"], 16384)
            self.assertEqual(instances[0].kwargs["compilation_config"], {"cudagraph_mode": "FULL_DECODE_ONLY"})

    def test_invalid_forced_sequences_are_rejected_before_sampling(self):
        with patch.dict(sys.modules, self.modules):
            runtime = load("probe_test_force_invalid", "vllm_ascend/observability/moe_forced_tokens.py")
        for tokens in ([], [True], [-1], [1, 2], "token"):
            with self.assertRaises(ValueError):
                runtime.ForcedContinuation.validate_params(
                    SimpleNamespace(extra_args={"routing_probe_tokens": tokens}, max_tokens=1)
                )
