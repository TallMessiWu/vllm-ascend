# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic architecture; the normal Qwen registration is unchanged.

Construct buffers and callbacks before model warmup/capture. No runner monkey
patches or environment switches. The delegated router computes the original
result once, snapshots it, and returns the very same tensors.
"""

import json
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from vllm_ascend.observability.moe_routing import TRACE_ROWS, RoutingTrace

DETAILED_LAYERS = (0, 23, 47)
L0_ATTENTION_WEIGHTS = frozenset(f"model.layers.0.self_attn.{projection}_proj.weight" for projection in "qkvo")


def load_l0_attention_weights(path):
    root = Path(path).resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in sorted(L0_ATTENTION_WEIGHTS):
        shard = (root / index[name]).resolve()
        if not shard.is_relative_to(root):
            raise ValueError("Layer 0 reference shard escapes the checkpoint")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weights[name] = handle.get_tensor(name)
    return weights


class ObservedRouter(BaseRouter):
    def __init__(self, base, trace):
        if base.eplb_state is not None or base.scoring_func != "softmax" or base.use_grouped_topk:
            raise ValueError("Diagnostic router requires plain softmax without EPLB/grouped top-k")
        super().__init__(base.top_k, base.global_num_experts)
        self.base, self.trace = base, trace

    @property
    def routing_method_type(self):
        return self.base.routing_method_type

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = self.base._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
        self.trace.record(hidden_states, router_logits, weights, ids)
        return weights, ids


def _capture_output(trace, module, inputs, output):
    trace.record_output(output)


class Qwen3MoeRoutingProbeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        pc = vllm_config.parallel_config
        config = vllm_config.model_config.hf_text_config
        sizes = (
            pc.tensor_parallel_size,
            pc.data_parallel_size,
            pc.pipeline_parallel_size,
            pc.prefill_context_parallel_size,
            pc.decode_context_parallel_size,
        )
        if any(size != 1 for size in sizes) or pc.enable_eplb or pc.use_sequence_parallel_moe:
            raise ValueError("Routing probe supports one TP/DP/PP/PCP/DCP rank, no EPLB/SP")
        if (config.num_hidden_layers, config.hidden_size, config.num_experts, config.num_experts_per_tok) != (
            48,
            2048,
            128,
            8,
        ):
            raise ValueError("Routing probe requires the supported plain Qwen3 MoE shape")
        if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Reference-only probe requires unquantized BF16 carriers and no speculation")
        if vllm_config.lora_config is not None or not vllm_config.use_v2_model_runner:
            raise ValueError("Routing probe requires Runner V2 without LoRA")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.l0_attention_source = getattr(config, "routing_probe_l0_attention_source", None)
        self.routing_traces = []
        self.routing_config = {
            "compilation_mode": str(vllm_config.compilation_config.mode),
            "cudagraph_mode": str(vllm_config.compilation_config.cudagraph_mode),
            "norm_topk_prob": bool(config.norm_topk_prob),
        }
        for layer_id, layer in enumerate(self.model.layers):
            block = layer.mlp
            runner = block.experts
            gate = runner.gate
            if block.shared_expert is not None or hasattr(gate, "weight_fp32"):
                raise ValueError("Shared experts/precast router are outside this diagnostic contract")
            if gate.weight.dtype != torch.bfloat16:
                raise ValueError("Expected source-preserving BF16 router parameters")
            trace = RoutingTrace(
                TRACE_ROWS,
                config.hidden_size,
                config.num_experts,
                config.num_experts_per_tok,
                gate.weight.device,
                detailed=layer_id in DETAILED_LAYERS,
            )
            router = ObservedRouter(runner.router, trace)
            runner.router = router
            runner.routed_experts.router = router
            if trace.detailed:
                block.register_forward_hook(partial(_capture_output, trace))
            self.routing_traces.append(trace)

    def load_weights(self, weights):
        if self.l0_attention_source is None:
            return super().load_weights(weights)

        replacements = load_l0_attention_weights(self.l0_attention_source)
        seen, changed = set(), set()

        def overlay():
            for name, original in weights:
                if name not in replacements:
                    yield name, original
                    continue
                replacement = replacements[name]
                if name in seen or replacement.shape != original.shape or replacement.dtype != torch.bfloat16:
                    raise ValueError(f"Invalid layer 0 attention replacement: {name}")
                if original.dtype != replacement.dtype:
                    raise ValueError(f"Layer 0 candidate projection was not BF16: {name}")
                seen.add(name)
                if not torch.equal(original.cpu(), replacement):
                    changed.add(name)
                yield name, replacement

        loaded = super().load_weights(overlay())
        if seen != L0_ATTENTION_WEIGHTS or not changed:
            raise ValueError("Layer 0 attention replacement was incomplete or had no numerical effect")
        self.routing_config["l0_attention_loaded"] = sorted(seen)
        self.routing_config["l0_attention_changed"] = sorted(changed)
        return loaded
