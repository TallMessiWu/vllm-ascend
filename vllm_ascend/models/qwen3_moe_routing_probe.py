# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic architecture; the normal Qwen registration is unchanged.

Construct buffers and callbacks before model warmup/capture. No runner monkey
patches or environment switches. The delegated router computes the original
result once, snapshots it, and returns the very same tensors.
"""

from functools import partial

import torch
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from vllm_ascend.observability.moe_routing import TRACE_ROWS, RoutingTrace

DETAILED_LAYERS = (0, 23, 47)


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
