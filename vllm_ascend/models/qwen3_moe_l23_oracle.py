# SPDX-License-Identifier: Apache-2.0
"""Layer 23 reference routing for the single-request Qwen3 MoE diagnostic."""

import torch

from vllm_ascend.models.qwen3_moe_routing_probe import ObservedRouter, Qwen3MoeRoutingProbeForCausalLM
from vllm_ascend.observability.moe_routing import DECODE_ROWS, ORACLE_LAYER, validate_routes


class Layer23OracleRouter(ObservedRouter):
    def __init__(self, base, trace):
        super().__init__(base, trace)
        self.reference_ids = torch.zeros_like(trace.ids)
        self.reference_weights = torch.zeros_like(trace.weights)
        self.enabled = torch.zeros((), dtype=torch.bool, device=trace.ids.device)

    def arm(self, reference):
        self.enabled.zero_()
        if reference is None:
            return
        ids, weights = reference["ids"], reference["weights"]
        validate_routes(ids, weights, self.trace.num_experts)
        if ids.shape != self.reference_ids.shape or weights.dtype != self.reference_weights.dtype:
            raise ValueError("Reference routes do not match the layer 23 buffers")
        self.reference_ids.copy_(ids)
        self.reference_weights.copy_(weights)
        self.enabled.fill_(True)

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = self.base._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
        # The same device cursor drives the trace write below. Keep the enabled
        # tensor and lookup storage stable across capture and request resets.
        positions = self.trace.counts[0].to(torch.int64) + torch.arange(ids.shape[0], device=ids.device)
        positions = positions % self.trace.capacity
        ids = torch.where(self.enabled, self.reference_ids.index_select(0, positions).to(ids.dtype), ids)
        weights = torch.where(self.enabled, self.reference_weights.index_select(0, positions), weights)
        self.trace.record(hidden_states, router_logits, weights, ids)
        return weights, ids


class Qwen3MoeL23OracleForCausalLM(Qwen3MoeRoutingProbeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        runner = self.model.layers[ORACLE_LAYER].mlp.experts
        router = Layer23OracleRouter(runner.router.base, self.routing_traces[ORACLE_LAYER])
        runner.router = router
        runner.routed_experts.router = router
        self.oracle_router = router
        self.continuation_logits = None

    def arm_l23_trial(self, reference):
        self.oracle_router.arm(reference)
        self.continuation_logits = []

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        # Runner V2 applies the LM head in sample_tokens, outside forward graph replay.
        if self.continuation_logits is not None:
            if logits is None or logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("Expected one raw continuation logit row per step")
            self.continuation_logits.append(logits.detach().clone())
        return logits

    def take_continuation_logits(self):
        chunks, self.continuation_logits = self.continuation_logits, None
        self.oracle_router.enabled.zero_()
        if chunks is None or len(chunks) != DECODE_ROWS + 1:
            raise ValueError("Missing or extra continuation logit rows")
        return torch.cat(chunks).float().cpu()
