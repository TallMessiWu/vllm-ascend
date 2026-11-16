# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed continuation for diagnostic teacher forcing, not quality scoring."""

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class ForcedContinuation(LogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        tokens = params.extra_args and params.extra_args.get("routing_probe_tokens")
        if tokens is not None and (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(tokens) != params.max_tokens
        ):
            raise ValueError("routing_probe_tokens must match max_tokens and contain nonnegative integer ids")

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        def state(params, prompt_ids, output_ids):
            self.validate_params(params)
            tokens = params.extra_args and params.extra_args.get("routing_probe_tokens")
            return (tokens, output_ids) if tokens is not None else None

        process_dict_updates(self.requests, batch_update, state)

    def apply(self, logits):
        for row, (tokens, output_ids) in self.requests.items():
            step = len(output_ids)
            if step >= len(tokens) or tokens[step] >= logits.shape[1]:
                raise ValueError("Fixed continuation exhausted or outside vocabulary")
            token = tokens[step]
            value = logits[row, token].clone()
            logits[row].fill_(-torch.inf)
            logits[row, token] = value
        return logits
