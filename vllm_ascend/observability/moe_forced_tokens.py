# SPDX-License-Identifier: Apache-2.0
"""Use Runner V2's native trace replay, not a custom logits processor.

The engine must set enable_trace_replay=True. V2 advances the trace from
committed device request lengths, avoiding CPU output-history placeholders.
"""

import math

from vllm import SamplingParams


def forced_continuation(tokens, *, score_tokens=False):
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("The fixed continuation must be a nonempty list of nonnegative integer token ids")
    return SamplingParams(
        n=1,
        temperature=0,
        top_p=1,
        min_p=0,
        max_tokens=len(tokens),
        ignore_eos=True,
        detokenize=False,
        logprobs=0 if score_tokens else None,
        trace_decode_token_ids=list(tokens),
    )


def continuation_nll(output, tokens):
    rows = output.logprobs
    if rows is None or len(rows) != len(tokens):
        raise ValueError("Missing forced-token log probabilities")
    scores = []
    for token, row in zip(tokens, rows, strict=True):
        if row is None or token not in row or not math.isfinite(row[token].logprob) or row[token].logprob > 0:
            raise ValueError("Invalid log probability for the forced token")
        scores.append(-row[token].logprob)
    return scores
