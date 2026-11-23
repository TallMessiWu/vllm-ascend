# SPDX-License-Identifier: Apache-2.0
"""Load selected reference projections through the model's original loader."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_attention_fallback(load_weights, weights, source, names):
    root = Path(source).resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    replacements = {}
    for name in sorted(names):
        shard = (root / index[name]).resolve()
        if not shard.is_relative_to(root):
            raise ValueError("Reference attention shard escapes the checkpoint")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            replacements[name] = handle.get_tensor(name)

    seen, changed = set(), set()

    def overlay():
        for name, original in weights:
            if name not in replacements:
                yield name, original
                continue
            replacement = replacements[name]
            if (
                name in seen
                or original.shape != replacement.shape
                or original.dtype != torch.bfloat16
                or replacement.dtype != original.dtype
            ):
                raise ValueError(f"Invalid attention replacement: {name}")
            seen.add(name)
            if not torch.equal(original.cpu(), replacement):
                changed.add(name)
            yield name, replacement

    loaded = load_weights(overlay())
    if seen != names or not changed:
        raise ValueError("Attention replacement was incomplete or had no numerical effect")
    return loaded, seen, changed
