#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Sizing and window arithmetic for the MXFP8 KV cache.

Size of the scale side tables.

The scales cannot live in the KV cache: sharing that allocation would make the
block stride 33/32 of the value bytes, and a value plane with that stride is
not contiguous, which the custom op requires. So the impl allocates them
itself - and vLLM, which budgets only what it allocates, cannot see them.

Both the impl that allocates the tables and the patch that reserves room for
them size them from here. If those two ever disagree the reservation is wrong
by exactly that difference, which is the kind of error that only shows up as
an out-of-memory on somebody else's model.
"""

import torch

# The kernel addresses the cache in 128-token blocks whatever block size the
# KV manager hands out, and shares one E8M0 scale per 32 tokens, packed in
# pairs - so a V window spans 64 tokens and yields one packed row.
QFA_KERNEL_BLOCK = 128
QFA_V_GROUP = 64
# Ints in the work-split blob the AICPU metadata op writes. Fixed on that side
# (quant_flash_attn_metadata_torch_adpt.h: METADATA_SIZE), and named here
# because the graph path has to allocate a resident buffer of exactly that size
# before the op ever runs.
QFA_METADATA_INTS = 4096


def qfa_scale_bytes_per_kernel_block(num_kv_heads: int, head_size: int) -> int:
    """Bytes of E8M0 scale one 128-token kernel block needs, K and V together.

    Mirrors the two allocations in `_qfa_attach_fp8_cache` exactly:
      K: (blocks, 128, N, D // 64, 2) - per token, per 64-channel group
      V: (blocks, 128 // 64, N, D, 2) - per 64-token window, per channel
    Both work out to 4 * N * D, so a block costs 8 * N * D, or 1/32 of the
    262144 value bytes it holds.
    """
    k_bytes = QFA_KERNEL_BLOCK * num_kv_heads * (head_size // 64) * 2
    v_bytes = (QFA_KERNEL_BLOCK // QFA_V_GROUP) * num_kv_heads * head_size * 2
    return k_bytes + v_bytes


def plan_v_windows(
    ctx_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    num_reqs: int,
    group: int = QFA_V_GROUP,
) -> dict[str, torch.Tensor]:
    """Which V windows a decode step touches, and how much of each is real.

    Kept out of the attention impl so it can be tested without an NPU. Every
    field is a device tensor derived from device tensors - the host-side
    lengths are optimistic under async scheduling, and one number read back
    from the device would stall the scheduler.

    Returns, for a batch of `num_reqs` requests:
      windows       (2R,) the two V windows per request this step may write:
                    the one holding the first new token and the one holding
                    the last. At most 1 + num_spec tokens arrive, so those are
                    the same window or adjacent ones, and the pair is a fixed
                    shape whatever the batch does - which aclgraph needs.
      window_slots  (2R,) where those windows sit in the value plane. A
                    128-token kernel block holds two of them.
      valid         (2R,) tokens of each window that are inside the request's
                    current length. Everything past it is a rejected
                    speculative token whose bytes are still committed, and it
                    must not be allowed to raise the window's scale.
      confirmed     (2R,) tokens of each window that cannot be rejected: the
                    context plus the first query token, which is the token the
                    previous step sampled. Equals `valid` whenever the step
                    carries one token per request.

    A request with no new tokens (a padded row) collapses to a duplicate
    write of its last window, which is harmless.
    """
    lens = seq_lens[:num_reqs]
    ctx = ctx_lens[:num_reqs]
    last_window = torch.div(lens - 1, group, rounding_mode="floor").clamp(min=0)
    first_window = torch.minimum(torch.div(ctx, group, rounding_mode="floor"), last_window)
    windows = torch.stack([first_window, last_window], dim=1).reshape(-1)

    per_block = QFA_KERNEL_BLOCK // group
    win_reqs = torch.arange(num_reqs, device=lens.device).repeat_interleave(2)
    blocks = block_tables[win_reqs, torch.div(windows, per_block, rounding_mode="floor")].to(torch.int64)
    window_slots = blocks * per_block + (windows % per_block)

    def _fill(limit: torch.Tensor) -> torch.Tensor:
        return (limit.repeat_interleave(2) - windows * group).clamp(0, group)

    return {
        "first_window": first_window,
        "last_window": last_window,
        "windows": windows,
        "window_slots": window_slots,
        "valid": _fill(lens),
        "confirmed": _fill(ctx + 1),
    }
