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
"""Window bookkeeping for the MXFP8 V cache.

This arithmetic decides which bytes a decode step rewrites and which of them
are allowed to set the window's shared scale. Getting it wrong does not raise -
it quietly writes one request's values into another's window, or lets a
rejected speculative token coarsen history that should never have moved. An
earlier version of this code was indexed by position in the batch, which vLLM
reassigns whenever a request finishes, and nothing caught it.

It lives in qfa_scale.py rather than inside the attention impl so it can be
exercised as plain tensor arithmetic, with no NPU and no operator involved.
"""

import torch

from vllm_ascend.attention.qfa_scale import (
    QFA_KERNEL_BLOCK,
    QFA_V_GROUP,
    plan_v_windows,
    qfa_scale_bytes_per_kernel_block,
)

# Block table where request r's c-th kernel block has a distinct id, so a slot
# computed from the wrong request or the wrong column is visible in the value.
BLOCK_TABLE = torch.arange(4 * 8).reshape(4, 8)


def plan(ctx: list[int], seq: list[int], columns: int = 8) -> dict[str, list[int]]:
    table = BLOCK_TABLE if columns == 8 else torch.arange(4 * columns).reshape(4, columns)
    out = plan_v_windows(torch.tensor(ctx), torch.tensor(seq), table, len(ctx))
    return {k: v.tolist() for k, v in out.items()}


def test_scale_table_size_is_one_thirtysecond_of_the_values():
    # The impl allocates from this and a worker patch reserves memory from it;
    # if they disagree the reservation is short by exactly the difference.
    for num_kv_heads, head_size in ((4, 256), (8, 128), (2, 64)):
        values = 2 * QFA_KERNEL_BLOCK * num_kv_heads * head_size
        assert qfa_scale_bytes_per_kernel_block(num_kv_heads, head_size) * 32 == values


def test_plain_decode_confirms_everything_it_writes():
    # One token per request: there is nothing speculative to hold back, so the
    # confirmed count must equal the valid count and the two-write split in
    # _qfa_write_value_decode collapses to the single write it replaced.
    p = plan([100, 60], [101, 61])
    assert p["confirmed"] == p["valid"]
    assert p["windows"] == [1, 1, 0, 0]  # neither request crossed a boundary


def test_verify_step_holds_back_exactly_the_drafts():
    p = plan([100, 60], [104, 64])  # 1 + 3 tokens per request
    assert p["valid"] == [40, 40, 64, 64]
    # ctx + 1: the first query token is the one the previous step sampled.
    assert p["confirmed"] == [37, 37, 61, 61]
    assert [v - c for v, c in zip(p["valid"], p["confirmed"])] == [3, 3, 3, 3]


def test_step_crossing_a_window_leaves_the_new_window_unconfirmed():
    # ctx=62, so the confirmed tokens are 0..62 and the drafts open window 1.
    p = plan([62], [66])
    assert p["windows"] == [0, 1]
    assert p["valid"] == [QFA_V_GROUP, 2]
    # Window 1 holds nothing but drafts. It has no confirmed scale to be held
    # at, and _qfa_clamp_to_scale must leave such a row alone rather than clamp
    # it to nothing - the zero here is what tells it so.
    assert p["confirmed"] == [63, 0]


def test_windows_address_the_right_kernel_block():
    # A 128-token kernel block holds two 64-token V windows, so window 2 is the
    # first one of the request's second block, not its first.
    p = plan([126], [130])
    assert p["windows"] == [1, 2]
    per_block = QFA_KERNEL_BLOCK // QFA_V_GROUP
    assert p["window_slots"] == [
        int(BLOCK_TABLE[0, 0]) * per_block + 1,
        int(BLOCK_TABLE[0, 1]) * per_block + 0,
    ]


def test_slots_follow_the_request_not_its_row():
    # Two requests whose block tables share no ids: a slot derived from the
    # wrong row would land in the other request's window.
    p = plan([10, 10], [11, 11])
    assert p["window_slots"][:2] == [int(BLOCK_TABLE[0, 0]) * 2] * 2
    assert p["window_slots"][2:] == [int(BLOCK_TABLE[1, 0]) * 2] * 2


def test_rollback_shrinks_the_valid_span():
    # A rejected speculative token leaves committed bytes past the tail. They
    # stay in the cache, so `valid` is what keeps them out of the next scale.
    long = plan([101], [105])["valid"]
    after_rollback = plan([101], [102])["valid"]
    assert after_rollback[0] < long[0]


def test_padded_row_degenerates_to_a_duplicate_write():
    # A row with no new tokens must not address a window past its own length.
    p = plan([64], [64])
    assert p["windows"] == [0, 0]
    assert p["window_slots"][0] == p["window_slots"][1]
    assert p["confirmed"] == p["valid"] == [QFA_V_GROUP, QFA_V_GROUP]


def test_confirmed_never_exceeds_valid():
    # The clamp bound is derived from the confirmed tokens and applied to the
    # valid ones; if confirmed could exceed valid it would be derived from
    # bytes that are not there.
    for ctx, seq in ((0, 1), (1, 5), (63, 64), (63, 67), (127, 131), (4095, 4099)):
        p = plan([ctx], [seq], columns=64)  # room for a 4k sequence
        assert all(c <= v for c, v in zip(p["confirmed"], p["valid"]))
        assert all(0 <= v <= QFA_V_GROUP for v in p["valid"])


def test_one_token_step_confirms_everything_at_every_boundary():
    # What makes the static spec_verify safe. The split write is now taken on
    # every target step, not only the ones attn_state calls SpecDecoding,
    # because attn_state is ChunkedPrefill during graph capture and a captured
    # single write would lose the split for every replay. That is only free if
    # a one-token step has nothing to hold back: confirmed == valid, so the
    # clamp bound is the window's own scale and both writes land on the same
    # bytes. Swept across window and kernel-block edges, where the two counts
    # are derived from different windows and could disagree.
    for ctx in (0, 1, 31, 32, 62, 63, 64, 126, 127, 128, 191, 4094, 4095):
        p = plan([ctx], [ctx + 1], columns=64)
        assert p["confirmed"] == p["valid"], f"ctx={ctx}: {p['confirmed']} != {p['valid']}"
