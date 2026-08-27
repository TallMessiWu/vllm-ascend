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
"""Size of the MXFP8 scale side tables.

The scales cannot live in the KV cache: sharing that allocation would make the
block stride 33/32 of the value bytes, and a value plane with that stride is
not contiguous, which the custom op requires. So the impl allocates them
itself - and vLLM, which budgets only what it allocates, cannot see them.

Both the impl that allocates the tables and the patch that reserves room for
them size them from here. If those two ever disagree the reservation is wrong
by exactly that difference, which is the kind of error that only shows up as
an out-of-memory on somebody else's model.
"""

# The kernel addresses the cache in 128-token blocks whatever block size the
# KV manager hands out, and shares one E8M0 scale per 32 tokens, packed in
# pairs - so a V window spans 64 tokens and yields one packed row.
QFA_KERNEL_BLOCK = 128
QFA_V_GROUP = 64


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
