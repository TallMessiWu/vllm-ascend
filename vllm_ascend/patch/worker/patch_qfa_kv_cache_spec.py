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

import dataclasses

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.attention.qfa_page import QFA_CACHE_DTYPE, qfa_fp8_attention_page

_original_get_kv_cache_spec = Attention.get_kv_cache_spec

# quant_flash_attn addresses the cache in 128-token blocks whatever block size
# the KV manager hands out, so any block size here stays a multiple of it.
_KERNEL_BLOCK = 128


def _qfa_get_kv_cache_spec(self: Attention, vllm_config: VllmConfig) -> KVCacheSpec | None:
    """Declare one byte per element for layers whose KV cache is MXFP8.

    Only the FP8 values live in the cache; their E8M0 scales are side tables
    the impl owns, so the page is exactly half the BF16 one. Halving it is not
    an optimization but a correctness requirement: the allocator hands blocks
    out by page, and a layer that stores 1 byte per element inside a page sized
    for 2 puts its block b on the page belonging to block b // 2 - which the
    hybrid allocator has given to somebody else.

    Platform setup has already sized cache_config.block_size for that halved
    page, so the FP8 page computed here should equal the model's shared page
    rather than be padded up to a BF16-sized one. Layers that stay BF16 need
    the opposite correction; see below.
    """
    spec = _original_get_kv_cache_spec(self, vllm_config)
    if not getattr(self.impl, "qfa_decode_enabled", False):
        if type(spec) is not FullAttentionSpec or not qfa_fp8_attention_page(vllm_config):
            return spec
        # Platform setup sized cache_config.block_size for an FP8 page, but
        # this layer serves BF16 - the MTP drafter, or anything the impl's
        # per-layer conditions rule out. Left alone it would pack twice the
        # bytes into that block, making its page the largest in the model, and
        # unify_hybrid_kv_cache_specs would pad every other page up to it -
        # costing more than the FP8 cache saves. Half the block puts it back
        # on the shared page: same bytes per page, half the tokens in it.
        if spec.block_size % (2 * _KERNEL_BLOCK) or spec.block_size < 2 * _KERNEL_BLOCK:
            raise AssertionError(
                f"{self.layer_name}: cannot halve a block of {spec.block_size} tokens onto the shared "
                f"FP8 page and stay a multiple of the {_KERNEL_BLOCK}-token kernel block"
            )
        halved = dataclasses.replace(spec, block_size=spec.block_size // 2)
        logger.info(
            "QFA MXFP8 spec: %s stays %s, block_size %d -> %d to hold the shared %d-byte page",
            self.layer_name,
            spec.dtype,
            spec.block_size,
            halved.block_size,
            halved.page_size_bytes,
        )
        return halved
    if type(spec) is not FullAttentionSpec:
        raise AssertionError(f"QFA decode expects a plain FullAttentionSpec, got {type(spec).__name__}")
    new_spec = dataclasses.replace(spec, dtype=QFA_CACHE_DTYPE)
    # The page is what the allocator budgets, and it only shrinks if nothing
    # pads it back up. Log both sides of the rewrite: with platform setup
    # sizing the block for FP8, the new page should now be the model's shared
    # page. If a later `GPU KV cache size` still matches the BF16 run's, the
    # alignment did not take and the saving evaporated again.
    logger.info(
        "QFA MXFP8 spec: %s dtype %s -> %s, page %d -> %d bytes (real %d -> %d), block_size %d",
        self.layer_name,
        spec.dtype,
        new_spec.dtype,
        spec.page_size_bytes,
        new_spec.page_size_bytes,
        spec.real_page_size_bytes,
        new_spec.real_page_size_bytes,
        new_spec.block_size,
    )
    return new_spec


Attention.get_kv_cache_spec = _qfa_get_kv_cache_spec
