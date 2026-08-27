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

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

_original_get_kv_cache_spec = Attention.get_kv_cache_spec


def _qfa_get_kv_cache_spec(self: Attention, vllm_config: VllmConfig) -> KVCacheSpec | None:
    """Declare one byte per element for layers whose KV cache is MXFP8.

    Only the FP8 values live in the cache; their E8M0 scales are side tables
    the impl owns, so the page is exactly half the BF16 one. Halving it is not
    an optimization but a correctness requirement: the allocator hands blocks
    out by page, and a layer that stores 1 byte per element inside a page sized
    for 2 puts its block b on the page belonging to block b // 2 - which the
    hybrid allocator has given to somebody else.
    """
    spec = _original_get_kv_cache_spec(self, vllm_config)
    if not getattr(self.impl, "qfa_decode_enabled", False):
        return spec
    if type(spec) is not FullAttentionSpec:
        raise AssertionError(f"QFA decode expects a plain FullAttentionSpec, got {type(spec).__name__}")
    new_spec = dataclasses.replace(spec, dtype=torch.float8_e4m3fn)
    # The page is what the allocator budgets, and it only shrinks if nothing
    # pads it back up. On hybrid models the mamba page is padded to match a
    # BF16 attention page during platform setup, before this patch runs, so
    # the halved page can get padded straight back and the memory saving
    # silently evaporates. Log both so the run says which one happened.
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
