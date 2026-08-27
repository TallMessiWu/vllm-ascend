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

# quant_flash_attn addresses the cache in 128-token blocks whatever block size
# the KV manager hands out, so the doubled block stays a multiple of it.
_KERNEL_BLOCK = 128


def _qfa_get_kv_cache_spec(self: Attention, vllm_config: VllmConfig) -> KVCacheSpec | None:
    """Declare one byte per element for layers whose KV cache is MXFP8.

    Only the FP8 values live in the cache; their E8M0 scales are side tables
    the impl owns, so a block of the same length is exactly half the page.
    Getting the page right is a correctness requirement before it is an
    optimization: the allocator hands blocks out by page, so a layer storing
    one byte per element inside a page sized for two puts its block b on the
    page belonging to block b // 2, which the hybrid allocator has given to
    somebody else.

    Doubling the block restores the page the rest of the model shares, and the
    doubling belongs here rather than in the platform's block sizing. Platform
    setup pads the mamba page to whatever attention page it computes, and
    EngineCore then rewrites `cache_config.block_size` to the smallest block
    among the KV cache groups (vllm/v1/engine/core.py) before this runs again.
    A block sized for FP8 over there therefore feeds back into itself: with an
    MTP drafter present the second pass halved the page and the run died.
    Doubling only this layer's block leaves the smallest block untouched, so
    the second pass sees exactly what the first one did.

    Net effect: the page is unchanged and holds twice the tokens. Measured on
    Qwen3.8-27B, concurrency 71.67x -> 86.00x and KV capacity 293,546 ->
    352,256 tokens, with the mamba page byte-identical either way.
    """
    spec = _original_get_kv_cache_spec(self, vllm_config)
    if not getattr(self.impl, "qfa_decode_enabled", False):
        return spec
    if type(spec) is not FullAttentionSpec:
        raise AssertionError(f"QFA decode expects a plain FullAttentionSpec, got {type(spec).__name__}")
    if spec.block_size % _KERNEL_BLOCK:
        raise AssertionError(
            f"{self.layer_name}: block of {spec.block_size} tokens is not a multiple of the "
            f"{_KERNEL_BLOCK}-token kernel block quant_flash_attn addresses the cache with"
        )
    new_spec = dataclasses.replace(spec, dtype=torch.float8_e4m3fn, block_size=spec.block_size * 2)
    if new_spec.page_size_bytes != spec.page_size_bytes:
        # Only true if something other than the element size changed the page;
        # letting it through would put this layer on a page of its own and pad
        # every other page in the model up to whichever is largest.
        raise AssertionError(
            f"{self.layer_name}: MXFP8 page {new_spec.page_size_bytes} does not match the "
            f"model's {spec.page_size_bytes}; the halved element size and doubled block "
            "should have cancelled out"
        )
    logger.info(
        "QFA MXFP8 spec: %s dtype %s -> %s, block_size %d -> %d, page %d bytes (unchanged), %d tokens/page",
        self.layer_name,
        spec.dtype,
        new_spec.dtype,
        spec.block_size,
        new_spec.block_size,
        new_spec.page_size_bytes,
        new_spec.block_size,
    )
    return new_spec


Attention.get_kv_cache_spec = _qfa_get_kv_cache_spec
