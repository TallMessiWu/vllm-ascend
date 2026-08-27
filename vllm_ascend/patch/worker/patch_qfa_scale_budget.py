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

import torch
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.attention.qfa_scale import QFA_KERNEL_BLOCK, qfa_scale_bytes_per_kernel_block

_original_get_kv_cache_config_from_groups = kv_cache_utils.get_kv_cache_config_from_groups

_QFA_CACHE_DTYPE = torch.float8_e4m3fn


def _qfa_scale_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Scale bytes every MXFP8 layer together needs per allocated block.

    The value planes are shared: a block id is unique across the whole pool,
    so each layer's tensor is a view of the same memory and the pool costs one
    page per block. The scale tables are not shared - every MXFP8 layer holds
    its own, sized to address every block, because any block may be handed to
    it. So the scales cost `layers * chunk * per_kernel_block` per block while
    the values cost one page, and the ratio depends on how the layers happened
    to be grouped rather than on the 1/32 the format implies. Measured on
    Qwen3.8-27B: 3.2% of the values without MTP, where 16 layers share a
    group, and 49% with it, where grouping falls apart into one layer each.
    """
    total = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if getattr(spec, "dtype", None) is not _QFA_CACHE_DTYPE:
            continue
        chunk = spec.block_size // QFA_KERNEL_BLOCK
        per_block = qfa_scale_bytes_per_kernel_block(spec.num_kv_heads, spec.head_size)
        total += len(group.layer_names) * chunk * per_block
    return total


def _qfa_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int | None:
    """Value bytes one block costs, or None if the layout is one we don't model.

    Mirrors the main path of vLLM's `_pool_bytes_per_block`. The packed and
    aggregated-spec layouts it also handles belong to MLA models, which have
    no MXFP8 layer to reserve for, so seeing one means our arithmetic does not
    apply and the caller should leave the budget alone rather than guess.
    """
    pages = {g.kv_cache_spec.page_size_bytes for g in kv_cache_groups}
    if len(pages) != 1:
        return None
    return pages.pop() * max(len(g.layer_names) for g in kv_cache_groups)


def _log_group_layout(kv_cache_groups: list[KVCacheGroupSpec]) -> None:
    """Print how the layers were bucketed, one line per distinct spec.

    vLLM groups KV cache layers by exact spec equality and then takes the
    smallest bucket as the group size, so a single layer that differs from the
    rest - an MTP drafter still on BF16 next to MXFP8 attention - collapses
    every group to one layer. Nothing downstream says that happened; what it
    changes is the pool arithmetic below, since value planes are shared across
    a group while scale side tables are per layer. Printing the layout makes
    the collapse visible at startup instead of only as a capacity number, and
    it prints for BF16 runs too, so the two can be compared directly.
    """
    by_spec: dict[str, list[int]] = {}
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        key = (
            f"{type(spec).__name__} block={spec.block_size} page={spec.page_size_bytes}B"
            f" dtype={getattr(spec, 'dtype', '?')}"
        )
        by_spec.setdefault(key, []).append(len(group.layer_names))
    group_size = max((len(g.layer_names) for g in kv_cache_groups), default=0)
    logger.info(
        "KV cache layout: %d groups, group_size %d (the pool costs one page per layer of it)",
        len(kv_cache_groups),
        group_size,
    )
    for key, sizes in by_spec.items():
        logger.info("KV cache layout:   %s -> %d layers in %d group(s)", key, sum(sizes), len(sizes))


def _qfa_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Hand vLLM only the memory left after the scale tables are paid for.

    vLLM budgets what it allocates, and it does not allocate the E8M0 scale
    side tables - the impl does, after this has already decided how many
    blocks fit. Left alone the two add up to more than the device has: on
    Qwen3.8-27B with MTP the tables came to 11.7 GiB on top of a 23.8 GiB
    cache and the run died allocating them, at the default 0.85 utilization.

    Shrinking the input rather than the result keeps every downstream number
    consistent - block count, capacity, the reported concurrency - instead of
    reporting a capacity the device cannot actually hold.
    """
    _log_group_layout(kv_cache_groups)
    scale_per_block = _qfa_scale_bytes_per_block(kv_cache_groups)
    if scale_per_block:
        pool_per_block = _qfa_pool_bytes_per_block(kv_cache_groups)
        if pool_per_block is None:
            logger.warning(
                "QFA MXFP8: KV cache groups have mixed page sizes, so the scale tables "
                "cannot be budgeted for; expect to need a lower gpu_memory_utilization."
            )
        else:
            reserved = available_memory * scale_per_block // (pool_per_block + scale_per_block)
            logger.info(
                "QFA MXFP8: reserving %.2f GiB of %.2f GiB for the scale side tables "
                "(%d bytes/block against %d for the values, %.1f%%)",
                reserved / 2**30,
                available_memory / 2**30,
                scale_per_block,
                pool_per_block,
                100 * scale_per_block / pool_per_block,
            )
            available_memory -= reserved
    return _original_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)


kv_cache_utils.get_kv_cache_config_from_groups = _qfa_get_kv_cache_config_from_groups
