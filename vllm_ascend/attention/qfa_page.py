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
"""The model-level half of the QFA MXFP8 page decision.

Two places decide how big an attention page is, at opposite ends of startup,
and they have to agree. Platform setup sizes the attention block before any
layer exists, so that the mamba page can be aligned to it; the impl decides
per layer whether that layer's KV cache is actually MXFP8. Both read this
module for the part of the decision that is a property of the model rather
than of a layer.
"""

import torch
from vllm.config import VllmConfig

from vllm_ascend import envs as envs_ascend
from vllm_ascend.utils import is_950

# One byte per element, against two for the BF16 cache it replaces.
QFA_CACHE_DTYPE = torch.float8_e4m3fn

# Head sizes quant_flash_attn supports with no D-axis padding (72 would need
# padding to 128). A model outside them has no eligible layer whatsoever, so
# the page stays BF16 and none of this applies.
QFA_HEAD_SIZES = (64, 128, 256)


def qfa_fp8_attention_page(vllm_config: VllmConfig) -> bool:
    """Whether eligible full-attention layers store one byte per element.

    Platform setup calls this before any layer exists, to size the attention
    block against the FP8 page instead of a BF16 one. That is not a tuning
    choice. The mamba page is padded to whatever attention page platform setup
    computed, and `unify_hybrid_kv_cache_specs` pads a smaller page up rather
    than splitting a block: size the block for BF16 while the layers store FP8
    and the halved page is padded straight back, saving nothing. Measured on
    Qwen3.8-27B before this: the spec did report page 6291456 -> 3145728, and
    concurrency stayed at 71.67x regardless.

    The impl's per-layer `qfa_decode_enabled` adds what only a built layer
    knows - sliding window, sinks, attention type, and which model it belongs
    to. A full-attention layer that fails those keeps BF16 and would then hold
    twice the bytes in a block sized for FP8, making its page the largest in
    the model and dragging every other page up to it; the kv_cache_spec patch
    halves that layer's block size so both kinds land on the same page.
    """
    return (
        envs_ascend.VLLM_ASCEND_ENABLE_QFA_PREFILL
        and envs_ascend.VLLM_ASCEND_ENABLE_QFA_DECODE
        and is_950()
        and vllm_config.model_config.get_head_size() in QFA_HEAD_SIZES
    )
