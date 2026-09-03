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

"""Dump QFA's inputs and outputs so its numerics can be checked off-device.

Two dump points, because "how accurate is QFA" splits into two questions that
need different tensors:

  cache-write  the bf16 K/V this step is about to quantize, next to what the
               quantization produced. Once K/V land in the cache only FP8 is
               left, so the ground truth has to be captured here or not at all.
               This is what makes the *quantization* loss measurable, including
               how much the checkpoint's static per-channel V scale gives up
               against a per-sequence optimum.
  qfa-call     everything the operator itself is handed, plus what it returned.
               Replaying this against a float reference isolates the *compute*
               loss of doing attention in FP8, with quantization held fixed.

Off unless VLLM_ASCEND_QFA_DUMP_DIR is set. Only the eager path is dumpable:
a D2H copy inside an aclgraph capture fails with EE1016, and on replay no Python
executes -- so run with GRAPH=0. The operator is the same one either way, so a
number measured in eager carries over to the graph.

Size is kept down by slicing the paged cache to the blocks this step actually
reads, derived from seqused_kv. A single 1594-token prefill at block_size 512
touches four blocks, a couple of MB, rather than the cache's full ~422MB.

FP8 tensors are moved as byte views on purpose: transpose/index_put_ on float8
either errors or falls back to AICPU on NPU, so anything one byte wide travels
as uint8 and is restored on the analysis side by load_dump().
"""

import os
from typing import Any

import torch

from vllm_ascend import envs

_call_counts: dict[str, int] = {}
_warned = False


def dump_enabled() -> bool:
    """True when a dump directory is configured."""
    return bool(envs.VLLM_ASCEND_QFA_DUMP_DIR)


def _should_dump(layer_name: str) -> bool:
    if not dump_enabled():
        return False
    return _call_counts.get(layer_name, 0) < envs.VLLM_ASCEND_QFA_DUMP_CALLS


def _is_byte_float(t: torch.Tensor) -> bool:
    """FP8 family: one byte wide and not an integer type."""
    return "float8" in str(t.dtype)


def _host(value: Any) -> Any:
    """Move one value to host, routing FP8 through a uint8 byte view."""
    if not isinstance(value, torch.Tensor):
        return value
    if _is_byte_float(value):
        # Byte view is mandatory here, not a convenience -- see module docstring.
        return {"__fp8_dtype__": str(value.dtype), "bytes": value.view(torch.uint8).cpu()}
    return value.detach().cpu()


def _index_blocks(cache: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Gather cache blocks by index, routing FP8 through a uint8 view.

    aclnnIndex has no FP8 kernel and rejects the tensor outright (EZ1001), and
    this runs before anything reaches the host, so the byte view has to happen
    on device. float8 is one byte wide, so the view addresses the same memory
    with the same shape and is reinterpreted after the gather.
    """
    if _is_byte_float(cache):
        return cache.view(torch.uint8)[ids].view(cache.dtype)
    return cache[ids]


def _host_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _host_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_host_tree(v) for v in obj)
    return _host(obj)


def _blocks_touched(block_table: torch.Tensor, seqused_kv: torch.Tensor, block_size: int) -> torch.Tensor:
    """Block ids this step reads, from each sequence's KV length.

    Padding columns in block_table are excluded by taking only ceil(len/block)
    entries per row, so unused (often zero) slots do not drag in stray blocks.
    """
    lengths = seqused_kv.to(torch.int64).tolist()
    ids: list[int] = []
    for row, kv_len in enumerate(lengths):
        if row >= block_table.shape[0]:
            break
        needed = (int(kv_len) + block_size - 1) // block_size
        if needed > 0:
            ids.extend(block_table[row, :needed].to(torch.int64).tolist())
    if not ids:
        return torch.zeros(0, dtype=torch.int64)
    return torch.unique(torch.tensor(ids, dtype=torch.int64))


def _write(layer_name: str, phase: str, payload: dict[str, Any]) -> None:
    directory = envs.VLLM_ASCEND_QFA_DUMP_DIR
    os.makedirs(directory, exist_ok=True)
    call = _call_counts.get(layer_name, 0)
    safe_layer = layer_name.replace("/", "_").replace(".", "_")
    path = os.path.join(directory, f"{safe_layer}__call{call}__{phase}.pt")
    torch.save(payload, path)


def dump_cache_write(
    layer_name: str,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mxfp8: torch.Tensor,
    key_scale: torch.Tensor,
    value_mxfp8: torch.Tensor,
    v_cache_scale: torch.Tensor,
    v_cache_scale_float_reciprocal: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_actual_tokens: int,
) -> None:
    """Ground truth for the quantization step: bf16 in, FP8 + scales out."""
    if not _should_dump(layer_name):
        return
    _write(
        layer_name,
        "cache_write",
        _host_tree(
            {
                "layer_name": layer_name,
                "num_actual_tokens": num_actual_tokens,
                "query_bf16": query[:num_actual_tokens],
                "key_bf16": key[:num_actual_tokens],
                "value_bf16": value[:num_actual_tokens],
                "key_mxfp8": key_mxfp8,
                "key_scale": key_scale,
                "value_mxfp8": value_mxfp8,
                # Static per-channel E8M0 exponent from the checkpoint, and the
                # reciprocal actually handed to npu_quantize.
                "v_cache_scale": v_cache_scale,
                "v_cache_scale_float_reciprocal": v_cache_scale_float_reciprocal,
                "slot_mapping": slot_mapping[:num_actual_tokens],
            }
        ),
    )


def _dump_attn_mask_once(attn_mask: torch.Tensor | None) -> str | None:
    """Store the shared mask in its own file; every layer points at the same one.

    get_splitfuse_attn_mask() hands out one cached 2048x2048 int8 tensor for the
    whole run, so writing it per call would add 4MB a time for no new content.
    """
    if attn_mask is None:
        return None
    directory = envs.VLLM_ASCEND_QFA_DUMP_DIR
    os.makedirs(directory, exist_ok=True)
    name = "attn_mask.pt"
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        torch.save(_host(attn_mask), path)
    return name


def dump_qfa_call(
    layer_name: str,
    *,
    query_bf16: torch.Tensor,
    q_fp8: torch.Tensor,
    q_descale: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    block_table: torch.Tensor,
    metadata: torch.Tensor,
    op_kwargs: dict[str, Any],
    attn_mask: torch.Tensor | None,
    attn_output: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Everything the operator saw, with the cache sliced to the read blocks."""
    if not _should_dump(layer_name):
        return
    mask_file = _dump_attn_mask_once(attn_mask)

    key_cache, value_cache, key_scale_cache, value_scale_cache = kv_cache
    block_size = key_cache.shape[1]
    seqused_kv = op_kwargs["seqused_kv"]
    block_ids = _blocks_touched(block_table, seqused_kv, block_size)

    # Renumber the block table against the sliced cache so the dump replays
    # standalone: block_ids[i] in the full cache becomes i in the slice.
    remap = {int(b): i for i, b in enumerate(block_ids.tolist())}
    local_block_table = block_table.detach().cpu().clone()
    for row in range(local_block_table.shape[0]):
        for col in range(local_block_table.shape[1]):
            local_block_table[row, col] = remap.get(int(local_block_table[row, col]), 0)

    device_ids = block_ids.to(key_cache.device)
    _write(
        layer_name,
        "qfa_call",
        _host_tree(
            {
                "layer_name": layer_name,
                "query_bf16": query_bf16,
                "q_fp8": q_fp8,
                "q_descale": q_descale,
                # Cache slices, in the same PA_BBND order the operator reads.
                "key_cache": _index_blocks(key_cache, device_ids),
                "value_cache": _index_blocks(value_cache, device_ids),
                "key_scale_cache": _index_blocks(key_scale_cache, device_ids),
                "value_scale_cache": _index_blocks(value_scale_cache, device_ids),
                "block_ids": block_ids,
                "block_table_full": block_table,
                "block_table_local": local_block_table,
                "block_size": block_size,
                "metadata": metadata,
                "op_kwargs": op_kwargs,
                # Sidecar file, so a replay can rebuild the full argument list.
                "attn_mask_file": mask_file,
                "attn_output": attn_output,
                "softmax_scale": softmax_scale,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_size": head_size,
            }
        ),
    )
    # Both phases of one forward share a call index; count once, at the end.
    _call_counts[layer_name] = _call_counts.get(layer_name, 0) + 1


def load_dump(path: str) -> dict[str, Any]:
    """Read a dump back, restoring the FP8 dtypes that travelled as uint8."""

    def restore(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__fp8_dtype__" in obj:
                name = obj["__fp8_dtype__"].removeprefix("torch.")
                dtype = getattr(torch, name, None)
                if dtype is None:
                    return obj["bytes"]
                return obj["bytes"].view(dtype)
            return {k: restore(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(restore(v) for v in obj)
        return obj

    return restore(torch.load(path, map_location="cpu", weights_only=False))
