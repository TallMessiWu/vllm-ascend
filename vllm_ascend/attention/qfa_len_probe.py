"""Debug-only probe: the length tensors every QFA call actually receives.

Off unless VLLM_ASCEND_QFA_LEN_DUMP names a directory. It lives on the
debug-qfa-kv-lens branch only; delete it together with that branch once the
operator team has its answer.

The record is taken right after the model forward returns, not inside
attention. Under FULL_DECODE_ONLY the attention forward is an ACL graph whose
Python runs once, at capture, on dummy data -- a print there fires once and
never again, and a host read inside the captured region is illegal anyway.
What each replay does read are the persistent device buffers behind
AscendMetadata.query_start_loc_gpu / seq_lens_gpu, and those still hold this
forward's values when the forward returns. Reading them there and applying
the sanitize AscendC8MXFPAttentionBackendImpl._qfa_step_lengths applies
reproduces the operator's inputs exactly, eager or replayed.

cu_seqlens_kv is absent on purpose: this path passes None. The KV side is
addressed by block_table + seqused_kv.

The two call sites are the real forwards only (execute_model and the drafter's
_propose), so profile runs, capture warmups and DP dummy batches never land in
the dump.

Cost when on: one device-to-host read per forward of a few hundred int32 at
most. That delays each step slightly, which under the async scheduler can
shift batch composition a little; it does not change what any single call
receives.
"""

import json
import os
import socket
import time
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

# One record per operator call. A long benchmark at a large batch writes a few
# KB a line, so stop well before the disk notices.
_MAX_RECORDS = 100_000


class _ProbeState:
    def __init__(self) -> None:
        self.resolved = False
        self.out_dir: str | None = None
        self.file: Any = None
        self.records = 0
        self.forwards = 0


_state = _ProbeState()


def _out_dir() -> str | None:
    if not _state.resolved:
        _state.out_dir = envs.VLLM_ASCEND_QFA_LEN_DUMP or None
        _state.resolved = True
    return _state.out_dir


def _writer(out_dir: str, dp_rank: int):
    if _state.file is None:
        os.makedirs(out_dir, exist_ok=True)
        name = f"qfa_len_{socket.gethostname()}_dp{dp_rank}_pid{os.getpid()}.jsonl"
        path = os.path.join(out_dir, name)
        # Line-buffered so a server killed mid-benchmark still leaves whole lines.
        _state.file = open(path, "a", buffering=1)  # noqa: SIM115
        logger.warning("QFA length probe on, writing %s", path)
    return _state.file


def _length_sources(per_step_metadata: list):
    """Yield (step, metadata) once per distinct length source.

    Each step is a per-layer dict, or a list of them under micro-batching.
    Every full-attention layer of a step shares one metadata object, so the
    identity check collapses them to one record per operator call pattern.
    """
    seen: set[int] = set()
    for step, per_layer in enumerate(per_step_metadata):
        groups = per_layer if isinstance(per_layer, list) else [per_layer]
        for group in groups:
            if not isinstance(group, dict):
                continue
            for metadata in group.values():
                if getattr(metadata, "seq_lens_gpu", None) is None:
                    continue
                if getattr(metadata, "query_start_loc_gpu", None) is None:
                    continue
                if id(metadata) in seen:
                    continue
                seen.add(id(metadata))
                yield step, metadata


def _record(step: int, metadata: Any, *, is_draft_model: bool, graph_mode: Any, dp_rank: int) -> dict:
    num_tokens = int(metadata.num_actual_tokens)
    # The one host read: the persistent buffers the captured graph consumed.
    query_start_loc = metadata.query_start_loc_gpu.to("cpu")
    seq_lens = metadata.seq_lens_gpu.to("cpu")
    # _qfa_step_lengths, verbatim: unused tail slots become zero-length
    # requests whose KV length is 1.
    cu_seqlens_q = query_start_loc.clamp(min=0, max=num_tokens).cummax(dim=0).values
    seqused_kv = seq_lens.clamp(min=1)
    attn_state = getattr(metadata, "attn_state", None)
    block_tables = getattr(metadata, "block_tables", None)
    return {
        "forward": _state.forwards,
        "time": round(time.time(), 3),
        "dp_rank": dp_rank,
        "draft": is_draft_model,
        "draft_step": step if is_draft_model else None,
        "graph_mode": getattr(graph_mode, "name", str(graph_mode)),
        "attn_state": getattr(attn_state, "name", str(attn_state)),
        "num_actual_tokens": num_tokens,
        "max_query_len": metadata.max_query_len,
        "num_decodes": getattr(metadata, "num_decodes", None),
        "num_prefills": getattr(metadata, "num_prefills", None),
        # Slots the runner zero-filled are batch padding, not requests; the
        # sanitize turns them into KV length 1, so count them before it does.
        "num_padding_kv_slots": int((seq_lens <= 0).sum()),
        "block_table_rows": None if block_tables is None else int(block_tables.shape[0]),
        "cu_seqlens_q": cu_seqlens_q.tolist(),
        "seqused_kv": seqused_kv.tolist(),
    }


def record_qfa_lengths(per_step_metadata: list, *, is_draft_model: bool, graph_mode: Any) -> None:
    """Append this forward's QFA length inputs to the dump; a no-op unless enabled."""
    out_dir = _out_dir()
    if out_dir is None or _state.records >= _MAX_RECORDS:
        return
    from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank

    # Every TP rank of a DP group sees the same lengths; DP ranks each have
    # their own batch, so keep one TP rank per DP rank.
    if get_tensor_model_parallel_rank() != 0:
        return
    dp_rank = get_dp_group().rank_in_group
    _state.forwards += 1
    writer = _writer(out_dir, dp_rank)
    for step, metadata in _length_sources(per_step_metadata):
        record = _record(step, metadata, is_draft_model=is_draft_model, graph_mode=graph_mode, dp_rank=dp_rank)
        writer.write(json.dumps(record) + "\n")
        _state.records += 1
    if _state.records >= _MAX_RECORDS:
        logger.warning("QFA length probe reached %d records, stopping", _MAX_RECORDS)
