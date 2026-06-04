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
"""轻量级 MoE 调试探针（仅 debug 分支使用）。

用于定位 "TP1 正常、TP2 精度劣化 + MoeInitRouting 算子耗时劣化" 问题。

开关（环境变量）:
    VLLM_ASCEND_MOE_DEBUG=1        打开全部调试日志与计时（默认关闭，零开销）
    VLLM_ASCEND_MOE_DEBUG_STEPS=8  每个埋点最多打印多少次，避免 decode 阶段刷屏

设计目标：默认完全关闭（不引入任何同步/打印开销）；打开后只在前若干步打印，
方便用 TP1 与 TP2 两次运行的日志做逐字段对比。
"""
from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_ASCEND_MOE_DEBUG", "0") == "1"
_MAX_LOGS = int(os.environ.get("VLLM_ASCEND_MOE_DEBUG_STEPS", "8"))

# 每个埋点 tag 的已打印次数，超过 _MAX_LOGS 后静默。
_counters: dict[str, int] = {}
# log_once 用：记录已打印过的 tag。
_once_seen: set[str] = set()


def moe_debug_enabled() -> bool:
    return _ENABLED


def _should_log(tag: str) -> bool:
    if not _ENABLED:
        return False
    n = _counters.get(tag, 0)
    if n >= _MAX_LOGS:
        return False
    _counters[tag] = n + 1
    return True


def _fmt(fields: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())


def log_kv(tag: str, **fields) -> None:
    """打印一行 key=value 调试信息（受 _MAX_LOGS 次数限制）。"""
    if _should_log(tag):
        logger.info("[MOE-DEBUG][%s] %s", tag, _fmt(fields))


def log_once(tag: str, **fields) -> None:
    """同一 tag 仅打印一次（适合每步都会命中的配置类日志）。"""
    if not _ENABLED or tag in _once_seen:
        return
    _once_seen.add(tag)
    logger.info("[MOE-DEBUG][%s] %s", tag, _fmt(fields))


def _stats(t) -> str:
    """单个张量的形状 + 数值概要（会触发 device→host 同步，仅 debug 用）。

    - 浮点 / float8：min/max/mean + nan/inf 计数（先转 float 规避 float8 reduce 限制）。
    - 整型：min/max/sum + neg（负值数量；用于看 expanded_row_idx 的 droppad 占比）。
    """
    if t is None:
        return "None"
    if not torch.is_tensor(t):
        return repr(t)
    out = f"shape={tuple(t.shape)} dtype={t.dtype}"
    if t.numel() == 0:
        return out + " (empty)"
    try:
        if t.is_floating_point():
            tf = t.detach().float()
            n_nan = int(torch.isnan(tf).sum())
            n_inf = int(torch.isinf(tf).sum())
            finite = tf[torch.isfinite(tf)]
            if finite.numel() > 0:
                out += f" min={finite.min().item():.4g} max={finite.max().item():.4g} mean={finite.mean().item():.4g}"
            out += f" nan={n_nan} inf={n_inf}"
        else:
            td = t.detach()
            out += f" min={int(td.min())} max={int(td.max())} sum={int(td.sum())} neg={int((td < 0).sum())}"
    except Exception as e:  # noqa: BLE001
        out += f" (stats_err={type(e).__name__})"
    return out


def log_tensor_stats(tag: str, **tensors) -> None:
    """打印一组张量的形状/数值概要（受 _MAX_LOGS 次数限制）。"""
    if not _should_log(tag):
        return
    parts = [f"{k}=[{_stats(v)}]" for k, v in tensors.items()]
    logger.info("[MOE-DEBUG][%s] %s", tag, " ".join(parts))


class MoeInitRoutingProbe:
    """上下文管理器：进入时记录入参，退出时用 NPU event 计 elapsed_ms，并一起打印。

    用法::

        with MoeInitRoutingProbe("AllGather.init_routing", full_load=..., ...):
            out = DeviceOperator.npu_moe_init_routing(...)

    关键诊断字段建议传入 ``full_load``（expert_start==0 且 expert_end==expert_num）
    与 ``active_expert_range``：当 full_load=False 时，custom 算子退化为 ep_=1 的
    逐 token 路径（perCoreTokens=1），这正是耗时劣化的根因。
    """

    def __init__(self, tag: str, **fields):
        self.tag = tag
        self.fields = fields
        self.active = _should_log(tag)
        self._start: torch.npu.Event | None = None
        self._end: torch.npu.Event | None = None

    def __enter__(self) -> "MoeInitRoutingProbe":
        if self.active:
            self._start = torch.npu.Event(enable_timing=True)
            self._end = torch.npu.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.active:
            return
        assert self._start is not None and self._end is not None
        self._end.record()
        # 仅同步这两个 event，等待算子真正完成以获得准确耗时。
        self._end.synchronize()
        elapsed_ms = self._start.elapsed_time(self._end)
        logger.info("[MOE-DEBUG][%s] %s elapsed_ms=%.3f", self.tag, _fmt(self.fields), elapsed_ms)
