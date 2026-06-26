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
"""Group-aligned tensor-parallel linear layers for MX-quantized ViT blocks.

Background
----------
MX (microscaling) quantization groups every ``group_size`` (e.g. 32) elements
along the input dimension and stores one shared scale per group. Splitting such
a weight across tensor-parallel ranks is only numerically valid if every
partition boundary lands on a group boundary -- otherwise a single group is cut
across two ranks and its scale can no longer be applied correctly.

Plain :class:`ColumnParallelLinear` / :class:`RowParallelLinear` split a
dimension evenly (``size // tp_size``). For the Qwen3-VL vision MLP the
intermediate size is ``4304`` which is not a multiple of ``group_size * tp_size``
(``64`` for tp=2), so the even split ``2152`` falls inside group #67 and the
weight_scale load fails with a ``narrow ... exceeds dimension size`` error.

These subclasses instead split the sharded dimension on **group boundaries**:
groups are distributed as evenly as possible (earlier ranks get the +1), and the
last rank absorbs the trailing partial group. This yields an uneven but
group-aligned partition (e.g. tp=2 -> ``[2176, 2128]`` elements / ``[68, 67]``
groups for 4304) that is correct for the row all-reduce.
"""

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs


def group_aligned_partition(total: int, tp_size: int, group_size: int) -> tuple[list[int], list[int]]:
    """Split ``total`` elements (along the sharded dim) across ``tp_size`` ranks
    on ``group_size`` boundaries so no MX group straddles a partition boundary.

    Returns ``(elem_sizes, group_sizes)`` -- per-rank element counts and group
    counts. Groups are distributed as evenly as possible (earlier ranks get the
    +1); the last rank absorbs the trailing partial group, if any.

    The invariant ``ceil(elem_sizes[r] / group_size) == group_sizes[r]`` holds
    for every rank, which keeps the loaded weight (split by elements) consistent
    with the loaded weight_scale (split by groups). For dimensions that are
    already a multiple of ``group_size * tp_size`` the result equals the plain
    even split; for ``tp_size == 1`` it is the full tensor.
    """

    def _cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    if tp_size <= 1:
        return [total], [_cdiv(total, group_size)]

    num_groups = _cdiv(total, group_size)
    base, rem = divmod(num_groups, tp_size)
    group_sizes = [base + (1 if r < rem else 0) for r in range(tp_size)]

    # Each rank owns a contiguous span of groups; its element count is that span
    # clamped to ``total`` so the trailing partial group stays intact and no rank
    # goes negative even in the degenerate ``num_groups < tp_size`` case.
    elem_sizes: list[int] = []
    groups_done = 0
    for r in range(tp_size):
        start = groups_done * group_size
        groups_done += group_sizes[r]
        end = min(groups_done * group_size, total)
        elem_sizes.append(max(0, end - start))
    return elem_sizes, group_sizes


def _resolve_group_size(quant_config, override: int | None = None, default: int = 32) -> int:
    if override is not None:
        return int(override)
    quant_description = getattr(quant_config, "quant_description", None)
    if isinstance(quant_description, dict):
        try:
            return int(quant_description.get("group_size", default))
        except (TypeError, ValueError):
            return default
    return default


def _v1_or_v2_loader(layer):
    return (
        layer.weight_loader_v2
        if layer.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
        else layer.weight_loader
    )


class GroupAlignedColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear that shards the output dim on MX group boundaries.

    For a column-parallel layer the MX groups run along the (unsharded) input
    dim, so the output weight, weight_scale and bias all share the same
    per-element output split -- no weight/scale distinction is needed here.
    """

    def __init__(self, input_size: int, output_size: int, *, group_size: int | None = None, **kwargs):
        # Let the parent build the (even) params first; output_size is divisible
        # by tp_size so this never raises.
        super().__init__(input_size, output_size, **kwargs)
        # Unquantized layers have no MX groups to align; keep the parent's split.
        if kwargs.get("quant_config") is None:
            self._ga_elem_sizes = None
            return
        g = _resolve_group_size(kwargs.get("quant_config"), group_size)
        self._ga_elem_sizes, self._ga_group_sizes = group_aligned_partition(output_size, self.tp_size, g)

        # Rebuild only when the group-aligned size differs from the even split.
        if self.tp_size > 1 and self._ga_elem_sizes[self.tp_rank] != self.output_size_per_partition:
            self.output_size_per_partition = self._ga_elem_sizes[self.tp_rank]
            self.output_partition_sizes = [self.output_size_per_partition]
            self.quant_method.create_weights(
                layer=self,
                input_size_per_partition=self.input_size,
                output_partition_sizes=self.output_partition_sizes,
                input_size=self.input_size,
                output_size=self.output_size,
                params_dtype=self.params_dtype,
                weight_loader=_v1_or_v2_loader(self),
            )
            if self.bias is not None:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=self.params_dtype))
                set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
            self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        sizes = getattr(self, "_ga_elem_sizes", None)
        # Only group-aligned params split along the output dim (weight rows,
        # weight_scale rows, bias). Anything else falls back to the parent.
        if output_dim is None or sizes is None or loaded_weight.shape[output_dim] != self.output_size:
            return super().weight_loader(param, loaded_weight)
        start = sum(sizes[: self.tp_rank])
        size = sizes[self.tp_rank]
        loaded_weight = loaded_weight.narrow(output_dim, start, size)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.data.shape == loaded_weight.shape, (
            f"{self.prefix}: param {tuple(param.data.shape)} != loaded {tuple(loaded_weight.shape)}"
        )
        param.data.copy_(loaded_weight)


class GroupAlignedRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that shards the input dim on MX group boundaries.

    Here the MX groups run along the sharded input dim, so the weight is split
    by *element* counts while the weight_scale is split by *group* counts. The
    two are told apart by the full size of the loaded tensor along ``input_dim``.
    """

    def __init__(self, input_size: int, output_size: int, *, group_size: int | None = None, **kwargs):
        super().__init__(input_size, output_size, **kwargs)
        # Unquantized layers have no MX groups to align; keep the parent's split.
        if kwargs.get("quant_config") is None:
            self._ga_elem_sizes = None
            return
        g = _resolve_group_size(kwargs.get("quant_config"), group_size)
        self._ga_elem_sizes, self._ga_group_sizes = group_aligned_partition(input_size, self.tp_size, g)
        self._ga_scale_groups_full = sum(self._ga_group_sizes)  # == ceil(input_size / g)

        if self.tp_size > 1 and self._ga_elem_sizes[self.tp_rank] != self.input_size_per_partition:
            self.input_size_per_partition = self._ga_elem_sizes[self.tp_rank]
            self.quant_method.create_weights(
                layer=self,
                input_size_per_partition=self.input_size_per_partition,
                output_partition_sizes=self.output_partition_sizes,
                input_size=self.input_size,
                output_size=self.output_size,
                params_dtype=self.params_dtype,
                weight_loader=_v1_or_v2_loader(self),
            )
            self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is None or getattr(self, "_ga_elem_sizes", None) is None:
            # bias (full, replicated), unquantized layers, and any
            # non-input-sharded param fall back to the parent loader.
            return super().weight_loader(param, loaded_weight)
        full = loaded_weight.shape[input_dim]
        if full == self.input_size:
            sizes = self._ga_elem_sizes  # weight: split by elements
        elif full == getattr(self, "_ga_scale_groups_full", None):
            sizes = self._ga_group_sizes  # weight_scale: split by groups
        else:
            return super().weight_loader(param, loaded_weight)
        start = sum(sizes[: self.tp_rank])
        size = sizes[self.tp_rank]
        loaded_weight = loaded_weight.narrow(input_dim, start, size)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.data.shape == loaded_weight.shape, (
            f"{self.prefix}: param {tuple(param.data.shape)} != loaded {tuple(loaded_weight.shape)}"
        )
        param.data.copy_(loaded_weight)
