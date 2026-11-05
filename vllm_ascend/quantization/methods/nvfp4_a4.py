# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NVFP4 A4 QDQ at real Ascend dense/expert GEMM boundaries.

Buffers are created by process_weights_after_loading, before warmup/capture.
No weight quantization, runner patch, forward-path CPU synchronization or per-step Python state.
"""

from dataclasses import replace

import torch
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.nvfp4_a4 import nvfp4_activation_qdq, routed_global_scales


class NVFP4A4LinearMethod(AscendUnquantizedLinearMethod):
    supports_weight_switch = False

    def __init__(self, scale: float):
        super().__init__()
        self.input_scale = scale

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        layer.register_buffer(
            "nvfp4_a4_input_scale",
            torch.tensor(self.input_scale, dtype=torch.float32, device=layer.weight.device),
            persistent=False,
        )
        logger.info_once("NVFP4 A4 QDQ dense path initialized (BF16 GEMM, not a native FP4 kernel)")

    def apply(self, layer, x, bias=None):
        return super().apply(layer, nvfp4_activation_qdq(x, layer.nvfp4_a4_input_scale), bias)


class NVFP4A4MoEMethod(AscendUnquantizedFusedMoEMethod):
    @property
    def supports_eplb(self) -> bool:
        return False

    def __init__(self, moe, gate_up_scales: list[float], down_scales: list[float]):
        super().__init__(moe=moe)
        self.gate_up_scales = gate_up_scales
        self.down_scales = down_scales
        if (
            moe.num_experts != len(gate_up_scales)
            or moe.num_local_experts != len(gate_up_scales)
            or len(down_scales) != len(gate_up_scales)
        ):
            raise ValueError("NVFP4 A4 requires identity single-rank logical expert placement")

    def process_weights_after_loading(self, layer):
        mapping = getattr(layer, "ascend_expert_map", None)
        if mapping is not None:
            # One-time load-stage check, before capture; never inspect device values in forward.
            expected = torch.arange(len(self.gate_up_scales), dtype=torch.int64, device="cpu")
            if not torch.equal(mapping.detach().to(device="cpu", dtype=torch.int64), expected):
                raise ValueError("NVFP4 A4 requires identity expert placement; calibration must not be misassigned")
        super().process_weights_after_loading(layer)
        device = layer.w13_weight.device
        layer.register_buffer(
            "nvfp4_a4_gate_up_scale",
            torch.tensor(self.gate_up_scales, dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "nvfp4_a4_down_scale", torch.tensor(self.down_scales, dtype=torch.float32, device=device), persistent=False
        )
        logger.info_once("NVFP4 A4 QDQ expert gate/up and down paths initialized; router weights unchanged")

    @staticmethod
    def _qdq(inputs, x, scales):
        if inputs.dynamic_scale is not None or inputs.dynamic_eplb or inputs.fusion or inputs.lora_context is not None:
            raise ValueError("NVFP4 A4 requires the unfused unquantized expert path")
        row_scales = routed_global_scales(scales, inputs.group_list, inputs.group_list_type, x.shape[0])
        return nvfp4_activation_qdq(x, row_scales)

    def apply_gmm1(self, inputs):
        x = self._qdq(inputs, inputs.hidden_states, inputs.layer.nvfp4_a4_gate_up_scale)
        return super().apply_gmm1(replace(inputs, hidden_states=x))

    def apply_act_quant(self, inputs, hidden_states):
        # QDQ must precede routing-weight multiplication: QDQ(x*w) != QDQ(x)*w.
        # The parent preserves the existing router-weight/GMM2 ordering.
        x = self._qdq(inputs, hidden_states, inputs.layer.nvfp4_a4_down_scale)
        return super().apply_act_quant(inputs, x)
