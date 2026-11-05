# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in NVFP4 activation simulation on a dequantized BF16 Qwen3 MoE carrier."""

import json
from pathlib import Path

import regex as re
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.quantization.nvfp4_a4 import ActivationScales, read_json
from vllm_ascend.quantization.utils import is_fused_moe_layer


@register_quantization_config("nvfp4_a4_qdq")
class AscendNVFP4A4Config(QuantizationConfig):
    """Only A changes; parameters remain ordinary BF16, including expert loaders."""

    def __init__(self, source: str | None = None):
        super().__init__()
        if not source:
            raise ValueError("nvfp4_a4_qdq requires source via hf_overrides.quantization_config_dict_json")
        self.scales = ActivationScales(source)
        self._runtime_checked = False  # Per-model configuration, finalized before warmup/capture.

    def get_name(self):
        return "nvfp4_a4_qdq"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        if set(config) != {"source"} or not isinstance(config["source"], str):
            raise ValueError("NVFP4 A4 config must contain only a source checkpoint path")
        return cls(config["source"])

    @classmethod
    def from_config_dict_json(cls, value):
        return cls.from_config(json.loads(value) if isinstance(value, str) else value)

    def _check_runtime(self):
        if self._runtime_checked:
            return
        from vllm_ascend.ascend_config import get_ascend_config

        config = get_current_vllm_config()
        parallel = config.parallel_config
        if any(
            getattr(parallel, key, 1) != 1
            for key in ("tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size")
        ):
            raise ValueError("NVFP4 A4 QDQ currently supports single-rank TP1/DP1/PP1 only")
        if config.model_config.hf_config.model_type != "qwen3_moe" or config.model_config.dtype != torch.bfloat16:
            raise ValueError("NVFP4 A4 QDQ requires a BF16 Qwen3 MoE carrier")
        if config.cache_config.cache_dtype != "bfloat16":
            raise ValueError("NVFP4 A4 QDQ reference requires explicit BF16 KV")
        ascend = get_ascend_config()
        if (
            config.lora_config is not None
            or config.speculative_config is not None
            or ascend.enable_fused_mc2
            or ascend.eplb_config.dynamic_eplb
            or getattr(parallel, "enable_eplb", False)
        ):
            raise ValueError("NVFP4 A4 QDQ does not support LoRA, speculation, fused MC2 or EPLB")
        carrier = Path(config.model_config.model)
        if (carrier / ".nvfp4_to_bf16.incomplete").exists():
            raise ValueError("BF16 carrier conversion is incomplete")
        saved_config = read_json(carrier / ".nvfp4_source_metadata/source-config.json")
        if saved_config != self.scales.config:
            raise ValueError("Activation calibration config does not match the BF16 carrier provenance")
        if read_json(carrier / "config.json").get("quantization_config"):
            raise ValueError("Expected decoded BF16 weights, not a packed NVFP4 checkpoint")
        self._runtime_checked = True

    def get_quant_method(self, layer, prefix, tid2eid=None):
        # Imported only when the selected quantization config builds a model.
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        from vllm_ascend.quantization.methods.nvfp4_a4 import NVFP4A4LinearMethod, NVFP4A4MoEMethod

        self._check_runtime()
        if is_fused_moe_layer(layer):
            if (
                not re.fullmatch(r"model\.layers\.\d+\.mlp\.experts(?:\.routed_experts)?", prefix)
                or tid2eid is not None
            ):
                raise ValueError("Unsupported NVFP4 A4 expert prefix/mapping")
            return NVFP4A4MoEMethod(
                layer.moe_config, self.scales.experts(prefix, "gate_proj"), self.scales.experts(prefix, "down_proj")
            )
        if isinstance(layer, LinearBase):
            if re.fullmatch(r"model\.layers\.\d+\.self_attn\.(qkv_proj|o_proj)", prefix):
                return NVFP4A4LinearMethod(self.scales.linear(prefix))
            if re.fullmatch(r"model\.layers\.\d+\.mlp\.gate", prefix) or prefix == "lm_head":
                return AscendUnquantizedLinearMethod()
            raise ValueError(f"Unsupported NVFP4 A4 Linear target: {prefix}")
        return None  # Embedding, norm and LM head retain their existing unquantized methods.
