# SPDX-License-Identifier: Apache-2.0
"""Normal Qwen3 MoE serving with an optional load-time attention fallback."""

from vllm.logger import logger
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from vllm_ascend.observability.moe_attention_fallback import load_attention_fallback
from vllm_ascend.observability.moe_routing import MIDDLE_BANDS, attention_projection_names

ATTENTION_WEIGHTS = attention_projection_names(MIDDLE_BANDS["layers_12_15"])


class Qwen3MoeAttentionFallbackForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.attention_reference = getattr(config, "l12_15_attention_source", None)

    def load_weights(self, weights):
        if self.attention_reference is None:
            return super().load_weights(weights)
        loaded, seen, changed = load_attention_fallback(
            super().load_weights, weights, self.attention_reference, ATTENTION_WEIGHTS
        )
        logger.info("MOE-L12-15-GPQA: loaded %d reference projections (%d changed)", len(seen), len(changed))
        return loaded
