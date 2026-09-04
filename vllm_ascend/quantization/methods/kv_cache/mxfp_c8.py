import functools

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger

from ..base import AscendAttentionScheme


def _quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor, *, head_size_v: int):
    """Load the static per-channel V-cache scale into this rank's slice.

    The scale is one E8M0 byte per (kv_head, channel), so the split is counted
    in KV heads rather than by dividing the tensor by tp_size. Two cases make
    an even split wrong: a checkpoint may store a single set of channel scales
    shared by every KV head, and when total_kv_heads < tp_size vLLM replicates
    a KV head across ranks instead of sharding it, leaving every rank with the
    full head.
    """
    # ModelSlim ships the V-cache scale as [hidden_size, 1]; the parameter is
    # 1-D, and the size assert below would reject the trailing axis.
    loaded_weight = loaded_weight.reshape(-1)
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_in_ckpt = loaded_weight.numel() // head_size_v
        heads_in_param = param.numel() // head_size_v
        if heads_in_ckpt == 1 and heads_in_param > 1:
            loaded_weight = loaded_weight.repeat(heads_in_param)
        elif heads_in_ckpt > heads_in_param:
            replicas = max(1, tp_size // heads_in_ckpt)
            first_head = (tp_rank // replicas) * heads_in_param
            loaded_weight = loaded_weight.narrow(0, first_head * head_size_v, heads_in_param * head_size_v)
        assert param.size() == loaded_weight.size(), (
            "[vllm-ascend/MXFP8_PER_CHANNEL] Attempted to load weight "
            f"({loaded_weight.size()}) into parameter ({param.size()}) "
            f"when TP size is {tp_size}, TP rank is {tp_rank} and head_size_v is {head_size_v}."
        )

        param.data.copy_(loaded_weight)


class AscendC8MXFPKVCacheAttentionMethod(AscendAttentionScheme):
    """MXFP8 KV cache storage for dense-attention models.

    K/V are cached as FP8 E4M3 and their per-32-element E8M0 scales are stored
    in extra cache tensors. The C8-MXFP backend owns the matching 512-token
    kernel block size, keeping hybrid cache scheduling and cache views aligned.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.float8_e4m3fn
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import (
                AscendC8MXFPAttentionBackend,
                AscendC8MXFPAttentionBackendImpl,
            )

            layer.attn_backend = AscendC8MXFPAttentionBackend
            layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
            layer.impl.save_v_scale_flag = False
            # Changing __class__ does not invoke the new class's __init__.
            # Hamming sparse/KVComp is unavailable on this baseline.
            layer.impl.enable_hamming_sparse = False

        # Load v_cache static quantization scale
        hidden_size = layer.num_kv_heads * layer.head_size_v
        # E8M0 stores the exponent with a bias of 127, so 127 represents a
        # neutral scale of 1.0. Use it as a deterministic fallback instead of
        # leaving the parameter with uninitialized memory when a checkpoint is
        # missing a layer's V-cache scale.
        weight_param = torch.nn.Parameter(
            torch.full((hidden_size,), 127, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("v_cache_scale", weight_param)
        # Some ModelSlim recipes emit a V offset next to the scale, borrowed
        # from the affine FAKQuant template. MXFP8 per-channel is symmetric and
        # the operator takes no offset, so the only correct value is zero --
        # register it to check that, rather than dropping it unread.
        offset_param = torch.nn.Parameter(
            torch.zeros((hidden_size,), dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("v_cache_offset", offset_param)
        # When loading weights, segment them according to TP
        loader = functools.partial(_quant_weight_loader, head_size_v=layer.head_size_v)
        weight_param.weight_loader = loader
        offset_param.weight_loader = loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        vllm_config = get_current_vllm_config()
        target_dtype = vllm_config.model_config.dtype
        offset = layer.v_cache_offset.data
        if bool(offset.any()):
            raise RuntimeError(
                "[vllm-ascend/MXFP8_PER_CHANNEL] V cache offset is non-zero "
                f"(min={float(offset.min())} max={float(offset.max())}), but the MXFP8 "
                "per-channel scheme and the QuantFlashAttn operator are both symmetric. "
                "This checkpoint was calibrated with an affine V quantizer and cannot be "
                "served by this scheme."
            )
        # A minmax calibrator emits 0 for a channel whose absmax was 0, and
        # 2^-127 there would make the quantization reciprocal 2^127 -- any
        # activation that is not exactly zero at inference would go to inf.
        # Treat those channels as unscaled instead; if they really are all
        # zero the result is identical, and if they are not the cost is
        # precision rather than a poisoned cache.
        raw = layer.v_cache_scale.data
        # All-127 here means nothing was loaded and V is being cached
        # unscaled -- the silent state this mapping was added to fix.
        logger.info_once(
            "[quantization] C8 MXFP V cache scale: min=%s max=%s distinct=%s (all-127 => not loaded)",
            int(raw.min()),
            int(raw.max()),
            int(raw.unique().numel()),
        )
        exponent = torch.where(raw == 0, torch.full_like(raw, 127), raw).to(torch.float32) - 127
        layer.v_cache_scale_float = torch.nn.Parameter(torch.exp2(exponent).to(target_dtype), requires_grad=False)
        layer.v_cache_scale_float_reciprocal = torch.nn.Parameter(
            1 / torch.exp2(exponent).to(target_dtype),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendC8MXFPKVCacheAttentionMethod.apply should not be called. "
            "C8_MXFP KV cache quantization is handled by the attention backend."
        )
