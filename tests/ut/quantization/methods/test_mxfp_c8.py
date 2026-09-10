from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import (
    AscendC8MXFPKVCacheAttentionMethod,
    _quant_weight_loader,
)


class TestAscendC8MXFPKVCacheAttentionMethod(TestBase):
    def test_missing_v_scale_uses_e8m0_unity_default(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(AscendC8MXFPKVCacheAttentionMethod)
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4

        with patch.object(
            layer,
            "register_parameter",
            wraps=layer.register_parameter,
        ):
            method.create_weights(layer)

        self.assertEqual(layer.v_cache_scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.v_cache_scale, torch.full((8,), 127, dtype=torch.uint8)))

    def test_affine_v_offset_is_rejected(self):
        # MXFP8 per-channel and the QFA operator are both symmetric, so a
        # non-zero offset means the checkpoint was calibrated for a scheme this
        # path cannot serve. Fail loudly instead of quantizing without it.
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(AscendC8MXFPKVCacheAttentionMethod)
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        method.create_weights(layer)
        layer.v_cache_offset.data[3] = 0.5

        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        with (
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_current_vllm_config",
                return_value=vllm_config,
            ),
            self.assertRaisesRegex(RuntimeError, "V cache offset is non-zero"),
        ):
            method.process_weights_after_loading(layer)

    def test_zero_v_offset_passes_through(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(AscendC8MXFPKVCacheAttentionMethod)
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        method.create_weights(layer)

        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        with patch(
            "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_current_vllm_config",
            return_value=vllm_config,
        ):
            method.process_weights_after_loading(layer)

        # The all-127 fallback is a neutral scale of 1.0.
        self.assertTrue(torch.equal(layer.v_cache_scale_float, torch.ones(8, dtype=torch.bfloat16)))

    def test_installs_c8_backend_with_512_token_blocks(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(AscendC8MXFPKVCacheAttentionMethod)
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        layer.impl = object.__new__(AscendC8MXFPAttentionBackendImpl.__base__)

        method.create_weights(layer)

        self.assertIs(layer.attn_backend, AscendC8MXFPAttentionBackend)
        self.assertIsInstance(layer.impl, AscendC8MXFPAttentionBackendImpl)
        self.assertEqual(AscendAttentionBackend.get_supported_kernel_block_sizes(), [128])
        self.assertEqual(AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes(), [512])


class TestQuantWeightLoader(TestBase):
    """How the static V-cache scale is spread over TP ranks.

    The scale is one E8M0 byte per (kv_head, channel), so the split is counted
    in KV heads. Dividing the tensor by tp_size instead breaks in both
    directions: a checkpoint that shares one set of channel scales across heads
    has nothing to divide, and a model whose total_kv_heads is below tp_size
    replicates a KV head rather than sharding it.
    """

    HEAD_SIZE_V = 4

    def _load(self, *, param_heads, ckpt_heads, tp_rank, tp_size):
        head = self.HEAD_SIZE_V
        param = torch.full((param_heads * head,), 127, dtype=torch.uint8)
        # Byte value encodes the head index, so the result names its source.
        loaded = torch.arange(ckpt_heads * head, dtype=torch.int32).div(head, rounding_mode="floor")
        with (
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
        ):
            _quant_weight_loader(param, loaded.to(torch.uint8), head_size_v=head)
        return param[::head].tolist()

    def test_shared_scale_is_tiled_over_kv_heads(self):
        # One set of channel scales for every KV head: TP=1 has to see it twice.
        self.assertEqual(self._load(param_heads=2, ckpt_heads=1, tp_rank=0, tp_size=1), [0, 0])

    def test_shared_scale_survives_tp_split(self):
        # Each rank holds one head, and both heads use the same set.
        for rank in range(2):
            self.assertEqual(self._load(param_heads=1, ckpt_heads=1, tp_rank=rank, tp_size=2), [0])

    def test_per_head_scale_is_sharded_by_rank(self):
        self.assertEqual(self._load(param_heads=1, ckpt_heads=2, tp_rank=0, tp_size=2), [0])
        self.assertEqual(self._load(param_heads=1, ckpt_heads=2, tp_rank=1, tp_size=2), [1])

    def test_per_head_scale_follows_replicated_kv_heads(self):
        # total_kv_heads=2 under TP=4: ranks 0/1 share head 0, ranks 2/3 head 1.
        self.assertEqual(
            [self._load(param_heads=1, ckpt_heads=2, tp_rank=r, tp_size=4)[0] for r in range(4)], [0, 0, 1, 1]
        )

    def test_trailing_axis_from_modelslim_is_accepted(self):
        param = torch.full((8,), 127, dtype=torch.uint8)
        loaded = torch.full((8, 1), 130, dtype=torch.uint8)
        with (
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
        ):
            _quant_weight_loader(param, loaded, head_size_v=4)
        self.assertTrue(torch.equal(param, torch.full((8,), 130, dtype=torch.uint8)))
