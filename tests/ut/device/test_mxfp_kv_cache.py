import torch

from vllm_ascend.device.mxfp_kv_cache import (
    mxfp_k_scale_cache_shape,
    mxfp_v_scale_cache_shape,
    scatter_mxfp_k_scale_cache,
)


def test_mxfp_scale_cache_shapes_follow_pa_bbnd():
    # Block and head axes in the order QuantFlashAttn's PA_BBND reads them,
    # matching the K/V caches, so attention transposes neither.
    assert mxfp_k_scale_cache_shape(2, 128, 3, 64) == (2, 128, 3, 1, 2)
    assert mxfp_v_scale_cache_shape(2, 128, 3, 64) == (2, 2, 3, 64, 2)


def test_scatter_mxfp_k_scale_cache_ignores_full_graph_padding():
    # One block of 64 slots, one head, one 64-element scale group.
    key_scale_cache = torch.full((1, 64, 1, 1, 2), 7, dtype=torch.uint8)
    key_scale = torch.tensor(
        [
            [[[3, 4]]],
            [[[9, 9]]],
        ],
        dtype=torch.uint8,
    )
    # Second token is FULL-graph padding: slot -1 must not land on any slot.
    slot_mapping = torch.tensor([1, -1], dtype=torch.int64)

    scatter_mxfp_k_scale_cache(
        key_scale,
        key_scale_cache,
        slot_mapping,
        block_size=64,
    )

    torch.testing.assert_close(
        key_scale_cache[0, 1, 0],
        torch.tensor([[3, 4]], dtype=torch.uint8),
    )
    # Slot 0 is where padding gets remapped, so it is the one to check.
    torch.testing.assert_close(
        key_scale_cache[0, 0, 0],
        torch.tensor([[7, 7]], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        key_scale_cache[0, 2, 0],
        torch.tensor([[7, 7]], dtype=torch.uint8),
    )
