import gc
import unittest

import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

# On A5 (Ascend 950) enable_custom_op() is intentionally a no-op: vllm-ascend
# currently gates ALL custom ops off on A5 (see the FIXME in vllm_ascend/utils.py),
# so it returns early without importing the extension and none of the
# torch.ops._C_ascend.* ops get registered. This single-op test calls
# npu_scatter_pa_kv_cache directly, so we bypass that gate and force-load the
# extension ourselves. Requires the custom op package's set_env.bash to have been
# sourced beforehand so the vendor op_api libs (libcust_opapi.so) resolve.
enable_custom_op()  # normal path on A2/A3; harmless no-op on A5
bootstrap_custom_op_env(include_vendor_lib=True)
# Only the eager (PrivateUse1) registration is needed here; importing
# vllm_ascend.meta_registration is intentionally avoided because it eagerly
# registers Meta kernels for ops that are not built into the A5 .so (e.g.
# get_masked_input_and_mask), which raises on this device.
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401  # registers torch.ops._C_ascend.* (eager)


class TestScatterPaKvCache(unittest.TestCase):
    """Single-op test for the custom aclnn op ``npu_scatter_pa_kv_cache``.

    Covers the "Norm" cache layout with scatter_mode "None" (scenario 2 in
    aclnnScatterPaKvCache.md), i.e. write the current step's key/value into the
    paged KV caches at the positions given by ``slot_mapping``:

        key:        [num_tokens, num_head, head_size]
        value:      [num_tokens, num_head, head_size]
        key_cache:  [num_blocks, block_size, num_head, head_size]   (in-place)
        value_cache:[num_blocks, block_size, num_head, head_size]   (in-place)
        slot_mapping[t] -> block = slot // block_size, offset = slot % block_size

    This is the simplest mode (ND format, no compression), so the golden value
    is an exact scatter-copy and the comparison can be bit-exact.
    """

    def compute_golden(self, key, value, key_cache, value_cache, slot_mapping, block_size):
        """CPU reference: scatter key/value into clones of the caches."""
        golden_k = key_cache.clone()
        golden_v = value_cache.clone()
        num_tokens = key.shape[0]
        for token_id in range(num_tokens):
            slot = int(slot_mapping[token_id].item())
            block_idx = slot // block_size
            block_offset = slot % block_size
            golden_k[block_idx, block_offset] = key[token_id]
            golden_v[block_idx, block_offset] = value[token_id]
        return golden_k, golden_v

    def test_scatter_pa_kv_cache_norm(self):
        # (num_blocks, block_size, num_head, head_size, num_tokens)
        test_cases = [
            (4, 128, 8, 128, 64),
            (4, 128, 8, 128, 512),  # num_tokens == num_blocks * block_size (full)
            (8, 64, 4, 128, 100),
            (2, 16, 2, 16, 4),  # tiny, easy to eyeball on failure
        ]
        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for num_blocks, block_size, num_head, head_size, num_tokens in test_cases:
                total_slots = num_blocks * block_size
                self.assertLessEqual(
                    num_tokens, total_slots, "num_tokens must fit in num_blocks * block_size"
                )
                with self.subTest(
                    dtype=dtype,
                    shape=f"(blocks={num_blocks}, bs={block_size}, h={num_head}, d={head_size}, t={num_tokens})",
                ):
                    key = torch.randn(num_tokens, num_head, head_size, dtype=dtype, device="npu")
                    value = torch.randn(num_tokens, num_head, head_size, dtype=dtype, device="npu")
                    key_cache = torch.randn(
                        num_blocks, block_size, num_head, head_size, dtype=dtype, device="npu"
                    )
                    value_cache = torch.randn(
                        num_blocks, block_size, num_head, head_size, dtype=dtype, device="npu"
                    )
                    # slots must be unique and in [0, total_slots - 1]
                    slot_mapping = torch.randperm(total_slots, device="npu")[:num_tokens].to(torch.int32)

                    golden_k, golden_v = self.compute_golden(
                        key, value, key_cache, value_cache, slot_mapping, block_size
                    )

                    key_cache_out, value_cache_out = torch.ops._C_ascend.npu_scatter_pa_kv_cache(
                        key,
                        key_cache,
                        slot_mapping,
                        value,
                        value_cache,
                        None,  # compress_lens
                        None,  # compress_seq_offset
                        None,  # seq_lens
                        "Norm",  # cache_mode
                        "None",  # scatter_mode
                        None,  # strides
                        None,  # offsets
                    )

                    # op updates the caches in place and returns the same tensors
                    self.assert_cache_equal(key_cache_out, golden_k, "key_cache")
                    self.assert_cache_equal(value_cache_out, golden_v, "value_cache")
                    # the in-place tensors must reflect the same update
                    self.assert_cache_equal(key_cache, golden_k, "key_cache(in-place)")
                    self.assert_cache_equal(value_cache, golden_v, "value_cache(in-place)")

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    def assert_cache_equal(self, actual, expected, name):
        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        self.assertEqual(actual_cpu.shape, expected_cpu.shape, f"{name}: shape mismatch")
        self.assertFalse(torch.isnan(actual_cpu).any(), f"{name}: result contains NaN")
        # Norm/None mode is a pure scatter-copy, so values are bit-exact.
        if not torch.equal(actual_cpu, expected_cpu):
            diff = (actual_cpu.float() - expected_cpu.float()).abs()
            mismatch = int((diff != 0).sum().item())
            raise AssertionError(
                f"{name}: scatter result mismatch; "
                f"mismatched_elems={mismatch}, max_abs_diff={diff.max().item()}"
            )


if __name__ == "__main__":
    unittest.main()
