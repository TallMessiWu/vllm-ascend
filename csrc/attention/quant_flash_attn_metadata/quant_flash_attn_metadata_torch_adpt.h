/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUANT_FLASH_ATTN_METADATA_TORCH_ADPT_H
#define QUANT_FLASH_ATTN_METADATA_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_quant_flash_attn_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &v_descale,
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t quant_mode, int64_t mask_mode, int64_t win_left, int64_t win_right,
    c10::string_view layout_q, c10::string_view layout_q_descale,
    c10::string_view layout_kv, c10::string_view layout_out)
{
    TORCH_CHECK(seqused_q.has_value() || cu_seqlens_q.has_value(),
                "quant_flash_attn_metadata requires seqused_q or cu_seqlens_q to derive the batch size "
                "and the target device.");
    TORCH_CHECK(quant_mode == 1,
                "quant_flash_attn_metadata only supports quant_mode=1 "
                "(A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), but got ",
                quant_mode);
    if (batch_size <= 0) {
        if (seqused_q.has_value()) {
            batch_size = seqused_q->size(0);
        } else {
            batch_size = cu_seqlens_q->size(0) - 1;
        }
    }

    // The AICPU load-balance kernel always writes a fixed-size int32 blob;
    // keep in sync with the 4096-element allocation used by ops-transformer.
    constexpr int64_t METADATA_SIZE = 4096;
    const at::Tensor &device_ref = seqused_q.has_value() ? *seqused_q : *cu_seqlens_q;
    at::Tensor metadata = at::empty({METADATA_SIZE}, device_ref.options().dtype(at::kInt));

    std::string layout_q_str = std::string(layout_q);
    std::string layout_q_descale_str = std::string(layout_q_descale);
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_out_str = std::string(layout_out);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_q_descale_ptr = const_cast<char *>(layout_q_descale_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    EXEC_NPU_CMD(
        aclnnQuantFlashAttnMetadata,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        v_descale,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        mask_mode,
        win_left,
        win_right,
        layout_q_ptr,
        layout_q_descale_ptr,
        layout_kv_ptr,
        layout_out_ptr,
        metadata);
    return metadata;
}
}  // namespace vllm_ascend

#endif  // QUANT_FLASH_ATTN_METADATA_TORCH_ADPT_H
