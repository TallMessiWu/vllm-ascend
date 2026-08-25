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
#ifndef QUANT_FLASH_ATTN_TORCH_ADPT_H
#define QUANT_FLASH_ATTN_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

int64_t infer_quant_flash_attn_value_head_dim(const at::Tensor &value, const std::string &layout_kv_str)
{
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t DIM_4 = 4;
    if (layout_kv_str == "TND") {
        TORCH_CHECK(value.dim() == 3, "value with layout TND expects 3 dims, but got ", value.dim());
        return value.size(DIM_2);
    }
    if (layout_kv_str == "PA_BBND" || layout_kv_str == "PA_BNBD") {
        TORCH_CHECK(value.dim() == 4, "value with layout ", layout_kv_str, " expects 4 dims, but got ", value.dim());
        return value.size(DIM_3);
    }
    if (layout_kv_str == "PA_NZ") {
        TORCH_CHECK(value.dim() == 5, "value with layout PA_NZ expects 5 dims, but got ", value.dim());
        return value.size(DIM_2) * value.size(DIM_4);
    }
    TORCH_CHECK(false, "Unsupported layout_kv for quant_flash_attn: ", layout_kv_str,
                ", expected one of TND/PA_BBND/PA_BNBD/PA_NZ");
    return 0;
}

}  // namespace

at::Tensor npu_quant_flash_attn(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
    const at::Tensor &metadata, double softmax_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &attn_mask,
    int64_t quant_mode, int64_t mask_mode, int64_t win_left, int64_t win_right,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_q_descale,
    c10::string_view layout_kv, c10::string_view layout_out)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.");
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.");
    TORCH_CHECK(v.numel() > 0, "Tensor v is empty.");
    TORCH_CHECK(quant_mode == 1,
                "quant_flash_attn only supports quant_mode=1 "
                "(A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), but got ",
                quant_mode);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_q_descale_str = std::string(layout_q_descale);
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_out_str = std::string(layout_out);
    TORCH_CHECK(layout_q_str == "TND" && layout_out_str == "TND",
                "quant_flash_attn with quant_mode=1 only supports layout_q=TND and layout_out=TND, but got layout_q=",
                layout_q_str, ", layout_out=", layout_out_str);
    TORCH_CHECK(q.dim() == 3, "q with layout TND expects 3 dims, but got ", q.dim());

    const int64_t value_head_dim = infer_quant_flash_attn_value_head_dim(v, layout_kv_str);
    at::Tensor attn_out = at::empty({q.size(0), q.size(1), value_head_dim}, q.options().dtype(at::kBFloat16));
    // Inference never needs softmax_lse; the aclnn op expects a {0}-shaped
    // float placeholder when return_softmax_lse is false.
    at::Tensor softmax_lse = at::empty({0}, q.options().dtype(at::kFloat));
    bool return_softmax_lse = false;

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_q_descale_ptr = const_cast<char *>(layout_q_descale_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    EXEC_NPU_CMD(
        aclnnQuantFlashAttn,
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        quant_mode,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q_ptr,
        layout_q_descale_ptr,
        layout_kv_ptr,
        layout_out_ptr,
        return_softmax_lse,
        attn_out,
        softmax_lse);
    return attn_out;
}
}  // namespace vllm_ascend

#endif  // QUANT_FLASH_ATTN_TORCH_ADPT_H
