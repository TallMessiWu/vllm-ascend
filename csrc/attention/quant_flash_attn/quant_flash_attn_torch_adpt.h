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

#include <mutex>
#include <string>
#include <tuple>
#include <unordered_set>

namespace vllm_ascend {
namespace quant_flash_attn_detail {

constexpr int64_t QUANT_MODE_MXFP8 = 1;
// The AICPU metadata op writes its split plan into a fixed-size int32 buffer;
// 4096 mirrors _calculate_metadata_size in the upstream torch_extension.
constexpr int64_t METADATA_NUM_INT32 = 4096;

// The aclnn executor keeps attribute pointers alive until ACL graph capture
// finalizes, so layout strings must outlive this call (see the dangling-
// pointer note in msa_index_score_torch_adpt.h). Layout vocabulary is tiny
// (TND/BSND/PA_BBND/...), so intern each distinct value for process lifetime.
inline char *InternedLayout(c10::string_view layout)
{
    static std::mutex mu;
    static std::unordered_set<std::string> pool;
    std::lock_guard<std::mutex> lock(mu);
    const std::string &stored = *pool.emplace(layout).first;
    return const_cast<char *>(stored.c_str());
}

inline int64_t ResolveBatchSize(
    const c10::optional<int64_t> &batch_size,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q)
{
    // Mirrors _calculate_batch_size in the upstream torch_extension: an
    // explicit value wins, otherwise derive B from cu_seqlens_q (B+1 entries)
    // or seqused_q (B entries).
    if (batch_size.has_value()) {
        return batch_size.value();
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q.value().defined()) {
        return cu_seqlens_q.value().size(0) - 1;
    }
    if (seqused_q.has_value() && seqused_q.value().defined()) {
        return seqused_q.value().size(0);
    }
    return -1;
}

}  // namespace quant_flash_attn_detail

at::Tensor npu_quant_flash_attn_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &v_descale, c10::optional<int64_t> batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t mask_mode,
    int64_t win_left, int64_t win_right,
    c10::string_view layout_q, c10::string_view layout_q_descale,
    c10::string_view layout_kv, c10::string_view layout_out)
{
    namespace detail = quant_flash_attn_detail;
    const c10::optional<at::Tensor> *device_source = nullptr;
    for (const auto *candidate : {&cu_seqlens_q, &cu_seqlens_kv, &seqused_q, &seqused_kv, &v_descale}) {
        if (candidate->has_value() && candidate->value().defined()) {
            device_source = candidate;
            break;
        }
    }
    TORCH_CHECK(device_source != nullptr,
                "npu_quant_flash_attn_metadata needs at least one tensor input "
                "(cu_seqlens/seqused/v_descale) to run on the NPU");
    at::Tensor metadata = at::empty(
        {detail::METADATA_NUM_INT32},
        device_source->value().options().dtype(at::kInt));

    int64_t resolved_batch = detail::ResolveBatchSize(batch_size, cu_seqlens_q, seqused_q);
    char *layout_q_ptr = detail::InternedLayout(layout_q);
    char *layout_q_descale_ptr = detail::InternedLayout(layout_q_descale);
    char *layout_kv_ptr = detail::InternedLayout(layout_kv);
    char *layout_out_ptr = detail::InternedLayout(layout_out);

    EXEC_NPU_CMD(aclnnQuantFlashAttnMetadata, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                 v_descale, resolved_batch, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv,
                 head_dim, quant_mode, mask_mode, win_left, win_right, layout_q_ptr,
                 layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr, metadata);
    return metadata;
}

std::tuple<at::Tensor, at::Tensor> npu_quant_flash_attn(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
    int64_t quant_mode,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_q_descale,
    c10::string_view layout_kv, c10::string_view layout_out,
    bool return_softmax_lse)
{
    namespace detail = quant_flash_attn_detail;
    // MXFP8 passes fp8/e8m0 tensors straight through ConvertType (both dtypes
    // are in kATenScalarTypeToAclDataTypeTable). HIF8/MXFP4 need uint8 views
    // re-typed to ACL_HIFLOAT8/ACL_FLOAT4_E2M1, which EXEC_NPU_CMD cannot
    // express today.
    TORCH_CHECK(quant_mode == detail::QUANT_MODE_MXFP8,
                "npu_quant_flash_attn currently supports only quant_mode=1 (MXFP8), got ",
                quant_mode);

    int64_t t_size = 0;
    int64_t n_size = 0;
    int64_t s_size = 0;
    int64_t b_size = 0;
    if (layout_q == "TND") {
        t_size = q.size(0);
        n_size = q.size(1);
    } else if (layout_q == "NTD") {
        n_size = q.size(0);
        t_size = q.size(1);
    } else if (layout_q == "BSND") {
        b_size = q.size(0);
        s_size = q.size(1);
        n_size = q.size(2);
    } else {  // BNSD
        b_size = q.size(0);
        n_size = q.size(1);
        s_size = q.size(2);
    }
    int64_t d_size = 0;
    if (layout_kv == "TND") {
        d_size = v.size(2);
    } else if (layout_kv == "PA_NZ") {
        d_size = v.size(2) * v.size(4);
    } else {
        d_size = v.size(v.dim() - 1);
    }

    at::SmallVector<int64_t, 4> lse_shape;
    if (return_softmax_lse) {
        if (q.dim() == 3) {
            lse_shape = {n_size, t_size};
        } else {
            lse_shape = {b_size, n_size, s_size};
        }
    } else {
        lse_shape = {0};
    }
    at::Tensor softmax_lse = at::empty(lse_shape, q.options().dtype(at::kFloat));

    at::SmallVector<int64_t, 4> out_shape;
    if (layout_out == "TND") {
        out_shape = {t_size, n_size, d_size};
    } else if (layout_out == "BNSD") {
        out_shape = {b_size, n_size, s_size, d_size};
    } else {  // BSND
        out_shape = {b_size, s_size, n_size, d_size};
    }
    at::Tensor attn_out = at::empty(out_shape, q.options().dtype(at::kBFloat16));

    char *layout_q_ptr = detail::InternedLayout(layout_q);
    char *layout_q_descale_ptr = detail::InternedLayout(layout_q_descale);
    char *layout_kv_ptr = detail::InternedLayout(layout_kv);
    char *layout_out_ptr = detail::InternedLayout(layout_out);

    EXEC_NPU_CMD(aclnnQuantFlashAttn, q, k, v, q_descale, k_descale, v_descale, block_table,
                 p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask,
                 metadata, quant_mode, softmax_scale, mask_mode, win_left, win_right, max_seqlen_q,
                 max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr,
                 return_softmax_lse, attn_out, softmax_lse);
    return std::make_tuple(attn_out, softmax_lse);
}

// Same op, caller-owned outputs.
//
// aclnnQuantFlashAttn already takes attnOut/softmaxLse as inputs; the
// allocating overload above only hides that behind at::empty. Exposing it is
// what lets a captured call be re-issued through torch.npu.graph_task_update,
// which is how FIA replays inside an aclgraph (attention_v1.py:1288 re-runs
// npu_fused_infer_attention_score.out with the step's fresh tensors). Nothing
// about the kernel or the aclnn interface changes.
//
// Shapes are the caller's to get right -- there is no inference here, because
// under a graph capture the output tensor is the one the graph already owns.
std::tuple<at::Tensor, at::Tensor> npu_quant_flash_attn_out(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
    int64_t quant_mode,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_q_descale,
    c10::string_view layout_kv, c10::string_view layout_out,
    bool return_softmax_lse,
    at::Tensor &attn_out, at::Tensor &softmax_lse)
{
    namespace detail = quant_flash_attn_detail;
    TORCH_CHECK(quant_mode == detail::QUANT_MODE_MXFP8,
                "npu_quant_flash_attn.out currently supports only quant_mode=1 (MXFP8), got ",
                quant_mode);

    char *layout_q_ptr = detail::InternedLayout(layout_q);
    char *layout_q_descale_ptr = detail::InternedLayout(layout_q_descale);
    char *layout_kv_ptr = detail::InternedLayout(layout_kv);
    char *layout_out_ptr = detail::InternedLayout(layout_out);

    EXEC_NPU_CMD(aclnnQuantFlashAttn, q, k, v, q_descale, k_descale, v_descale, block_table,
                 p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask,
                 metadata, quant_mode, softmax_scale, mask_mode, win_left, win_right, max_seqlen_q,
                 max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr,
                 return_softmax_lse, attn_out, softmax_lse);
    return std::make_tuple(attn_out, softmax_lse);
}

}  // namespace vllm_ascend

#endif  // QUANT_FLASH_ATTN_TORCH_ADPT_H
