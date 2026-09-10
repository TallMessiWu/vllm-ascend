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

// Ops::Base::ToString definitions that the upstream ops-transformer build
// links in from the CANN opbase library. csrc does not link opbase, so the
// vendored QFA checkers (base_checker.cpp / quant_checker.cpp) would leave
// these two symbols undefined inside libcust_opmaster_rt2.0.so -- and a
// single undefined symbol makes TBE's dlopen of the packaged tiling so fail,
// wiping tiling registration for EVERY standard-mechanism op in the package
// ("do not registe tiling struct"). Semantics mirror the inline helpers in
// ops-transformer's fused_infer_attention_score_tiling_utils.h.

#include <sstream>
#include <string>

#include "exe_graph/runtime/shape.h"
#include "graph/utils/type_utils.h"

namespace Ops {
namespace Base {

std::string ToString(ge::Format format)
{
    return ge::TypeUtils::FormatToSerialString(format);
}

std::string ToString(const gert::Shape &shape)
{
    std::ostringstream oss;
    oss << "[";
    const size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; ++i) {
        oss << shape.GetDim(i);
        if (i + 1 < dimNum) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

} // namespace Base
} // namespace Ops
