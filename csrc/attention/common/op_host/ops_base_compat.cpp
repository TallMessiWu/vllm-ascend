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

// CANN 9.1.0 op_common headers declare Ops::Base::ToString for gert::Shape and
// ge::Format, but its shipped libraries define neither, so liboptiling.so ends
// up with undefined symbols at dlopen time (newer CANN releases export them
// from the same base libraries). Provide local definitions with the exact
// mangled signatures (_ZN3Ops4Base8ToStringERKN4gert5ShapeE /
// _ZN3Ops4Base8ToStringEN2ge6FormatE); they resolve within this shared object
// and stay hidden-visibility, so a CANN that does ship them is unaffected.

#include <string>

#include "exe_graph/runtime/shape.h"
#include "graph/types.h"

namespace Ops {
namespace Base {

std::string ToString(const gert::Shape &shape)
{
    std::string result = "[";
    const size_t dim_num = shape.GetDimNum();
    for (size_t i = 0; i < dim_num; ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}

std::string ToString(ge::Format format)
{
    switch (format) {
        case ge::FORMAT_NCHW:
            return "NCHW";
        case ge::FORMAT_NHWC:
            return "NHWC";
        case ge::FORMAT_ND:
            return "ND";
        case ge::FORMAT_NC1HWC0:
            return "NC1HWC0";
        case ge::FORMAT_FRACTAL_Z:
            return "FRACTAL_Z";
        case ge::FORMAT_FRACTAL_NZ:
            return "FRACTAL_NZ";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int32_t>(format)) + ")";
    }
}

}  // namespace Base
}  // namespace Ops
