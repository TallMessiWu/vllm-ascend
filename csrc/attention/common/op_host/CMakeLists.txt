# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPS_RTY_KERNEL)
else()
    add_modules_sources(OPTYPE common ACLNNTYPE no_need_aclnn)
    # ops_base_compat.cpp supplies the Ops::Base::ToString overloads that
    # CANN 9.1.0 declares but ships no definition for. The add_modules_sources
    # globs only collect *_tiling*.cpp, so register it explicitly with the
    # tiling object, which both liboptiling.so and libcust_opmaster_rt2.0.so
    # link.
    if (TARGET ${OPHOST_NAME}_tiling_obj)
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ops_base_compat.cpp)
    endif()
endif()
