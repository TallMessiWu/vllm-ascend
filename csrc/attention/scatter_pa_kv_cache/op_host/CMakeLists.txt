add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        scatter_pa_kv_cache_def.cpp
    )
endif()

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ScatterPaKvCache
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
else()
    add_ops_compile_options(
        OP_NAME ScatterPaKvCache
        OPTIONS
            --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE scatter_pa_kv_cache ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
