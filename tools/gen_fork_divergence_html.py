#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 index.html —— 可视化本私仓相对 upstream/main 的代码结构与变更（用作 GitHub Pages 首页）。

用法（在仓库根目录执行）:
    git fetch upstream
    python tools/gen_fork_divergence_html.py

脚本会运行 `git diff upstream/main <当前分支>`，解析每个文件的 diff，
配合下方 FILES_META 的分类/说明，渲染为自包含单页 HTML（无外部依赖，离线可用）。

分类叠加：本仓由 upstream/main → 私仓自有 → 提前合入 PR → 自定义算子 逐层叠加而成，
FILES_META 为每个文件标注所属类目（来自某上游 PR / 私仓自有）。每张卡片直接展示该文件
**相对 upstream/main 的当前差异**（git diff upstream/main <当前分支>），即"本仓相对上游
改了什么"——天然适配上游重组（文件改名/挪目录），不依赖任何硬编码历史 commit。
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from collections import OrderedDict

BASE = "upstream/main"


def _detect_head() -> str:
    """动态探测当前分支名作为 diff 目标；detached HEAD（无分支名）时回退为 'HEAD'。"""
    try:
        name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
        return name or "HEAD"
    except subprocess.CalledProcessError:
        return "HEAD"


HEAD = _detect_head()  # 当前所在分支（原先硬编码为 "alpha"）
OUT = "index.html"

# 分类定义：id -> (中文标题, 颜色)
# 约定：来自上游未合并 PR 的改动各自单独成类，与「私仓自有」严格区分。
CATEGORIES = OrderedDict(
    [
        ("pr_9310", ("来自 PR #9310 · Chunk 元数据预构建 + GDN Attn Builder 重构 + Eagle Spec Decode", "#9333ea")),
        ("pr_9715", ("来自 PR #9715 · 修复 scheduler 版本兼容性导致的运行时错误（精简 balance scheduler 补丁）", "#db61a2")),
        ("deps", ("私仓自有 · 依赖与构建", "#7c3aed")),
        ("dev", ("私仓自有 · 开发调试", "#f59e0b")),
        ("ops", ("私仓自有 · 自定义算子（scatter_pa_kv_cache）", "#06b6d4")),
    ]
)

# 每个变更条目的元数据：(path, category_id, 说明, 是否为新增文件)
# 注意：同一个 path 可以出现多次（被多个类目/层修改），每条对应一张卡片，
# 卡片只展示该类目所属层的 diff。
FILES_META = [
    # ============================ 来自 PR #9310 ============================
    (
        "tests/ut/attention/a2/test_attention_v1.py",
        "pr_9310", "在既有上游单测中新增用例 test_unpadded_preserves_internal_seq_lens_cpu：校验 unpadded 后内部 _seq_lens_cpu 正确截断、seq_lens_cpu 保持 None", False,
    ),
    (
        "tests/ut/ops/test_gdn_attn_builder.py",
        "pr_9310", "由 tests/ut/patch/worker/patch_common/test_patch_gdn_attn.py 重命名迁移并扩展：覆盖 AscendGDNAttentionMetadataBuilder 的 chunk meta 预构建与 argsort stable 排序逻辑", False,
    ),
    (
        "tests/ut/ops/a2/test_gdn_chunk_meta.py",
        "pr_9310", "新增 GDN chunk 元数据用例，覆盖 _build_seq_lens / _validate_cu_seqlens / build_chunk_meta_device", False,
    ),
    (
        "tests/ut/spec_decode/a2/test_eagle_proposer.py",
        "pr_9310", "新增 Ascend Eagle Proposer 用例，覆盖异步 spec decode 下 proposer 初始化与行为", False,
    ),
    (
        "vllm_ascend/ops/gdn.py",
        "pr_9310", "新增 get_attn_backend() 返回 AscendGDNAttentionBackend；AscendGatedDeltaNetAttention 移除对 monkey-patch 的依赖，改用正式 ops 模块", False,
    ),
    (
        "vllm_ascend/ops/gdn_attn_builder.py",
        "pr_9310", "由 vllm_ascend/patch/worker/patch_gdn_attn.py 重命名重构为正式 ops 模块：实现 AscendGDNAttentionBackend / AscendGDNAttentionMetadataBuilder / GDNChunkedPrefillMetadata，prefill/decode 路径预构建 varlen chunk 元数据并附加到 attn_metadata，避免 forward 时 host→device round-trip", False,
    ),
    (
        "vllm_ascend/ops/triton/fla/chunk.py",
        "pr_9310", "chunk_gated_delta_rule_fwd / chunk_fwd_o：新增 cu_seqlens_host / chunk_indices_chunk64_host 通过 prebuilt meta 提前提取为 Python tuple，传入 AscendC 算子时避免每次 .tolist() 的同步开销", False,
    ),
    (
        "vllm_ascend/patch/__init__.py",
        "pr_9310", "更新补丁注释文档：patch_gdn_attn 说明替换为 patch_module（torch.argsort 补丁 + gdn_attn_builder 覆盖），标注相关 PR 与未来移除计划", False,
    ),
    (
        "vllm_ascend/patch/worker/__init__.py",
        "pr_9310", "移除 `import vllm_ascend.patch.worker.patch_gdn_attn`：GDN attn builder 已正式化至 ops 模块", False,
    ),
    (
        "vllm_ascend/spec_decode/eagle_proposer.py",
        "pr_9310", "简化 AscendEagleProposer.__init__：移除冗余命名参数传递，改为位置参数直接调用父类", False,
    ),
    (
        "vllm_ascend/worker/model_runner_v1.py",
        "pr_9310", "异步 spec decode 路径优化：将 num_computed_tokens_cpu_tensor→device 拷贝提前复用，避免重复 H2D 拷贝", False,
    ),
    # ============================ 来自 PR #9715 ============================
    (
        "vllm_ascend/patch/platform/patch_balance_schedule.py",
        "pr_9715", "精简 balance scheduler 补丁：删去随上游已收敛的大段重复 Scheduler 实现，仅保留版本兼容性修复所需的最小改动，消除 scheduler 运行时错误（+44/-621）", False,
    ),
    # ============================ 私仓自有 ============================
    (
        "requirements.txt",
        "deps", "numpy 锁定 1.26.4；注释掉 torch-npu==2.10.0 与 triton-ascend==3.2.1，避免安装时覆盖已装环境", False,
    ),
    # ============================ 私仓自有 · 开发调试 ============================
    (
        "vllm_ascend/profiler/torch_npu_profiler.py",
        "dev", "NPU profiler 默认开启 PipeUtilization 指标，方便查看算子利用率（关闭无用的 AiCoreNone）", False,
    ),
    # ============================ 私仓自有 · 自定义算子（scatter_pa_kv_cache） ============================
    # --- 共享构建接线 ---
    (
        "csrc/build_aclnn.sh",
        "ops", "ascend950 分支 CUSTOM_OPS_ARRAY 加入 scatter_pa_kv_cache（mega_moe 以注释保留为 DEFERRED 占位）", False,
    ),
    (
        "csrc/CMakeLists.txt",
        "ops", "新增注释说明 MC2_OPT 钩子由 mega_moe 触发（当前以注释保留，待 mega_moe 单独接入时启用）", False,
    ),
    (
        "csrc/torch_binding.cpp",
        "ops", "新增 npu_scatter_pa_kv_cache wrapper + ops.def/impl（+33 行，scatter_pa 算子注册）", False,
    ),
    (
        "csrc/torch_binding_meta.cpp",
        "ops", "新增 npu_scatter_pa_kv_cache_meta + Meta ops.impl，并修复 Tensor(a!)/Tensor(b!) alias 约束（返回输入自身而非新建 tensor，适配 FULL_AND_PIECEWISE 图捕获）", False,
    ),
    # --- Python 侧算子桥接 ---
    (
        "vllm_ascend/device/device_op.py",
        "ops", "A5 上通过 _ensure_custom_ops_loaded 在 get_device_adaptor 时一次性加载 vllm_ascend_C，绕过 enable_custom_op 的 A5 gate，使全部 custom op（scatter_pa、fused_gdn_gating、recurrent_gated_delta_rule、causal_conv1d 等）可用；reshape_and_cache 改为直调 torch.ops._C_ascend（+32 行）", False,
    ),
    # --- scatter_pa_kv_cache 子树：CMake 构建 ---
    (
        "csrc/attention/scatter_pa_kv_cache/CMakeLists.txt",
        "ops", "算子顶层 CMake：自动 glob op_host / op_kernel 子目录", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/CMakeLists.txt",
        "ops", "op_host CMake：950 专用 COMPUTE_UNIT=Ascend950PR_9599，注册 aclnn_exclude", True,
    ),
    # --- scatter_pa_kv_cache 子树：平台配置 ---
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/config/ascend910_93/scatter_pa_kv_cache_binary.json",
        "ops", "ascend910_93 平台 tiling 编译产物（binary config，1812 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/config/ascend910_93/scatter_pa_kv_cache_simplified_key.ini",
        "ops", "ascend910_93 simplified key 配置", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/config/ascend910b/scatter_pa_kv_cache_binary.json",
        "ops", "ascend910b 平台 tiling 编译产物（binary config，1812 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/config/ascend910b/scatter_pa_kv_cache_simplified_key.ini",
        "ops", "ascend910b simplified key 配置", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/config/ascend950/scatter_pa_kv_cache_simplified_key.ini",
        "ops", "ascend950 simplified key 配置", True,
    ),
    # --- scatter_pa_kv_cache 子树：op_host 算子定义 ---
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/op_api/aclnn_scatter_pa_kv_cache.cpp",
        "ops", "aclnn 算子 API 实现（设备端调用入口，554 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/op_api/aclnn_scatter_pa_kv_cache.h",
        "ops", "aclnn 算子 API 头文件", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/op_api/scatter_pa_kv_cache.cpp",
        "ops", "算子封装（参数校验 → aclnn 调用）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/op_api/scatter_pa_kv_cache.h",
        "ops", "算子封装头文件", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_def.cpp",
        "ops", "算子定义注册（op def，396 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_infershape.cpp",
        "ops", "输出 shape/type 推导（infershape，82 行）", True,
    ),
    # --- scatter_pa_kv_cache 子树：op_host tiling ---
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp",
        "ops", "tiling 计算（确定各 core 的分块策略，739 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.h",
        "ops", "tiling 数据结构定义（288 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling_arch35.cpp",
        "ops", "arch35（ascend950）专用 tiling 实现（1242 行）", True,
    ),
    # --- scatter_pa_kv_cache 子树：op_kernel arch35 内核 ---
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/common.h",
        "ops", "arch35 kernel 公共宏/常量定义", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_fully_load.h",
        "ops", "arch35 kernel：alibi 全加载路径（253 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_not_fully_load.h",
        "ops", "arch35 kernel：alibi 非全加载路径（235 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_norm_non_contiguous.h",
        "ops", "arch35 kernel：normal 非连续布局路径（349 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_fully_load.h",
        "ops", "arch35 kernel：normal 全加载路径（217 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_normal_not_fully_load.h",
        "ops", "arch35 kernel：normal 非全加载路径（220 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_nz_fully_load.h",
        "ops", "arch35 kernel：NZ 布局全加载路径（134 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_nz_non_contiguous.h",
        "ops", "arch35 kernel：NZ 布局非连续路径（382 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_nz_not_fully_load.h",
        "ops", "arch35 kernel：NZ 布局非全加载路径（194 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_omni_fully_load.h",
        "ops", "arch35 kernel：omni 全加载路径（286 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_omni_not_fully_load.h",
        "ops", "arch35 kernel：omni 非全加载路径（281 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_fully_load.h",
        "ops", "arch35 kernel：rope 全加载路径（513 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_not_fully_load.h",
        "ops", "arch35 kernel：rope 非全加载路径（580 行）", True,
    ),
    # --- scatter_pa_kv_cache 子树：op_kernel 公共实现 ---
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache.cpp",
        "ops", "kernel 主入口（dispatch 到各 arch35 变体，135 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_apt.cpp",
        "ops", "kernel APT（Ascend Pipeline Template）实现（252 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_common.h",
        "ops", "kernel 公共数据结构定义", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_compress_alibi.h",
        "ops", "kernel：compress alibi 变体（119 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_compress_common.h",
        "ops", "kernel：compress 公共逻辑（255 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_compress_omni.h",
        "ops", "kernel：compress omni 变体（181 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_compress_rope.h",
        "ops", "kernel：compress rope 变体（265 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_nhsd.h",
        "ops", "kernel：NHSD（Non-Hierarchical Scatter Dispatch）实现（157 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_normal.h",
        "ops", "kernel：normal 路径主实现（149 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_normal_common.h",
        "ops", "kernel：normal 路径公共逻辑（67 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_normal_nz_fully_load.h",
        "ops", "kernel：normal + NZ 布局全加载（121 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_normal_nz_not_fully_load.h",
        "ops", "kernel：normal + NZ 布局非全加载（188 行）", True,
    ),
    (
        "csrc/attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_normal_siso.h",
        "ops", "kernel：normal 单输入单输出（SISO）路径（110 行）", True,
    ),
    # --- scatter_pa_kv_cache 测试 ---
    (
        "tests/e2e/nightly/single_node/ops/singlecard_ops/test_scatter_pa_kv_cache.py",
        "ops", "单算子 e2e 测试（覆盖 normal/rope/alibi/omni/compress 等 KV cache 场景，132 行）", True,
    ),
]

# 提前合入的上游 PR：下列文件的改动来自尚未合并进 upstream/main 的 PR，
# 为本仓需要而提前合入；待上游合并后即可随上游同步、移除本地副本。
# 注：PR #9382（GDN A5 自定义算子）已于上游合并（upstream/main 含 #9382），
# 本仓的提前合入副本已随本次同步 upstream/main 自动收敛，不再单独成类。
PR_9310 = {
    "url": "https://github.com/vllm-project/vllm-ascend/pull/9310",
    "title": "[Performance] Reuse prebuilt chunk host metadata for Ascend chunk ops and earse synchronize for qwen3.5 model",
    "state": "OPEN",
    "category": "pr_9310",
}

PR_9715 = {
    "url": "https://github.com/vllm-project/vllm-ascend/pull/9715",
    "title": "[Feature]Fix the scheduler runtime error caused by version compatibility issues.",
    "state": "OPEN",
    "category": "pr_9715",
}

# 所有上游 PR 的汇总列表，用于渲染 PR 标签和概述
PRS = [PR_9310, PR_9715]

# 类别 -> 该类别对应的上游 PR（用于卡片 PR 角标）。私仓自有类别返回 None。
CAT_PR = {pr["category"]: pr for pr in PRS}


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout


def parse_diffs(diff_text: str) -> dict[str, str]:
    """把 `git diff` 输出按文件切分，返回 path -> 该文件的 diff 文本。"""
    out: dict[str, str] = {}
    cur_path = None
    cur_lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if cur_path is not None:
                out[cur_path] = "\n".join(cur_lines)
            # diff --git a/<path> b/<path>
            cur_path = line.split(" b/", 1)[-1].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_path is not None:
        out[cur_path] = "\n".join(cur_lines)
    return out


def render_diff(diff_text: str) -> str:
    """把单个文件的 diff 渲染为带着色的 HTML 行。"""
    rows = []
    for line in diff_text.splitlines():
        esc = html.escape(line)
        if line.startswith("@@"):
            cls = "hunk"
        elif line.startswith("+++") or line.startswith("---"):
            cls = "meta"
        elif line.startswith(("index ", "new file", "deleted file", "similarity", "rename ", "old mode", "new mode")):
            cls = "meta"
        elif line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        else:
            cls = "ctx"
        rows.append(f'<div class="dl {cls}">{esc or "&nbsp;"}</div>')
    return "\n".join(rows)


def _tree(paths: list[str]) -> dict:
    root: dict = {}
    for p in paths:
        node = root
        for part in p.split("/"):
            node = node.setdefault(part, {})
    return root


def render_tree(node: dict, path_cats: dict[str, list[str]], path_anchor: dict[str, str], prefix: str = "") -> str:
    items = []
    for name in sorted(node.keys(), key=lambda k: (not bool(node[k]), k)):  # 目录在前
        child = node[name]
        full = f"{prefix}{name}"
        if child:  # 目录
            items.append(
                f'<li class="dir"><span class="tname">{html.escape(name)}/</span>'
                f'<ul>{render_tree(child, path_cats, path_anchor, full + "/")}</ul></li>'
            )
        else:  # 文件
            cats = path_cats.get(full, [])
            dots = "".join(
                f'<span class="dot" style="background:{CATEGORIES[c][1]}"></span>' for c in cats
            )
            anchor = path_anchor.get(full, "")
            items.append(
                f'<li class="file" data-cats="{" ".join(cats)}">{dots}'
                f'<a href="#{anchor}" class="tname">{html.escape(name)}</a></li>'
            )
    return "".join(items)


def anchor_for(path: str, cat: str = "") -> str:
    # 确定性锚点：基于 路径+类别 生成（同一文件在不同类别有不同卡片，需各自唯一）
    base = "f_" + re.sub(r"[^0-9A-Za-z]+", "_", path).strip("_")
    return f"{base}__{cat}" if cat else base


def main() -> int:
    base_sha = run(["git", "rev-parse", "--short", BASE]).strip()

    # ========== 相对上游的当前差异（git diff upstream/main HEAD）==========
    # 每张卡片直接展示该文件相对 upstream/main 的当前差异（按文件切分）。
    # 此前用硬编码历史 commit 计算「分层 diff」，但每次同步 upstream 都可能因上游
    # 重组（文件改名/挪目录）导致历史路径与当前路径对不上而失效；FILES_META 内
    # 每个文件均只属于一个类目（无跨类目重复），故按文件取「当前完整差异」即可
    # 准确表达「本仓相对上游改了什么」，且天然适配上游重组、不再依赖历史 commit。
    full_diff_text = run(["git", "diff", BASE, HEAD])
    full_diffs = parse_diffs(full_diff_text)
    full_numstat = run(["git", "diff", "--numstat", BASE, HEAD])
    full_ns: dict[str, tuple[str, str]] = {}
    for ln in full_numstat.splitlines():
        cols = ln.split("\t")
        if len(cols) == 3:
            full_ns[cols[2]] = (cols[0], cols[1])

    # 唯一文件路径（去重，保持出现顺序）及其所属类别列表 / 首张卡片锚点
    unique_paths: list[str] = []
    path_cats: dict[str, list[str]] = {}
    path_anchor: dict[str, str] = {}
    for path, cat, _desc, _new in FILES_META:
        if path not in path_cats:
            path_cats[path] = []
            unique_paths.append(path)
            path_anchor[path] = anchor_for(path, cat)  # 链接指向该文件的首张卡片
        path_cats[path].append(cat)

    # 头部统计：用完整 diff，但只统计 FILES_META 涉及的（去重）文件
    add_total = del_total = 0
    for ln in full_numstat.splitlines():
        cols = ln.split("\t")
        if len(cols) == 3:
            a, d, p = cols
            if p in path_cats:
                if a.isdigit():
                    add_total += int(a)
                if d.isdigit():
                    del_total += int(d)

    tree_html = render_tree(_tree(unique_paths), path_cats, path_anchor)

    # 分类计数（按条目数，同一文件在多个类别各计一次）
    cat_counts = {cid: 0 for cid in CATEGORIES}
    for _path, cat, _desc, _new in FILES_META:
        cat_counts[cat] += 1

    # 工具栏分类按钮
    cat_buttons = ['<button class="cat-btn active" data-cat="all">全部 ({})</button>'.format(len(unique_paths))]
    for cid, (title, color) in CATEGORIES.items():
        cat_buttons.append(
            f'<button class="cat-btn" data-cat="{cid}" style="--c:{color}">'
            f'<span class="dot" style="background:{color}"></span>{html.escape(title)} ({cat_counts[cid]})</button>'
        )

    # 文件卡片（按类别顺序，再按 FILES_META 内出现顺序）
    cards = []
    for cid in CATEGORIES:
        for path, cat, desc, is_new in FILES_META:
            if cat != cid:
                continue
            title, color = CATEGORIES[cid]
            # 展示该文件相对 upstream/main 的当前差异
            a, d = full_ns.get(path, ("-", "-"))
            dtext = full_diffs.get(path, "（该文件相对上游无差异 / 已收敛）")
            new_badge = '<span class="newtag">新增</span>' if is_new else ""
            pr = CAT_PR.get(cid)
            pr_badge = ""
            if pr:
                pr_num = pr["url"].rsplit("/", 1)[-1]
                pr_badge = (
                    f'<a class="prtag" href="{pr["url"]}" target="_blank" rel="noopener" '
                    f'title="{html.escape(pr["title"])}（{pr["state"]}）">PR #{pr_num} ↗</a>'
                )
            cards.append(
                f'''
<section class="card" id="{anchor_for(path, cid)}" data-cats="{cid}">
  <div class="card-h" style="--c:{color}">
    <div class="card-h-left">
      <span class="catbadge" style="background:{color}">{html.escape(title)}</span>
      {new_badge}
      {pr_badge}
      <code class="path">{html.escape(path)}</code>
    </div>
    <div class="card-h-right">
      <span class="stat add">+{a}</span><span class="stat del">-{d}</span>
    </div>
  </div>
  <div class="desc">{html.escape(desc)}</div>
  <details class="diff-wrap" open>
    <summary>查看 diff</summary>
    <div class="diff">{render_diff(dtext)}</div>
  </details>
</section>'''
            )

    cards_html = "\n".join(cards)
    cat_btn_html = "\n".join(cat_buttons)
    pr_notes = []
    for pr in PRS:
        pr_num = pr["url"].rsplit("/", 1)[-1]
        n_files = sum(1 for _p, c, _d, _n in FILES_META if c == pr["category"])
        pr_notes.append(
            f'其中 <b>{n_files}</b> 条改动来自上游 '
            f'<a href="{pr["url"]}" target="_blank" rel="noopener">PR #{pr_num}</a>'
            f'（{html.escape(pr["title"])}，当前 <b>{pr["state"]}</b>，为本仓需要提前合入；'
            f'待上游合并后可随上游同步移除本地副本）。'
        )
    pr_note = "<br>".join(pr_notes)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fork Divergence —— 私仓相对上游的代码结构与变更</title>
<style>
:root {{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2129; --border:#30363d;
  --fg:#e6edf3; --muted:#9da7b3; --add-bg:rgba(46,160,67,.18); --add-fg:#7ee787;
  --del-bg:rgba(248,81,73,.18); --del-fg:#ffa198; --hunk:#79c0ff; --ctx:#c9d1d9;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
code,.diff,.path {{ font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace; }}
a {{ color:var(--hunk); text-decoration:none; }}
header.top {{ padding:24px 28px; border-bottom:1px solid var(--border); background:linear-gradient(180deg,#161b22,#0d1117); }}
header.top h1 {{ margin:0 0 6px; font-size:22px; }}
header.top .sub {{ color:var(--muted); font-size:13px; line-height:1.7; }}
header.top .sub b {{ color:var(--fg); }}
.stats {{ display:flex; gap:14px; margin-top:14px; flex-wrap:wrap; }}
.stat-card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 16px; min-width:96px; }}
.stat-card .n {{ font-size:22px; font-weight:700; }}
.stat-card .l {{ font-size:12px; color:var(--muted); }}
.layout {{ display:grid; grid-template-columns:300px 1fr; gap:0; align-items:start; }}
aside {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto;
  border-right:1px solid var(--border); padding:16px; background:var(--panel); }}
aside h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:8px 0; }}
.tree {{ font-size:13px; line-height:1.9; }}
.tree ul {{ list-style:none; margin:0; padding-left:14px; border-left:1px dashed var(--border); }}
.tree > ul {{ padding-left:0; border:none; }}
.tree li.dir > .tname {{ color:var(--muted); }}
.tree li.file {{ position:relative; }}
.tree li.file a {{ color:var(--fg); }}
.tree li.file a:hover {{ color:var(--hunk); }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
main {{ padding:18px 22px 80px; min-width:0; }}
.toolbar {{ position:sticky; top:0; z-index:5; display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  padding:12px 0; background:var(--bg); border-bottom:1px solid var(--border); margin-bottom:18px; }}
.cat-btn {{ cursor:pointer; border:1px solid var(--border); background:var(--panel); color:var(--fg);
  padding:6px 12px; border-radius:20px; font-size:13px; display:inline-flex; align-items:center; }}
.cat-btn.active {{ background:var(--panel2); border-color:var(--muted); }}
.toolbar .spacer {{ flex:1; }}
.toolbar .mini {{ cursor:pointer; border:1px solid var(--border); background:transparent; color:var(--muted);
  padding:6px 10px; border-radius:8px; font-size:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; margin-bottom:18px; overflow:hidden; }}
.card-h {{ display:flex; justify-content:space-between; align-items:center; gap:12px;
  padding:12px 14px; border-bottom:1px solid var(--border); border-left:4px solid var(--c); background:var(--panel2); }}
.card-h-left {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; min-width:0; }}
.catbadge {{ color:#fff; font-size:11px; padding:2px 8px; border-radius:20px; white-space:nowrap; }}
.newtag {{ font-size:11px; color:#fff; background:#16a34a; padding:2px 7px; border-radius:6px; }}
.prtag {{ font-size:11px; color:#fff; background:#db61a2; padding:2px 8px; border-radius:6px; text-decoration:none; white-space:nowrap; }}
.prtag:hover {{ filter:brightness(1.12); }}
.path {{ font-size:13px; color:var(--fg); word-break:break-all; }}
.card-h-right {{ display:flex; gap:8px; white-space:nowrap; }}
.stat {{ font-size:12px; font-weight:700; }}
.stat.add {{ color:var(--add-fg); }} .stat.del {{ color:var(--del-fg); }}
.desc {{ padding:11px 14px; color:var(--muted); font-size:13.5px; line-height:1.7; border-bottom:1px solid var(--border); }}
.diff-wrap summary {{ cursor:pointer; padding:9px 14px; font-size:12.5px; color:var(--muted); user-select:none; }}
.diff-wrap summary:hover {{ color:var(--fg); }}
.diff {{ overflow-x:auto; font-size:12.5px; line-height:1.55; padding:6px 0 10px; }}
.dl {{ white-space:pre; padding:0 14px; }}
.dl.add {{ background:var(--add-bg); color:var(--add-fg); }}
.dl.del {{ background:var(--del-bg); color:var(--del-fg); }}
.dl.hunk {{ color:var(--hunk); background:rgba(56,139,253,.1); }}
.dl.meta {{ color:var(--muted); }}
.dl.ctx {{ color:var(--ctx); }}
.hidden {{ display:none !important; }}
footer {{ color:var(--muted); font-size:12px; padding:18px 22px; border-top:1px solid var(--border); }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ position:static; height:auto; }} }}
</style>
</head>
<body>
<header class="top">
  <h1>Fork Divergence · 私仓相对上游的代码结构与变更</h1>
  <div class="sub">
    上游基线：<b>{BASE}</b> @ <b>{base_sha}</b>　·　本仓分支：<b>{HEAD}</b><br>
    目的：在 <b>Atlas A5（ascend950 / arch35，__CCE_AICORE__ == 310）</b> 上运行 <b>Qwen3.5</b>，修复 / 规避若干算子在 A5 上的精度与支持问题。<br>
    分层说明：本仓由 <b>upstream/main → 私仓自有 → PR #9310 → PR #9715 → scatter_pa_kv_cache 算子</b> 逐层叠加而成（PR #9382 已合入上游，提前合入副本随同步 upstream/main 自动收敛）；本页每张卡片直接展示该文件 <b>相对 upstream/main 的当前差异</b>（<code>git diff {BASE} {HEAD}</code>），即"本仓相对上游改了什么"。<br>
    {pr_note}<br>
    重新生成：<code>git fetch upstream &amp;&amp; python tools/gen_fork_divergence_html.py</code>
  </div>
  <div class="stats">
    <div class="stat-card"><div class="n">{len(unique_paths)}</div><div class="l">变更文件</div></div>
    <div class="stat-card"><div class="n">{len(CATEGORIES)}</div><div class="l">分类</div></div>
    <div class="stat-card"><div class="n" style="color:var(--add-fg)">+{add_total}</div><div class="l">新增行</div></div>
    <div class="stat-card"><div class="n" style="color:var(--del-fg)">-{del_total}</div><div class="l">删除行</div></div>
  </div>
</header>
<div class="layout">
  <aside>
    <h2>文件结构</h2>
    <div class="tree"><ul>{tree_html}</ul></div>
  </aside>
  <main>
    <div class="toolbar">
      {cat_btn_html}
      <span class="spacer"></span>
      <button class="mini" id="expand">展开全部</button>
      <button class="mini" id="collapse">折叠全部</button>
    </div>
    {cards_html}
    <footer>本页由 <code>git diff {BASE} {HEAD}</code> 自动生成；分类与说明见 tools/gen_fork_divergence_html.py。</footer>
  </main>
</div>
<script>
const btns = document.querySelectorAll('.cat-btn');
const hasCat = (el, cat) => cat === 'all' || (el.dataset.cats || '').split(' ').includes(cat);
btns.forEach(b => b.addEventListener('click', () => {{
  btns.forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  const cat = b.dataset.cat;
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('hidden', !hasCat(c, cat)));
  document.querySelectorAll('.tree li.file').forEach(li => li.classList.toggle('hidden', !hasCat(li, cat)));
}}));
document.getElementById('expand').onclick = () =>
  document.querySelectorAll('.card:not(.hidden) details').forEach(d => d.open = true);
document.getElementById('collapse').onclick = () =>
  document.querySelectorAll('details').forEach(d => d.open = false);
</script>
</body>
</html>
"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"已生成 {OUT}（{len(unique_paths)} 个文件 / {len(FILES_META)} 条改动，+{add_total}/-{del_total} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
