#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 index.html —— 可视化本私仓相对 upstream/main 的代码结构与变更（用作 GitHub Pages 首页）。

用法（在仓库根目录执行）:
    git fetch upstream
    python tools/gen_fork_divergence_html.py

脚本会运行 `git diff upstream/main <当前分支>`，解析每个文件的 diff，
配合下方 FILES_META 的分类/说明，渲染为自包含单页 HTML（无外部依赖，离线可用）。

本仓 main 分支直接追踪 upstream/main，仅保留少量本地补丁（依赖锁定、忽略项、差异追踪工具）。
FILES_META 为每个文件标注说明。每张卡片直接展示该文件
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
        ("local", ("私仓本地补丁 · 依赖、构建与差异追踪", "#7c3aed")),
    ]
)

# 每个变更条目的元数据：(path, category_id, 说明, 是否为新增文件[, match 正则列表])
FILES_META = [
    (
        "requirements.txt",
        "local", "numpy 锁定 1.26.4；注释掉 torch-npu==2.10.0 与 triton-ascend==3.2.1，避免安装时覆盖已装环境", False,
    ),
    (
        ".gitignore",
        "local", "追加忽略项：私仓本地配置（CLAUDE.local.md、.claude/settings.local.json、AGENTS.local.md）与构建产物目录（csrc/build_out/）", False,
    ),
    (
        "tools/gen_fork_divergence_html.py",
        "local", "差异可视化生成脚本：扫描本仓相对 upstream/main 的代码差异，生成自包含 HTML 页面", True,
    ),
    (
        "index.html",
        "local", "GitHub Pages 首页：本仓相对上游的差异可视化页面（由 gen_fork_divergence_html.py 自动生成）", True,
    ),
    (
        ".nojekyll",
        "local", "禁用 GitHub Pages 的 Jekyll 处理，确保 index.html 原样渲染", True,
    ),
]

PRS = []
CAT_PR = {}


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


def _render_diff_line(line: str) -> str:
    """把单行 diff 渲染为带着色的 HTML。"""
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
    return f'<div class="dl {cls}">{esc or "&nbsp;"}</div>'


def render_diff(diff_text: str) -> str:
    """把单个文件的完整 diff 渲染为带着色的 HTML 行。"""
    return "\n".join(_render_diff_line(line) for line in diff_text.splitlines())


def _is_change_line(ln: str) -> bool:
    """是否为实际增删行（排除 +++/--- 文件头）。"""
    return ln[:1] in ("+", "-") and not ln.startswith(("+++", "---"))


def parse_hunks(file_diff: str):
    """切分单文件 diff 为 (header_lines, hunks)；hunk = [hunk_header, [body_lines]]。"""
    header: list[str] = []
    hunks: list[list] = []
    cur = None
    for ln in file_diff.splitlines():
        if ln.startswith("@@"):
            if cur is not None:
                hunks.append(cur)
            cur = [ln, []]
        elif cur is None:
            header.append(ln)
        else:
            cur[1].append(ln)
    if cur is not None:
        hunks.append(cur)
    return header, hunks


def assign_owners(hunks, entries):
    """为每个 hunk 的每个改动行（+/-）判定归属类目。

    entries: [(cat, [compiled_regex, ...]), ...]，按 FILES_META 出现顺序。
    判定规则：先按正则匹配「行内容（去掉首列 +/-）」，命中第一个 entry 即归之；
    未命中的改动行用「双向就近继承」——先取本 hunk 内上方最近、再取下方最近的
    已判定改动行的类目（空行、括号、装饰器等通用行借此跟随相邻代码块）。
    返回 (owners, unassigned)：owners 与 hunks 平行；unassigned 为始终无法判定的行。
    """
    owners = []
    unassigned: list[str] = []
    for _hh, body in hunks:
        row_owner: list = [None] * len(body)
        for i, ln in enumerate(body):
            if _is_change_line(ln):
                content = ln[1:]
                for cat, regexes in entries:
                    if any(rx.search(content) for rx in regexes):
                        row_owner[i] = cat
                        break
        for i, ln in enumerate(body):
            if _is_change_line(ln) and row_owner[i] is None:
                owner = None
                for j in range(i - 1, -1, -1):
                    if _is_change_line(body[j]) and row_owner[j] is not None:
                        owner = row_owner[j]
                        break
                if owner is None:
                    for j in range(i + 1, len(body)):
                        if _is_change_line(body[j]) and row_owner[j] is not None:
                            owner = row_owner[j]
                            break
                row_owner[i] = owner
                if owner is None:
                    unassigned.append(ln)
        owners.append(row_owner)
    return owners, unassigned


def owner_counts(hunks, owners, this_cat) -> tuple[int, int]:
    """统计归属 this_cat 的增删行数。"""
    a = d = 0
    for (_hh, body), row_owner in zip(hunks, owners):
        for i, ln in enumerate(body):
            if row_owner[i] == this_cat:
                if ln.startswith("+"):
                    a += 1
                elif ln.startswith("-"):
                    d += 1
    return a, d


def render_split_diff(header, hunks, owners, this_cat, cat_titles) -> str:
    """渲染某类目专属的 diff：只展示属于 this_cat 的增删行；不属于本类目的连续改动行
    折叠为一行「⋯ 省略 N 行 ⋯」并标注其所属类目。仅含本类目改动行的 hunk 才输出。"""
    rows = [_render_diff_line(ln) for ln in header]
    for (hh, body), row_owner in zip(hunks, owners):
        if not any(row_owner[i] == this_cat for i in range(len(body)) if _is_change_line(body[i])):
            continue
        rows.append(_render_diff_line(hh))
        i = 0
        while i < len(body):
            ln = body[i]
            if _is_change_line(ln) and row_owner[i] != this_cat:
                j = i
                others = set()
                while j < len(body) and _is_change_line(body[j]) and row_owner[j] != this_cat:
                    if row_owner[j]:
                        others.add(row_owner[j])
                    j += 1
                names = "、".join(cat_titles.get(c, c) for c in sorted(others)) or "其它类目"
                rows.append(
                    f'<div class="dl fold">⋯ 省略 {j - i} 行（属于：{html.escape(names)}，见对应卡片）⋯</div>'
                )
                i = j
            else:
                rows.append(_render_diff_line(ln))
                i += 1
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
    # 每张卡片展示该文件相对 upstream/main 的当前差异（按文件切分）。按文件取「当前完整
    # 差异」即可准确表达「本仓相对上游改了什么」，天然适配上游重组（改名/挪目录），不依赖
    # 历史 commit。一个文件若被多个类目改动（FILES_META 中同一 path 多条且带 match 正则），
    # 则按行归类、每张卡片只展示属于本类目的增删行（见下方 split_cache）。
    full_diff_text = run(["git", "diff", BASE, HEAD])
    full_diffs = parse_diffs(full_diff_text)
    full_numstat = run(["git", "diff", "--numstat", BASE, HEAD])
    full_ns: dict[str, tuple[str, str]] = {}
    for ln in full_numstat.splitlines():
        cols = ln.split("\t")
        if len(cols) == 3:
            full_ns[cols[2]] = (cols[0], cols[1])

    # 需要「按行拆分到多个类目」的文件：path -> [(cat, [compiled regex]), ...]
    path_match_entries: dict[str, list] = {}
    for item in FILES_META:
        if len(item) > 4 and item[4]:
            path_match_entries.setdefault(item[0], []).append(
                (item[1], [re.compile(p) for p in item[4]])
            )
    # 预解析这些文件的 hunks 与每个改动行的类目归属（含未归类告警）
    split_cache: dict[str, tuple] = {}
    for path, entries in path_match_entries.items():
        header, hunks = parse_hunks(full_diffs.get(path, ""))
        owners, unassigned = assign_owners(hunks, entries)
        if unassigned:
            print(
                f"⚠ {path}: {len(unassigned)} 个改动行未能归类，请补充 FILES_META 的 match 正则：",
                file=sys.stderr,
            )
            for u in unassigned:
                print(f"    {u}", file=sys.stderr)
        split_cache[path] = (header, hunks, owners)

    # 唯一文件路径（去重，保持出现顺序）及其所属类别列表 / 首张卡片锚点
    unique_paths: list[str] = []
    path_cats: dict[str, list[str]] = {}
    path_anchor: dict[str, str] = {}
    for item in FILES_META:
        path, cat = item[0], item[1]
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
    for item in FILES_META:
        cat_counts[item[1]] += 1

    # 类目短标题映射（用于折叠行标注「省略 N 行属于：xxx」）
    cat_title_map = {cid: CATEGORIES[cid][0] for cid in CATEGORIES}

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
        for item in FILES_META:
            path, cat, desc, is_new = item[0], item[1], item[2], item[3]
            if cat != cid:
                continue
            match = item[4] if len(item) > 4 else None
            title, color = CATEGORIES[cid]
            # 展示该文件相对 upstream/main 的当前差异；多类目文件只展示本类目的增删行
            if match and path in split_cache:
                header, hunks, owners = split_cache[path]
                ai, di = owner_counts(hunks, owners, cid)
                a, d = str(ai), str(di)
                diff_html = render_split_diff(header, hunks, owners, cid, cat_title_map)
            else:
                a, d = full_ns.get(path, ("-", "-"))
                diff_html = render_diff(full_diffs.get(path, "（该文件相对上游无差异 / 已收敛）"))
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
    <div class="diff">{diff_html}</div>
  </details>
</section>'''
            )

    cards_html = "\n".join(cards)
    cat_btn_html = "\n".join(cat_buttons)
    pr_notes = []
    for pr in PRS:
        pr_num = pr["url"].rsplit("/", 1)[-1]
        n_files = sum(1 for it in FILES_META if it[1] == pr["category"])
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
.dl.fold {{ color:var(--muted); background:rgba(139,148,158,.08); font-style:italic; opacity:.85; }}
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
    本仓 <b>main</b> 分支直接追踪 <b>upstream/main</b>，仅保留少量本地补丁（依赖锁定、忽略项、差异追踪工具）。<br>
    本页每张卡片展示该文件 <b>相对 upstream/main 的当前差异</b>（<code>git diff {BASE} {HEAD}</code>）。<br>
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
