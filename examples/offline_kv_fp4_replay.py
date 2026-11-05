# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-2 offline CPU replay of a bounded single-layer Qwen GQA snapshot.

No checkpoint, NPU initialization, server hooks, prompt/answer output or network.
Real snapshot capture is a separate integration; synthetic metadata is never
reported as model evidence. E4M3 scales are explicit, not fitted to test inputs.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

CONTRACT = "qwen3-causal-gqa-post-rope-v1"
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_QUERY_ROWS = 128
SNAPSHOT_KEYS = ("q", "k", "v", "query_positions")


def numerical_modules():
    # This CPU-only process must not auto-load device plugins with torch.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # These standalone tools must not import vllm_ascend.__init__/torch_npu.
    package = types.ModuleType("_kv_replay_tools")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "vllm_ascend/quantization")]
    sys.modules[package.__name__] = package
    return (
        importlib.import_module(package.__name__ + ".kv_fp4"),
        importlib.import_module(package.__name__ + ".kv_replay"),
    )


def source_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if Path(top).resolve() != root:
            return "UNKNOWN"
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def load_snapshot(path):
    from safetensors import safe_open

    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Missing snapshot or file exceeds the 512 MiB diagnostic limit")
    with safe_open(path, framework="pt", device="cpu") as f:
        if set(f.keys()) != set(SNAPSHOT_KEYS):
            raise ValueError("Expected only q/k/v/query_positions, not a model checkpoint")
        metadata = f.metadata() or {}
        if metadata.get("contract") != CONTRACT or metadata.get("source_kind") not in ("captured", "synthetic"):
            raise ValueError("Missing or unsupported attention/provenance contract")
        layer = metadata.get("layer", "")
        if not layer.isdecimal() or not 0 <= int(layer) < 128:
            raise ValueError("Expected a layer index in [0,128)")
        q_shape = f.get_slice("q").get_shape()
        k_shape = f.get_slice("k").get_shape()
        v_shape = f.get_slice("v").get_shape()
        if (
            len(q_shape) != 3
            or len(k_shape) != 3
            or v_shape != k_shape
            or not 0 < q_shape[0] <= MAX_QUERY_ROWS
            or not 0 < k_shape[0] <= 40960
            or not 0 < q_shape[1] <= 128
            or not 0 < k_shape[1] <= 128
            or not 0 < q_shape[2] <= 512
            or k_shape[2] != q_shape[2]
        ):
            raise ValueError("Snapshot dimensions exceed the single-sequence diagnostic contract")
        return (
            {key: f.get_tensor(key) for key in SNAPSHOT_KEYS},
            {"source_kind": metadata["source_kind"], "layer": int(layer)},
        )


def output_path(value, source):
    path = Path(value)
    root = Path.cwd().resolve()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Report path must be relative to the execution directory")
    path = root / path
    if not path.resolve().is_relative_to(root) or path.resolve() == source.resolve():
        raise ValueError("Report escapes the execution directory or overlaps the input")
    return path


def publish_report(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace, never follow a pre-existing output link into a tensor/weight file.
    fd, temporary = tempfile.mkstemp(prefix=".kv-replay-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="single-layer safetensors snapshot, not model weights"
    )
    parser.add_argument(
        "--k-outer-scale", type=float, required=True, help="fixed external E4M3 K scale, not fitted here"
    )
    parser.add_argument(
        "--v-outer-scale", type=float, required=True, help="fixed external E4M3 V scale, not fitted here"
    )
    parser.add_argument("--output", default="outputs/kv-replay.txt")
    args = parser.parse_args(argv)
    try:
        target = output_path(args.output, args.input)
        codec, replay = numerical_modules()
        tensors, provenance = load_snapshot(args.input)
        q, k, v, positions = (tensors[key] for key in SNAPSHOT_KEYS)
        # Validate external scales before performing any candidate replay.
        codec.FP4Recipe(16, "e4m3", args.k_outer_scale)
        codec.FP4Recipe(16, "e4m3", args.v_outer_scale)
        lines = [
            "KV-2 CPU offline numerical replay; NOT native attention, full-model quality or performance acceptance",
            f"source_kind={provenance['source_kind']} (caller-declared); "
            f"layer={provenance['layer']}; contract={CONTRACT}",
            f"source_sha={source_revision()}; torch={sys.modules['torch'].__version__}; "
            "input_provenance=caller_declared",
            f"Q={tuple(q.shape)} K/V={tuple(k.shape)} dtype={q.dtype}",
            f"fixed external outer_scale: K={args.k_outer_scale} V={args.v_outer_scale}; no fitting performed",
        ]
        for name, group, kind in (
            ("G16-B", 16, "bf16"),
            ("G32-B", 32, "bf16"),
            ("G16-E", 16, "e4m3"),
            ("G32-E", 32, "e4m3"),
            ("MX4", 32, "e8m0"),
        ):
            k_recipe = codec.FP4Recipe(group, kind, args.k_outer_scale if kind == "e4m3" else None)
            v_recipe = codec.FP4Recipe(group, kind, args.v_outer_scale if kind == "e4m3" else None)
            rows = replay.compare_kv(q, k, v, positions, k_recipe, v_recipe)
            for row in rows:
                summary = {
                    "candidate": name,
                    "mode": row["mode"],
                    "k": row["k"],
                    "v": row["v"],
                    "attention": row["attention"],
                }
                summary["worst_attention_heads"] = sorted(
                    ({"head": h, **metrics} for h, metrics in enumerate(row["attention_heads"])),
                    key=lambda m: m["rmse"],
                    reverse=True,
                )[:3]
                lines.append(json.dumps(summary, allow_nan=False, sort_keys=True))
        lines.append("COMPLETED: 15 local interventions; NO quality pass/fail threshold or deployment claim")
        text = "\n".join(lines) + "\n"
        publish_report(target, text)
        print(text, end="")
        print(f"Report: {target.relative_to(Path.cwd().resolve())}")
        return 0
    except Exception as exc:
        # Includes safetensors parse errors; never turn a corrupt input into a pass.
        message = str(exc).replace("\n", " ")[:500]
        print(f"RED: replay failed ({type(exc).__name__}): {message}", file=sys.stderr)
        print("No new report published; an existing report belongs to an earlier run.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
