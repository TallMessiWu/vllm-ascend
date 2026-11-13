# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Weight-free K+V scale diagnostics. Synthetic arithmetic, NOT model acceptance."""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Standalone CPU diagnostics must not auto-load the installed NPU plugin.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch  # noqa: E402


def load_codec():
    source = Path(__file__).resolve().parents[1] / "vllm_ascend/quantization/kv_fp4.py"
    spec = importlib.util.spec_from_file_location("_kv_numerics_codec", source)
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    return codec


def critical_channel_case():
    # Constructed counterexample, not a claimed distribution from any model.
    k = torch.tensor([[7.5] + [0.375] * 31, [7.25] * 32], dtype=torch.bfloat16)[:, None]
    v = torch.zeros_like(k)
    v[0] = 1
    q = torch.zeros(1, 2, 32, dtype=k.dtype)
    q[..., 0] = 4
    return {"name": "query-critical-channel", "q": q, "k": k, "v": v, "positions": torch.tensor([1])}


def attention(q, k, v, positions):
    """CPU FP64 causal GQA math on fixed BF16 inputs; no native attention kernel."""
    if any(x.ndim != 3 or x.numel() == 0 or x.device.type != "cpu" for x in (q, k, v)):
        raise ValueError("Expected nonempty CPU Q[S,Hq,D], K/V[T,Hkv,D]")
    if k.shape != v.shape or q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid GQA dimensions")
    if any(x.dtype != torch.bfloat16 or not torch.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Expected finite BF16 inputs")
    if positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.shape != (q.shape[0],):
        raise ValueError("Expected CPU int64 absolute query positions")
    if not ((positions >= 0) & (positions < k.shape[0])).all():
        raise ValueError("Positions outside the complete [0,T) prefix")
    repeats = q.shape[1] // k.shape[1]
    scores = torch.einsum("shd,thd->sht", q.double(), k.double().repeat_interleave(repeats, dim=1))
    scores = scores / math.sqrt(q.shape[-1])
    valid = (torch.arange(k.shape[0])[None, None] <= positions[:, None, None]).expand_as(scores)
    log_probs = torch.log_softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
    probabilities = log_probs.exp()
    output = torch.einsum("sht,thd->shd", probabilities, v.double().repeat_interleave(repeats, dim=1))
    return scores, log_probs, output, valid


def observed_qdq(codec, x, recipe, shift=None):
    stages = {}

    def observe(name, value):
        if name in ("scale", "code"):
            stages[name] = value

    result = codec.qdq_fp4(x, recipe, observer=observe, exponent_shift=shift)
    return result, stages


def recipes(codec):
    return (
        ("g32-b", codec.FP4Recipe(32, "bf16")),
        ("mx4-ceil", codec.FP4Recipe(32, "e8m0")),
        ("mx4-mse", codec.FP4Recipe(32, "e8m0", scale_selection="mse")),
    )


def kv_stats(raw, decoded, scales, ceil_scales, group):
    expanded = scales.repeat_interleave(group, dim=-1)[..., : raw.shape[-1]]
    return {
        "mse": (decoded.double() - raw.double()).square().mean().item(),
        "clip_pct": 100 * (raw.float().abs() > 6 * expanded).double().mean().item(),
        "new_zero_pct": 100 * ((raw != 0) & (decoded == 0)).double().mean().item(),
        "scale_changed_pct": 100 * (scales != ceil_scales).double().mean().item() if ceil_scales is not None else None,
    }


def evaluate_case(codec, case):
    q, k, v, positions = (case[key] for key in ("q", "k", "v", "positions"))
    scores, log_probs, output, valid = attention(q, k, v, positions)
    ceil = codec.FP4Recipe(32, "e8m0")
    ceil_scales = [observed_qdq(codec, x, ceil)[1]["scale"] for x in (k, v)]
    rows = []
    for name, recipe in recipes(codec):
        # Always quantize BOTH K and V; Q, masks and reference arithmetic are fixed.
        k_low, k_stages = observed_qdq(codec, k, recipe)
        v_low, v_stages = observed_qdq(codec, v, recipe)
        low_scores, low_log_probs, low_output, _ = attention(q, k_low, v_low, positions)
        error = low_output - output
        norm = torch.linalg.vector_norm(output).item()
        row = {"format": name}
        for side, raw, low, stages, scale in zip(("k", "v"), (k, v), (k_low, v_low), (k_stages, v_stages), ceil_scales):
            row.update(
                {
                    f"{side}_{key}": value
                    for key, value in kv_stats(
                        raw, low, stages["scale"], scale if recipe.scale_type == "e8m0" else None, recipe.group_size
                    ).items()
                }
            )
        log_delta = torch.where(valid, log_probs - low_log_probs, 0.0)
        row.update(
            {
                "qk_rmse": (low_scores[valid] - scores[valid]).square().mean().sqrt().item(),
                "attention_kl": (log_probs.exp() * log_delta).sum(-1).mean().clamp_min(0).item(),
                "out_rmse": error.square().mean().sqrt().item(),
                "out_relative_l2": torch.linalg.vector_norm(error).item() / norm if norm else None,
                "worst_head_out_rmse": error.square().mean(dim=(0, 2)).sqrt().max().item(),
            }
        )
        rows.append(row)
    return rows


def cases():
    generator = torch.Generator().manual_seed(20261113)
    result = [critical_channel_case()]
    for name, tokens, width in (
        ("zero-tail", 17, 37),
        ("gaussian", 65, 128),
        ("channel-outliers", 65, 128),
        ("scale-boundary", 65, 128),
    ):
        q = torch.randn(3, 4, width, generator=generator).bfloat16()
        k = torch.randn(tokens, 2, width, generator=generator).bfloat16()
        v = torch.randn(tokens, 2, width, generator=generator).bfloat16()
        if name == "zero-tail":
            k.zero_()
            v.zero_()
        elif name == "channel-outliers":
            k[..., 0] *= 8
            v[..., 7] *= 8
        elif name == "scale-boundary":
            k.fill_(0.375)
            k[::2, :, ::32] = 6.0
            k[1::2, :, ::32] = 6.03125
        result.append({"name": name, "q": q, "k": k, "v": v, "positions": torch.tensor([0, tokens // 2, tokens - 1])})
    return result


def check_contracts(codec):
    """Independent literal codec anchors plus an analytic two-key softmax oracle."""
    raw = torch.tensor([[6.03125] + [0.375] * 31, [9.0] * 32], dtype=torch.bfloat16)
    for selection, scales, decoded in (
        ("ceil", [[128], [128]], [[6.0] + [0.0] * 31, [8.0] * 32]),
        ("mse", [[127], [128]], [[6.0] + [0.5] * 31, [8.0] * 32]),
    ):
        recipe = codec.FP4Recipe(32, "e8m0", scale_selection=selection)
        packed = codec.pack_fp4(raw, recipe)
        expected = torch.tensor(decoded, dtype=raw.dtype)
        if packed.scales.tolist() != scales or not torch.equal(codec.qdq_fp4(raw, recipe), expected):
            raise ValueError("Literal MX4 scale/code contract failed")
        if not torch.equal(codec.unpack_fp4(packed, raw.dtype), expected):
            raise ValueError("Literal MX4 packed round-trip failed")
    case = critical_channel_case()
    _, _, output, _ = attention(case["q"], case["k"], case["v"], case["positions"])
    expected = 1 / (1 + math.exp(-0.25 * 4 / math.sqrt(32)))
    if not torch.allclose(output, torch.full_like(output, expected), atol=1e-12, rtol=1e-12):
        raise ValueError("Analytic attention oracle failed")


def payload_mismatches(actual, expected):
    """Bitwise payload checks, including signed zero. Called only outside capture."""
    counts = {}
    for key in ("qdq", "scale", "code"):
        a, b = actual[key].cpu().contiguous(), expected[key].cpu().contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Mismatched numerical payload metadata")
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        counts[key] = int((a.view(dtype) != b.view(dtype)).sum())
    return counts


def prepare_npu():
    selected = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not selected.isdecimal() or str(int(selected)) != selected:
        raise ValueError("NPU mode requires one explicit ASCEND_RT_VISIBLE_DEVICES device")
    # Lazy import: CPU mode never initializes/imports the NPU plugin.
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    print(f"NPU={torch.npu.get_device_name(0)} torch_npu={getattr(torch_npu, '__version__', 'UNKNOWN')}")


@torch.inference_mode()
def npu_parity(codec, host, recipe):
    """Raw codec ACL replay, NOT a compiled model/native attention validation."""
    x = torch.zeros_like(host, device="npu")
    shift = torch.tensor([23], dtype=torch.int32, device="cpu").to(device="npu", non_blocking=False)

    def payload(value, operand=None):
        output, stages = observed_qdq(codec, value, recipe, operand)
        return {"qdq": output, **stages}

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            payload(x, shift)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    # Retain x/shift and all observed stage tensors for every replay. These
    # retained intermediates change lifetimes versus an uninstrumented model.
    with torch.npu.graph(graph, stream=stream):
        replay = payload(x, shift)
    counts = {
        name: dict.fromkeys(("qdq", "scale", "code"), 0)
        for name in ("eager_vs_cpu", "replay_vs_eager", "replay_vs_cpu")
    }
    phases = (1.0, 2.3, 0.0, -0.7, 1.0)
    for factor in phases:
        changed = (host.float() * factor).bfloat16()
        x.copy_(changed)
        torch.npu.synchronize()
        eager = payload(x, shift)
        graph.replay()
        torch.npu.synchronize()
        expected = payload(changed)
        for name, actual, reference in (
            ("eager_vs_cpu", eager, expected),
            ("replay_vs_eager", replay, eager),
            ("replay_vs_cpu", replay, expected),
        ):
            for stage, count in payload_mismatches(actual, reference).items():
                counts[name][stage] += count
    return {"replays": len(phases), **counts}


def source_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    args = parser.parse_args(argv)
    try:
        torch.set_num_threads(1)
        codec = load_codec()
        print(f"KV-NUMERICS source_kind=synthetic mode={args.device} seed=20261113; no weights/server/data")
        print(f"source_sha={source_revision()} codec={codec.__file__} torch={torch.__version__}")
        print("Attention reference: CPU FP64 causal GQA, fixed BF16 Q, BOTH K+V quantized; not native attention")
        print("Metrics: clip=abs(input)>6*scale; new_zero=nonzero input->0, percentages over valid elements")
        print("scale_changed=group percentage versus ceil (MX4 only); null=undefined/not applicable")
        if args.device == "npu":
            prepare_npu()
        else:
            print("NPU=UNVERIFIED (CPU mode does not import torch_npu or claim device validation)")
        check_contracts(codec)
        failures, comparisons, evaluated = 0, 0, 0
        regressions = []
        for case in cases():
            print(f"CASE {case['name']} Q={tuple(case['q'].shape)} KV={tuple(case['k'].shape)}")
            rows = evaluate_case(codec, case)
            if len(rows) != len(recipes(codec)):
                raise ValueError("Missing candidate measurements")
            evaluated += len(rows)
            by_name = {row["format"]: row for row in rows}
            ceil, mse = by_name["mx4-ceil"], by_name["mx4-mse"]
            if mse["k_mse"] + mse["v_mse"] < ceil["k_mse"] + ceil["v_mse"] and mse["out_rmse"] > ceil["out_rmse"]:
                regressions.append(case["name"])
            for row, (_, recipe) in zip(rows, recipes(codec)):
                print(json.dumps(row, sort_keys=True, allow_nan=False))
                if args.device == "npu":
                    parity = npu_parity(codec, torch.stack((case["k"], case["v"])), recipe)
                    print(f"PARITY {case['name']} {row['format']} {json.dumps(parity, sort_keys=True)}")
                    comparisons += parity["replays"]
                    failures += sum(sum(value.values()) for key, value in parity.items() if key != "replays")
        if evaluated == 0:
            raise ValueError("No numerical cases evaluated")
        print(f"MEASURED={evaluated} synthetic K+V candidates; lower_KV_MSE_but_worse_attention={regressions}")
        if failures or (args.device == "npu" and comparisons == 0):
            print(f"ARITHMETIC=RED payload_mismatches={failures} replays={comparisons}")
            return 1
        print(f"ARITHMETIC=GREEN scope=CPU_literal_oracles{'+NPU_eager_raw_ACL' if args.device == 'npu' else ''}")
        print("MODEL_QUALITY=UNVERIFIED; real KV distribution/native attention/model graphs/performance untested")
        print("DECISION=HOLD_E2E: review attention errors, not KV MSE alone; synthetic checks cannot approve a model")
        return 0
    except Exception as exc:
        print(f"ARITHMETIC=RED {type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
