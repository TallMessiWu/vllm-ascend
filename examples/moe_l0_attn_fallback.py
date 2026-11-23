# SPDX-License-Identifier: Apache-2.0
"""Fixed-token comparison of R-M16 with and without a layer 0 attention fallback."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import moe_mxfp4_w_check as comparison
import moe_routing_check as reference

SIDES = ("baseline", "fallback")


def prepare(args, output):
    comparison.validate_pair(args.reference, args.candidate)
    report = Path(args.moe1_report).read_text(encoding="utf-8").splitlines()
    path = next((line for line in report if line.startswith("原始产物=")), "")
    if not path or not any(line.startswith("COMPLETED:") for line in report):
        raise ValueError("This comparison needs the completed MoE-1 snapshots")
    run = reference.relative_new(path.removeprefix("原始产物=").split("；", 1)[0])
    inputs = (run / "inputs.json").read_text()
    token_hash = hashlib.sha256(json.dumps(json.loads(inputs), sort_keys=True).encode()).hexdigest()
    for side, model in (("reference", args.reference), ("candidate", args.candidate)):
        receipt = json.loads((run / side / "collection.json").read_text())
        if receipt != {"side": side, "model": str(Path(model).resolve()), "token_sha256": token_hash}:
            raise ValueError(f"MoE-1 snapshot identity mismatch for {side}")
    (output / "inputs.json").write_text(inputs, encoding="utf-8", newline="\n")
    (output / "moe1-run.txt").write_text(str(run), encoding="utf-8", newline="\n")
    reference.emit("fallback_inputs", source_run=str(run), token_sha256=token_hash)


def collect(args, output):
    side = output / args.phase
    side.mkdir(exist_ok=False)
    scores = reference.collect(
        SimpleNamespace(
            phase="probe",
            model=args.candidate,
            source=args.source,
            inputs=str(output / "inputs.json"),
            output=str(side),
        ),
        score_tokens=True,
        l0_attention_source=Path(args.reference).resolve() if args.phase == "fallback" else None,
    )
    (side / "continuation-nll.json").write_text(json.dumps(scores, allow_nan=False), encoding="utf-8", newline="\n")


def analyze(output):
    import torch

    from vllm_ascend.observability.moe_routing import selection_metrics, tensor_error

    run = reference.relative_new((output / "moe1-run.txt").read_text())
    ref = comparison.load_artifacts(run, "reference")
    baseline, fallback = (comparison.load_artifacts(output, side) for side in SIDES)
    if not all(data[2] for data in (ref, baseline, fallback)):
        raise ValueError("Missing request-level graph replay evidence")
    for target, before, after in zip(ref[1], baseline[1], fallback[1], strict=True):
        config = dict(after["config"])
        loaded = config.pop("l0_attention_loaded", ())
        changed = config.pop("l0_attention_changed", ())
        if config != before["config"] or config != target["config"] or len(loaded) != 4 or not changed:
            raise ValueError("Missing layer 0 weight replacement or different runtime settings")

    for layer in (0, 23):
        for phase, span in (("prefill", slice(0, reference.PREFILL)), ("decode", slice(reference.PREFILL, None))):
            tokens = [data[0][:, span, layer, :].flatten(0, 1) for data in (ref, baseline, fallback)]
            hidden = [
                torch.cat([s["layers"][layer]["hidden"][span] for s in data[1]]) for data in (ref, baseline, fallback)
            ]
            reference.emit(
                "fallback_routing",
                layer=layer,
                phase=phase,
                baseline=selection_metrics(tokens[0], tokens[1], reference.EXPERTS),
                fallback=selection_metrics(tokens[0], tokens[2], reference.EXPERTS),
                baseline_hidden_rel_l2=tensor_error(hidden[1], hidden[0])["relative_l2"],
                fallback_hidden_rel_l2=tensor_error(hidden[2], hidden[0])["relative_l2"],
            )

    nll = []
    for side in SIDES:
        scores = json.loads((output / side / "continuation-nll.json").read_text())
        values = torch.tensor(scores, dtype=torch.float64)
        if values.shape != (4, reference.DECODE + 1) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Incomplete or invalid fixed-token scores")
        nll.append(values)
    for phase, span in (("end_of_prefill", slice(0, 1)), ("decode", slice(1, None))):
        before, after = (values[:, span] for values in nll)
        reference.emit(
            "fallback_nll",
            phase=phase,
            tokens=before.numel(),
            baseline_nll=before.mean().item(),
            fallback_nll=after.mean().item(),
            delta_nll=(after - before).mean().item(),
            lower_nll_tokens=int((after < before).sum()),
        )
    reference.emit("result", decision="L0_ATTENTION_FALLBACK_MEASURED", graph_replay="GREEN", task_quality="UNVERIFIED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", *SIDES, "analyze"), required=True)
    for name in ("reference", "candidate", "source", "output", "moe1-report"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        output = reference.relative_new(args.output)
        if args.phase == "prepare":
            prepare(args, output)
        elif args.phase == "analyze":
            analyze(output)
        else:
            collect(args, output)
    except Exception as exc:
        reference.emit("error", phase=args.phase, exception_type=type(exc).__name__, reason=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
