# SPDX-License-Identifier: Apache-2.0
"""Whole-model comparison with reference expert IDs and scores at layer 23."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import moe_mxfp4_w_check as comparison
import moe_routing_check as reference

SIDES = ("reference", "candidate", "oracle")
FIXTURE = Path(__file__).with_name("moe_l23_independent.txt")


def prepare(args, output):
    from transformers import AutoTokenizer

    comparison.validate_pair(args.reference, args.candidate)
    tokenizer = AutoTokenizer.from_pretrained(args.reference, local_files_only=True, trust_remote_code=False)
    passages = FIXTURE.read_text(encoding="utf-8").split("\n---\n")
    if len(passages) != 4:
        raise ValueError("Expected four independent input passages")
    samples = []
    for sample_id, text in enumerate(passages):
        tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
        prompt = tokens[: reference.PREFILL]
        forced = tokens[reference.PREFILL : reference.PREFILL + reference.DECODE + 1]
        if len(forced) != reference.DECODE + 1 or any(
            tokens[start : start + len(forced)] == forced for start in range(reference.PREFILL)
        ):
            raise ValueError(f"Missing or repeated fixed continuation for sample {sample_id}")
        samples.append({"prompt": prompt, "forced": forced})
    body = json.dumps(samples, sort_keys=True)
    (output / "inputs.json").write_text(body + "\n", encoding="utf-8", newline="\n")
    reference.emit(
        "fixed_inputs",
        requests=len(samples),
        input_set="independent_prose_v1",
        token_sha256=hashlib.sha256(body.encode()).hexdigest(),
    )


def collect(args, output):
    side = output / args.phase
    side.mkdir(exist_ok=False)
    reference.collect(
        SimpleNamespace(
            phase="probe",
            model=args.reference if args.phase == "reference" else args.candidate,
            source=args.source,
            inputs=str(output / "inputs.json"),
            output=str(side),
        ),
        probe_architecture="Qwen3MoeL23OracleForCausalLM",
        reference_routes=output / "reference" if args.phase == "oracle" else None,
    )


def compare_predictions(target, baseline, oracle, forced_ids):
    import torch

    logits = (target, baseline, oracle)
    if any(t.shape != target.shape or t.ndim != 2 or not torch.isfinite(t).all() for t in logits):
        raise ValueError("Unaligned or nonfinite raw continuation logits")
    if len(forced_ids) != target.shape[0]:
        raise ValueError("Forced token and raw logit row counts differ")
    scores = [torch.log_softmax(t, dim=-1, dtype=torch.float64) for t in logits]
    target_prob = scores[0].exp()
    ids = torch.tensor(forced_ids, dtype=torch.long).unsqueeze(1)
    nll = [-t.gather(1, ids).squeeze(1) for t in scores]
    kl = [(target_prob * (scores[0] - t)).sum(dim=-1).clamp_min(0) for t in scores[1:]]
    return torch.stack((*nll, *kl), dim=0)


def analyze(output):
    import torch

    from vllm_ascend.observability.moe_routing import ORACLE_LAYER, selection_metrics

    inputs = json.loads((output / "inputs.json").read_text())
    ref, candidate, oracle = (comparison.load_artifacts(output, side) for side in SIDES)
    if not all(data[2] for data in (ref, candidate, oracle)):
        raise ValueError("Missing request-level graph replay evidence")
    if not torch.equal(candidate[0][:, :, :ORACLE_LAYER], oracle[0][:, :, :ORACLE_LAYER]):
        raise ValueError("Candidate routes changed before the intervention layer")

    predictions = []
    paired = zip(inputs, ref[1], candidate[1], oracle[1], strict=True)
    for sample_id, (sample, target, baseline, restored) in enumerate(paired):
        if target["config"] != baseline["config"] or baseline["config"] != restored["config"]:
            raise ValueError("Routing runtime settings differ between treatment arms")
        ref_row = target["layers"][ORACLE_LAYER]
        baseline_row = baseline["layers"][ORACLE_LAYER]
        oracle_row = restored["layers"][ORACLE_LAYER]
        if any(not torch.equal(ref_row[key], oracle_row[key]) for key in ("ids", "weights")):
            raise ValueError("Layer 23 did not dispatch the reference IDs and weights")
        if any(not torch.equal(baseline_row[key], oracle_row[key]) for key in ("hidden", "logits")):
            raise ValueError("Layer 23 inputs changed before routing")
        for phase, span in (("prefill", slice(0, reference.PREFILL)), ("decode", slice(reference.PREFILL, None))):
            native = selection_metrics(ref_row["ids"][span], baseline_row["ids"][span], reference.EXPERTS)
            reference.emit(
                "layer23_routes",
                sample=sample_id,
                phase=phase,
                tokens=native["tokens"],
                candidate_overlap=native["overlap_mean"],
                oracle_ids_and_weights="IDENTICAL_TO_REFERENCE",
            )
        logits = [snapshot["continuation_logits"] for snapshot in (target, baseline, restored)]
        predictions.append(compare_predictions(*logits, sample["forced"]))

    report_predictions(torch.stack(predictions), treatment="oracle")
    reference.emit("result", decision="L23_ORACLE_MEASURED", graph_replay="GREEN", task_quality="UNVERIFIED")


def report_predictions(scores, *, treatment):
    for label, span in (("end_of_prefill", slice(0, 1)), ("decode", slice(1, None))):
        reference_nll, candidate_nll, treated_nll, candidate_kl, treated_kl = scores[:, :, span].transpose(0, 1)
        reference.emit(
            "predictions",
            phase=label,
            tokens=candidate_nll.numel(),
            reference_nll=reference_nll.mean().item(),
            candidate_nll=candidate_nll.mean().item(),
            delta_nll=(treated_nll - candidate_nll).mean().item(),
            candidate_kl_to_reference=candidate_kl.mean().item(),
            delta_kl=(treated_kl - candidate_kl).mean().item(),
            lower_nll_tokens=int((treated_nll < candidate_nll).sum()),
            delta_nll_by_request=(treated_nll - candidate_nll).mean(dim=1).tolist(),
            delta_kl_by_request=(treated_kl - candidate_kl).mean(dim=1).tolist(),
            **{
                f"{treatment}_nll": treated_nll.mean().item(),
                f"{treatment}_kl_to_reference": treated_kl.mean().item(),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", *SIDES, "analyze"), required=True)
    for name in ("reference", "candidate", "source", "output"):
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
