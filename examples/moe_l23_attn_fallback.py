# SPDX-License-Identifier: Apache-2.0
"""Weight-loaded layer 23 attention fallback on the completed independent replay inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import moe_l23_oracle as prediction
import moe_mxfp4_w_check as comparison
import moe_routing_check as reference


def prepare(args, output):
    comparison.validate_pair(args.reference, args.candidate)
    lines = Path(args.previous_report).read_text(encoding="utf-8").splitlines()
    path = next((line.removeprefix("原始产物=") for line in lines if line.startswith("原始产物=")), "")
    events = [json.loads(line.removeprefix("MOE1 ")) for line in lines if line.startswith("MOE1 ")]
    fixture = next((event for event in events if event.get("event") == "fixed_inputs"), {})
    completed = any(event.get("decision") == "L23_ORACLE_MEASURED" for event in events)
    if not path or not completed or fixture.get("input_set") != "independent_prose_v1":
        raise ValueError("Need the completed independent-text layer 23 comparison")
    run = reference.relative_new(path)
    body = json.dumps(json.loads((run / "inputs.json").read_text()), sort_keys=True)
    token_hash = hashlib.sha256(body.encode()).hexdigest()
    if token_hash != fixture.get("token_sha256"):
        raise ValueError("Archived token manifest does not match the completed comparison")
    (output / "inputs.json").write_text(body + "\n", encoding="utf-8", newline="\n")
    (output / "baseline-run.txt").write_text(str(run), encoding="utf-8", newline="\n")
    reference.emit("fallback_inputs", baseline_run=str(run), token_sha256=token_hash)


def collect(args, output, *, attention_band=None):
    side = output / (attention_band or "fallback")
    side.mkdir(exist_ok=False)
    reference.collect(
        SimpleNamespace(
            phase="probe",
            model=args.candidate,
            source=args.source,
            inputs=str(output / "inputs.json"),
            output=str(side),
        ),
        probe_architecture="Qwen3MoeL23OracleForCausalLM",
        l23_attention_source=Path(args.reference).resolve(),
        attention_band=attention_band,
    )


def analyze(output):
    import torch

    from vllm_ascend.observability.moe_routing import ORACLE_LAYER, routing_metrics, tensor_error

    run = reference.relative_new((output / "baseline-run.txt").read_text())
    ref, candidate = (comparison.load_artifacts(run, side) for side in ("reference", "candidate"))
    fallback = comparison.load_artifacts(output, "fallback")
    inputs = json.loads((output / "inputs.json").read_text())
    if not all(data[2] for data in (ref, candidate, fallback)):
        raise ValueError("Missing request-level graph replay evidence")
    if not torch.equal(candidate[0][:, :, :ORACLE_LAYER], fallback[0][:, :, :ORACLE_LAYER]):
        raise ValueError("Candidate routes changed before the fallback layer")

    predictions = []
    paired = zip(inputs, ref[1], candidate[1], fallback[1], strict=True)
    for sample_id, (sample, target, baseline, restored) in enumerate(paired):
        config = dict(restored["config"])
        loaded = config.pop("attention_fallback_loaded", ())
        changed = config.pop("attention_fallback_changed", ())
        if config != baseline["config"] or config != target["config"] or len(loaded) != 4 or not changed:
            raise ValueError("Missing four attention replacements or different runtime settings")
        ref_row, base_row, fall_row = (s["layers"][ORACLE_LAYER] for s in (target, baseline, restored))
        native = routing_metrics(
            fall_row["eager_ids"], fall_row["ids"], fall_row["eager_weights"], fall_row["weights"], reference.EXPERTS
        )
        if native["set_equal_rate"] != 1 or native["weight_l1_mean"] != 0:
            raise ValueError("Fallback did not use its native router outputs")
        for phase, span in (("prefill", slice(0, reference.PREFILL)), ("decode", slice(reference.PREFILL, None))):
            base_metrics, fall_metrics = (
                routing_metrics(
                    ref_row["ids"][span],
                    row["ids"][span],
                    ref_row["weights"][span],
                    row["weights"][span],
                    reference.EXPERTS,
                )
                for row in (base_row, fall_row)
            )
            reference.emit(
                "attention_fallback_routes",
                sample=sample_id,
                phase=phase,
                tokens=base_metrics["tokens"],
                candidate_top8_overlap=base_metrics["overlap_mean"],
                fallback_top8_overlap=fall_metrics["overlap_mean"],
                candidate_weight_l1=base_metrics["weight_l1_mean"],
                fallback_weight_l1=fall_metrics["weight_l1_mean"],
                candidate_hidden_rel_l2=tensor_error(base_row["hidden"][span], ref_row["hidden"][span])["relative_l2"],
                fallback_hidden_rel_l2=tensor_error(fall_row["hidden"][span], ref_row["hidden"][span])["relative_l2"],
            )
        logits = [snapshot["continuation_logits"] for snapshot in (target, baseline, restored)]
        predictions.append(prediction.compare_predictions(*logits, sample["forced"]))
    prediction.report_predictions(torch.stack(predictions), treatment="fallback")
    reference.emit(
        "result", decision="L23_ATTENTION_FALLBACK_MEASURED", graph_replay="GREEN", task_quality="UNVERIFIED"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fallback", "analyze"), required=True)
    for name in ("reference", "candidate", "source", "output", "previous-report"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        output = reference.relative_new(args.output)
        if args.phase == "prepare":
            prepare(args, output)
        elif args.phase == "fallback":
            collect(args, output)
        else:
            analyze(output)
    except Exception as exc:
        reference.emit("error", phase=args.phase, exception_type=type(exc).__name__, reason=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
