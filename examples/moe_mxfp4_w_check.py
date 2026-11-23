# SPDX-License-Identifier: Apache-2.0
"""Fixed-token routing comparison: source BF16-DQ vs MXFP4-W BF16 carrier.

Both sides reuse the validated probe, graph buffers, named RPCs and native V2
continuation. No model-runner patch, runtime QDQ, A/KV change or recovery oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import moe_routing_check as reference

SIDES = ("reference", "candidate", "candidate-repeat")


def emit(event, **fields):
    reference.emit(event, **fields)


def validate_pair(model, candidate):
    """Validate conversion identity without claiming to rehash weight payloads."""
    source, target = Path(model).resolve(), Path(candidate).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Reference and candidate must be independent directories")
    configs = [reference.validate_model(str(path)) for path in (source, target)]
    if configs[0] != configs[1]:
        raise ValueError("Only W values may differ; model configs must match")
    if (target / ".mxfp4_w_to_bf16.incomplete").exists():
        raise ValueError("Candidate conversion is incomplete")
    manifest = json.loads((target / ".mxfp4_w_metadata/manifest.json").read_text())
    recipe = manifest.get("recipe", {})
    expected = {
        "format": "MXFP4_E2M1_E8M0",
        "group_size": 32,
        "axis": -1,
        "scale_selection": "floor_log2_amax_minus_2",
        "rounding": "RNE",
        "carrier": "BF16",
        "activations": "BF16",
        "kv": "BF16",
        "router": "unchanged_BF16",
        "scale_exponent_range": [-127, 127],
        "flush_fp32_subnorms": False,
        "reference_sha": "7bc41952de394f5cc5e782baf132e7c7542eb4e4",
    }
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("candidate") != "R-M16"
        or recipe != expected
        or manifest.get("statistics", {}).get("changed_values", 0) <= 0
    ):
        raise ValueError("Wrong candidate identity/recipe or no actual quantization")
    for key, directory in (("source", source), ("output", target)):
        recorded = manifest.get(key)
        if not isinstance(recorded, str) or not Path(recorded).is_absolute() or Path(recorded).resolve() != directory:
            raise ValueError("Reference/candidate directory does not match conversion manifest")
    for name, key in (("config.json", "source_config_sha256"), ("model.safetensors.index.json", "source_index_sha256")):
        for root in (source, target):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest.get(key):
                raise ValueError("Metadata no longer matches conversion manifest")
    for root, key in ((source, "source_shards"), (target, "output_shards")):
        signatures = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in root.glob("*.safetensors")}
        if not signatures or signatures != manifest.get(key):
            raise ValueError("Shard size/mtime changed after conversion")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        paths = [root / name for root in (source, target)]
        if paths[0].exists() != paths[1].exists() or (
            paths[0].exists() and paths[0].read_bytes() != paths[1].read_bytes()
        ):
            raise ValueError("Tokenizer/generation protocol differs")
    return manifest


def collect(args):
    folder = reference.relative_new(args.output) / args.phase
    folder.mkdir(exist_ok=False)
    emit("comparison_run", side=args.phase, weight_condition="R-N16" if args.phase == "reference" else "R-M16")
    inputs_before = Path(args.inputs).read_bytes()
    reference.collect(
        SimpleNamespace(
            phase="probe",
            model=args.model if args.phase == "reference" else args.candidate,
            source=args.source,
            inputs=args.inputs,
            output=str(folder),
        )
    )
    if Path(args.inputs).read_bytes() != inputs_before:
        raise ValueError("Input manifest changed during collection")
    receipt = {
        "token_sha256": hashlib.sha256(json.dumps(json.loads(inputs_before), sort_keys=True).encode()).hexdigest(),
        "side": args.phase,
        "model": str(Path(args.model if args.phase == "reference" else args.candidate).resolve()),
    }
    (folder / "collection.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    emit("collection_identity", side=args.phase, token_sha256=receipt["token_sha256"])


def load_artifacts(root, side):
    import torch

    folder = root / side
    native = torch.load(folder / "probe-0.pt", map_location="cpu", weights_only=True)
    expected = (4, reference.PREFILL + reference.DECODE, reference.LAYERS, reference.TOP_K)
    if tuple(native.shape) != expected:
        raise ValueError("Native routing shape mismatch")
    snapshots = []
    replay = True
    for sample in range(4):
        snapshot = torch.load(folder / f"probe-snapshot-{sample}.pt", map_location="cpu", weights_only=True)
        if len(snapshot["layers"]) != reference.LAYERS:
            raise ValueError("Missing layers")
        for layer, row in enumerate(snapshot["layers"]):
            if (
                not torch.equal(row["ids"].int(), native[sample, :, layer].int())
                or row["device_calls"] != reference.DECODE + 1
                or row["prefill_rows"] != reference.PREFILL
                or row["decode_rows"] != reference.DECODE
                or not 0 <= row["host_calls"] <= row["device_calls"]
            ):
                raise ValueError("Incomplete/misaligned snapshot or invalid call counts")
            if tuple(row["weights"].shape) != (expected[1], reference.TOP_K):
                raise ValueError("Invalid captured weight shape")
            replay &= row["device_calls"] - row["host_calls"] >= reference.DECODE
        snapshots.append(snapshot)
    return native, snapshots, replay


def boundary(logits, changed):
    ordered = logits.double().sort(dim=-1, descending=True).values
    margins = ordered[:, reference.TOP_K - 1] - ordered[:, reference.TOP_K]
    return {
        "minimum": margins.min().item(),
        "median": margins.median().item(),
        "exact_ties": int((margins == 0).sum()),
        "changed_tokens": int(changed.sum()),
        "changed_margin_median": margins[changed].median().item() if changed.any() else None,
        "unchanged_margin_median": margins[~changed].median().item() if (~changed).any() else None,
    }


def analyze(args):
    """Check collection identity/coverage before interpreting numerical drift.

    Candidate repeats and capture/eager agreement gate execution validity.
    Reference/candidate differences are measurements, never quality gates or a
    causal isolation of routing from changed expert computation. Detailed CPU
    math is reported separately from the stricter capture/eager comparisons.
    """
    import torch

    from vllm_ascend.observability.moe_routing import router_reference, routing_metrics, tensor_error

    root = reference.relative_new(args.output)
    expected_hash = hashlib.sha256(
        json.dumps(json.loads(Path(args.inputs).read_bytes()), sort_keys=True).encode()
    ).hexdigest()
    for side in SIDES:
        receipt = json.loads((root / side / "collection.json").read_text())
        model = args.model if side == "reference" else args.candidate
        if receipt != {"token_sha256": expected_hash, "side": side, "model": str(Path(model).resolve())}:
            raise ValueError("Collection input/model identity differs between sides")
    data = {side: load_artifacts(root, side) for side in SIDES}
    replay = all(value[2] for value in data.values())
    unstable, arithmetic_review, any_drift = False, False, False
    config = reference.validate_model(args.model)
    router_weights = {}
    for layer in reference.DETAIL_LAYERS:
        weight = reference.load_router_weight(args.model, layer)
        if not torch.equal(weight, reference.load_router_weight(args.candidate, layer)):
            raise ValueError("Router parameters changed; not a W-only candidate under this contract")
        router_weights[layer] = weight
    for sample in range(4):
        snapshots = {side: value[1][sample] for side, value in data.items()}
        if any(snapshot["config"] != snapshots["reference"]["config"] for snapshot in snapshots.values()):
            raise ValueError("Graph/routing runtime config differs between sides")
        emit("runtime_config", sample=sample, **snapshots["reference"]["config"])
        for stage, span in (("prefill", slice(0, reference.PREFILL)), ("decode", slice(reference.PREFILL, None))):
            first_changed = None
            for layer in range(reference.LAYERS):
                rows = {side: snapshot["layers"][layer] for side, snapshot in snapshots.items()}
                measured = routing_metrics(
                    rows["reference"]["ids"][span],
                    rows["candidate"]["ids"][span],
                    rows["reference"]["weights"][span],
                    rows["candidate"]["weights"][span],
                    reference.EXPERTS,
                )
                repeated = routing_metrics(
                    rows["candidate"]["ids"][span],
                    rows["candidate-repeat"]["ids"][span],
                    rows["candidate"]["weights"][span],
                    rows["candidate-repeat"]["weights"][span],
                    reference.EXPERTS,
                )
                unstable |= repeated["set_equal_rate"] != 1.0 or repeated["weight_l1_mean"] != 0.0
                changed = measured["set_equal_rate"] != 1.0
                any_drift |= changed
                if changed and first_changed is None:
                    first_changed = layer
                changed_mask = ~(
                    rows["reference"]["ids"][span, :, None] == rows["candidate"]["ids"][span, None, :]
                ).any(dim=-1).all(dim=-1)
                positions = changed_mask.nonzero().flatten()
                emit(
                    "weight_drift",
                    sample=sample,
                    stage=stage,
                    layer=layer,
                    reference_vs_candidate=measured,
                    candidate_repeat=repeated,
                    changed_positions=int(positions.numel()),
                    first_changed_positions=(positions[:8] + (reference.PREFILL if stage == "decode" else 0)).tolist(),
                )
                if layer not in reference.DETAIL_LAYERS:
                    continue
                for side, row in rows.items():
                    eager_error = tensor_error(row["logits"][span], row["eager_logits"][span])
                    eager_routes = routing_metrics(
                        row["ids"][span],
                        row["eager_ids"][span],
                        row["weights"][span],
                        row["eager_weights"][span],
                        reference.EXPERTS,
                    )
                    arithmetic_review |= (
                        not eager_error["exact"]
                        or eager_routes["set_equal_rate"] != 1.0
                        or eager_routes["weight_l1_mean"] != 0.0
                    )
                    cpu_logits, cpu_ids, cpu_weights, margin = router_reference(
                        row["hidden"][span], router_weights[layer], reference.TOP_K, bool(config["norm_topk_prob"])
                    )
                    if not torch.isfinite(row["output"][span]).all():
                        raise ValueError("Nonfinite MoE output")
                    emit(
                        "arithmetic",
                        side=side,
                        sample=sample,
                        stage=stage,
                        layer=layer,
                        capture_vs_npu_eager_logits=eager_error,
                        capture_vs_npu_eager_routes=eager_routes,
                        capture_vs_cpu_fp64_logits=tensor_error(row["logits"][span], cpu_logits),
                        capture_vs_cpu_fp64_routes=routing_metrics(
                            cpu_ids, row["ids"][span], cpu_weights, row["weights"][span], reference.EXPERTS
                        ),
                        cpu_fp64_boundary=margin,
                        parameter_dtype=row["parameter_dtype"],
                        cpu_oracle="mathematical_reference_not_kernel_equivalence",
                    )
                ref, cand = rows["reference"], rows["candidate"]
                emit(
                    "representative_drift",
                    sample=sample,
                    stage=stage,
                    layer=layer,
                    hidden=tensor_error(cand["hidden"][span], ref["hidden"][span]),
                    router_logits=tensor_error(cand["logits"][span], ref["logits"][span]),
                    total_moe_output=tensor_error(cand["output"][span], ref["output"][span]),
                    reference_boundary=boundary(ref["logits"][span], changed_mask),
                    candidate_boundary=boundary(cand["logits"][span], changed_mask),
                    causal_attribution="UNVERIFIED_requires_fixed_candidate_expert_oracles",
                )
            emit("stage_summary", sample=sample, stage=stage, first_changed_layer=first_changed)
    review = unstable or arithmetic_review or not replay
    emit(
        "result",
        capture="GREEN",
        graph_replay="GREEN" if replay else "UNVERIFIED",
        candidate_repeat="REVIEW" if unstable else "IDENTICAL",
        capture_vs_eager="REVIEW" if arithmetic_review else "IDENTICAL",
        quantization_drift="OBSERVED" if any_drift else "NO_SET_CHANGE_IN_THIS_FIXTURE",
        model_quality="UNVERIFIED",
        routing_recovery="NOT_TESTED",
        performance="NOT_TESTED",
        decision="REVIEW_EXECUTION" if review else "DRIFT_MEASURED",
    )
    return int(review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", *SIDES, "analyze"), required=True)
    for name in ("model", "candidate", "source", "inputs", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        manifest = validate_pair(args.model, args.candidate)
        reference.relative_new(args.output).mkdir(parents=True, exist_ok=True)
        if args.phase == "prepare":
            reference.make_inputs(args.model, args.inputs)
            emit(
                "candidate_identity",
                candidate="R-M16",
                recipe=manifest["recipe"],
                statistics=manifest["statistics"],
                quantized_tensors=manifest["quantized_tensors"],
                preserved_tensors=manifest["preserved_tensors"],
                sampled_groups=manifest["sampled_groups_readback"],
                launch_payload_check="size_mtime_not_full_rehash",
                model_quality="UNVERIFIED",
            )
        elif args.phase == "analyze":
            return analyze(args)
        else:
            collect(args)
    except Exception as exc:
        emit("error", phase=args.phase, exception_type=type(exc).__name__, reason=str(exc)[:500])
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
