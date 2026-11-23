# SPDX-License-Identifier: Apache-2.0
"""Compare three upstream attention fallbacks using archived controls."""

import argparse
import json
from statistics import fmean

import moe_l23_attn_fallback as fallback
import moe_l23_oracle as prediction
import moe_mxfp4_w_check as comparison
import moe_routing_check as reference


def layer_metrics(ref_ids, ids, ref_weights, weights):
    import torch

    matches = ref_ids.unsqueeze(-1) == ids.unsqueeze(-2)
    selected = matches.any(-1)
    # Sparse equivalent of L1 over all experts. Each set of top-k IDs is unique.
    ref_weights, weights = ref_weights.double(), weights.double()
    common = torch.minimum(ref_weights.unsqueeze(-1), weights.unsqueeze(-2)).masked_fill(~matches, 0)
    l1 = ref_weights.sum(-1) + weights.sum(-1) - 2 * common.sum((-2, -1))
    return {
        "top8_overlap": selected.double().mean((0, 1, 3)).tolist(),
        "top8_set_equal": selected.all(-1).double().mean((0, 1)).tolist(),
        "weight_l1": l1.mean((0, 1)).tolist(),
    }


def routing_weights(bundle):
    import torch

    return torch.stack([torch.stack([r["weights"] for r in s["layers"]], dim=1) for s in bundle[1]])


def analyze(output):
    import torch

    from vllm_ascend.observability.moe_routing import (
        ATTENTION_BANDS,
        ORACLE_LAYER,
        attention_projection_names,
        tensor_error,
    )

    run = reference.relative_new((output / "baseline-run.txt").read_text())
    ref, candidate = (comparison.load_artifacts(run, side) for side in ("reference", "candidate"))
    inputs = json.loads((output / "inputs.json").read_text())
    if not ref[2] or not candidate[2]:
        raise ValueError("Missing request-level graph replay evidence in the archived controls")

    spans = {"prefill": slice(0, reference.PREFILL), "decode": slice(reference.PREFILL, None)}
    ref_weights = routing_weights(ref)

    def drifts(bundle):
        weights = routing_weights(bundle)
        return {
            phase: layer_metrics(ref[0][:, span], bundle[0][:, span], ref_weights[:, span], weights[:, span])
            for phase, span in spans.items()
        }

    by_band = {"candidate": drifts(candidate)}
    candidate_hidden = {
        phase: fmean(
            tensor_error(b["layers"][ORACLE_LAYER]["hidden"][span], r["layers"][ORACLE_LAYER]["hidden"][span])[
                "relative_l2"
            ]
            for r, b in zip(ref[1], candidate[1], strict=True)
        )
        for phase, span in spans.items()
    }
    ranking = {}
    for group, layers in ATTENTION_BANDS.items():
        treated = comparison.load_artifacts(output, group)
        if not treated[2]:
            raise ValueError(f"Missing graph replay evidence for {group}")
        if not torch.equal(candidate[0][:, :, : layers.start], treated[0][:, :, : layers.start]):
            raise ValueError(f"Routing changed before the first fallback layer in {group}")
        expected = attention_projection_names(layers)
        predictions = []
        hidden = {phase: [] for phase in spans}
        for sample, target, baseline, changed in zip(inputs, ref[1], candidate[1], treated[1], strict=True):
            config = dict(changed["config"])
            loaded = frozenset(config.pop("attention_fallback_loaded", ()))
            differences = frozenset(config.pop("attention_fallback_changed", ()))
            if loaded != expected or not differences or not differences <= loaded:
                raise ValueError(f"Invalid attention replacement scope in {group}")
            if config != baseline["config"] or config != target["config"]:
                raise ValueError(f"Different runtime settings in {group}")
            for phase, span in spans.items():
                hidden[phase].append(
                    tensor_error(
                        changed["layers"][ORACLE_LAYER]["hidden"][span], target["layers"][ORACLE_LAYER]["hidden"][span]
                    )["relative_l2"]
                )
            logits = [snapshot["continuation_logits"] for snapshot in (target, baseline, changed)]
            predictions.append(prediction.compare_predictions(*logits, sample["forced"]))

        by_band[group] = drifts(treated)
        scores = torch.stack(predictions)
        prediction.report_predictions(scores, treatment="fallback", band=group)
        ranking[group] = (scores[:, 2, 1:] - scores[:, 1, 1:]).mean().item()
        for phase in spans:
            reference.emit(
                "layer23_input",
                band=group,
                phase=phase,
                candidate_hidden_rel_l2=candidate_hidden[phase],
                fallback_hidden_rel_l2=fmean(hidden[phase]),
            )

    for phase in spans:
        for layer in range(reference.LAYERS):
            reference.emit(
                "layer_drift",
                phase=phase,
                layer=layer,
                by_band={
                    group: {metric: values[layer] for metric, values in phases[phase].items()}
                    for group, phases in by_band.items()
                },
            )
    reference.emit(
        "result",
        decision="UPSTREAM_BANDS_MEASURED",
        graph_replay="GREEN",
        task_quality="UNVERIFIED",
        bands_ranked_by_decode_delta_nll=sorted(ranking.items(), key=lambda item: item[1]),
    )


def main():
    from vllm_ascend.observability.moe_routing import ATTENTION_BANDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", *ATTENTION_BANDS, "analyze"), required=True)
    for name in ("reference", "candidate", "source", "output", "previous-report"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        output = reference.relative_new(args.output)
        if args.phase == "prepare":
            fallback.prepare(args, output)
        elif args.phase == "analyze":
            analyze(output)
        else:
            fallback.collect(args, output, attention_band=args.phase)
    except Exception as exc:
        reference.emit("error", phase=args.phase, exception_type=type(exc).__name__, reason=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
