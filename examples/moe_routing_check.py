# SPDX-License-Identifier: Apache-2.0
"""Reference-only Qwen routing collection and CPU analysis. No quality claim.

Run with the workspace launcher. The NPU phases use fresh processes, identical
teacher-forced tokens and native routed-expert outputs. Analysis imports no NPU.
Raw tensors/tokens remain local; stdout contains aggregate diagnostics only.
"""

import argparse
import hashlib
import json
import subprocess
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PREFILL, DECODE, LAYERS, EXPERTS, TOP_K = 512, 32, 48, 128, 8
DETAIL_LAYERS = (0, 23, 47)
TEXTS = (
    "Explain how rainwater moves from a mountain forest to a river. Discuss soil, roots, evaporation, "
    "and measurements a student can record. Compare observations before and after a storm. ",
    "A library lends books for several weeks. Design a small program that records borrowers, due dates, "
    "renewals and returns. Explain data structures, validation and tests without using real personal data. ",
    "Describe a careful method for comparing two ways to calculate a sum. Separate rounding error from "
    "a wrong formula. Use simple examples, explain edge cases and describe what evidence is needed. ",
    "请说明如何组织一次社区花园活动。讨论工具清点、分工、植物生长记录和安全注意事项。"
    "比较晴天与雨天的安排，解释为什么单次观察不足以得出长期结论。",
)


def emit(event, **fields):
    print("MOE1 " + json.dumps({"event": event, **fields}, ensure_ascii=True, sort_keys=True), flush=True)


def relative_new(path):
    result = Path(path)
    if result.is_absolute() or ".." in result.parts or not result.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Diagnostic outputs must stay within the execution directory")
    return result


def validate_model(path):
    model = Path(path)
    if not model.is_absolute() or not model.is_relative_to(Path("/mnt/share/weight")):
        raise ValueError("Reference weights must be under /mnt/share/weight/")
    config = json.loads((model / "config.json").read_text())
    if (model / ".nvfp4_to_bf16.incomplete").exists() or config.get("quantization_config"):
        raise ValueError("Use the completed BF16-DQ carrier, not a quantized/incomplete checkpoint")
    if config.get("architectures") != ["Qwen3MoeForCausalLM"]:
        raise ValueError("Expected plain Qwen3MoeForCausalLM")
    values = tuple(
        config.get(key) for key in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok")
    )
    if values != (48, 2048, 128, 8) or config.get("max_position_embeddings", 0) < 40960:
        raise ValueError("Unexpected reference model dimensions/context")
    return config


def provenance(source):
    import torch
    import vllm

    import vllm_ascend

    installed = Path(vllm_ascend.__file__).resolve().parent.parent
    if installed != Path(source).resolve():
        raise ValueError("Installed Ascend module is not the selected editable worktree")
    sha = subprocess.check_output(["git", "-C", str(installed), "rev-parse", "HEAD"], text=True).strip()
    vllm_git = subprocess.run(
        ["git", "-C", str(Path(vllm.__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    try:
        torch_npu_version = version("torch-npu")
    except PackageNotFoundError:
        torch_npu_version = "UNVERIFIED"
    emit(
        "source",
        vllm_sha=vllm_git.stdout.strip() if vllm_git.returncode == 0 else "UNVERIFIED",
        torch_npu_version=torch_npu_version,
        ascend_sha=sha,
        ascend_module=str(Path(vllm_ascend.__file__).resolve()),
        vllm_module=vllm.__file__,
        vllm_version=vllm.__version__,
        torch_version=torch.__version__,
    )


def make_inputs(model, dest):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    samples = []
    for text in TEXTS:
        # Repeated, explicitly synthetic text is a bounded smoke fixture, not a
        # representative calibration corpus or a task-accuracy benchmark.
        tokens = tokenizer.encode(text * 64, add_special_tokens=False)
        if len(tokens) < PREFILL + DECODE + 1:
            raise ValueError("Fixture tokenization produced too few tokens")
        samples.append({"prompt": tokens[:PREFILL], "forced": tokens[PREFILL : PREFILL + DECODE + 1]})
    body = json.dumps(samples, sort_keys=True)
    path = relative_new(dest)
    if path.exists():
        raise ValueError("Refusing to replace input manifest")
    path.write_text(body + "\n", encoding="utf-8")
    emit(
        "inputs",
        samples=4,
        prefill=PREFILL,
        actual_decode=DECODE,
        forced_samples=DECODE + 1,
        token_sha256=hashlib.sha256(body.encode()).hexdigest(),
        fixture="synthetic_repeated_text_not_calibration",
    )


def collect(args):
    import torch
    from vllm import LLM

    from vllm_ascend.observability.moe_forced_tokens import forced_continuation
    from vllm_ascend.observability.moe_routing_rpc import reset_probe, save_probe

    provenance(args.source)
    samples = json.loads(Path(args.inputs).read_text())
    if len(samples) != 4 or any(len(s["prompt"]) != PREFILL or len(s["forced"]) != DECODE + 1 for s in samples):
        raise ValueError("Unexpected fixed input manifest")
    overrides = {"architectures": ["Qwen3MoeRoutingProbeForCausalLM"]} if args.phase == "probe" else None
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        max_model_len=40960,
        max_num_seqs=100,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        # AuxOutput uses prefix-block hashes even for an uncached diagnostic run.
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        enable_return_routed_experts=True,
        enable_trace_replay=True,
        hf_overrides=overrides,
        seed=0,
        generation_config="auto",
    )
    try:
        runtime_config = llm.llm_engine.vllm_config
        emit(
            "engine_config",
            phase=args.phase,
            compilation_mode=str(runtime_config.compilation_config.mode),
            cudagraph_mode=str(runtime_config.compilation_config.cudagraph_mode),
            runner_v2=bool(runtime_config.use_v2_model_runner),
            prefix_caching=bool(runtime_config.cache_config.enable_prefix_caching),
            prefix_cache_policy="idle_reset_before_every_request",
            continuation_mode="native_trace_decode_token_ids",
            trace_replay=bool(runtime_config.model_config.enable_trace_replay),
            checkpoint_revision="UNVERIFIED",
        )
        if not runtime_config.use_v2_model_runner or not runtime_config.cache_config.enable_prefix_caching:
            raise ValueError("AuxOutput requires Runner V2 and prefix caching")
        if not runtime_config.model_config.enable_trace_replay:
            raise ValueError("Fixed continuation requires native trace replay")
        rounds = 2 if args.phase == "reference" else 1
        for repeat in range(rounds):
            all_ids = []
            for sample_id, sample in enumerate(samples):
                # Required in BOTH processes/repeats: cached native routing can
                # look complete even when most prefill rows were not recomputed.
                # The scheduler also advances the auxiliary output namespace on
                # a successful reset. Never preempt running requests to force it.
                if llm.reset_prefix_cache(reset_running_requests=False, reset_connector=False) is not True:
                    raise ValueError("Idle prefix-cache reset failed; refusing a potentially cached request")
                emit("prefix_cache_reset", phase=args.phase, repeat=repeat, sample=sample_id, success=True)
                if args.phase == "probe":
                    armed = llm.apply_model(reset_probe)
                    if len(armed) != 1 or armed[0] != {"armed_layers": LAYERS}:
                        raise ValueError("Missing or unexpected worker arm response")
                params = forced_continuation(sample["forced"])
                outputs = llm.generate([{"prompt_token_ids": sample["prompt"]}], params, use_tqdm=False)
                if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                    raise ValueError("Expected exactly one completed request")
                cached_tokens = getattr(outputs[0], "num_cached_tokens", None)
                if cached_tokens != 0:
                    raise ValueError(f"Expected zero prefix-cache hits, got {cached_tokens!r}")
                out = outputs[0].outputs[0]
                if list(outputs[0].prompt_token_ids) != sample["prompt"] or list(out.token_ids) != sample["forced"]:
                    raise ValueError("Teacher-forced token alignment failed")
                ids = torch.as_tensor(out.routed_experts.copy()) if out.routed_experts is not None else None
                if ids is None or tuple(ids.shape) != (PREFILL + DECODE, LAYERS, TOP_K):
                    raise ValueError("Missing/truncated/misaligned native routed-experts output")
                all_ids.append(ids.int())
                if args.phase == "probe":
                    dest = str(relative_new(args.output) / f"probe-{sample_id}.pt")
                    saved = llm.apply_model(partial(save_probe, path=dest))
                    if len(saved) != 1 or saved[0].get("rows_per_layer") != PREFILL + DECODE:
                        raise ValueError("Missing worker snapshot result")
                emit(
                    "request_complete",
                    phase=args.phase,
                    repeat=repeat,
                    sample=sample_id,
                    forwarded_rows=PREFILL + DECODE,
                    layers=LAYERS,
                    token_alignment="GREEN",
                    num_cached_tokens=cached_tokens,
                )
            path = relative_new(args.output) / f"{args.phase}-{repeat}.pt"
            if path.exists():
                raise ValueError("Refusing to replace routing artifacts")
            torch.save(torch.stack(all_ids), path)
        emit("collection_complete", phase=args.phase, rounds=rounds, model_quality="UNVERIFIED")

    finally:
        # Only shut down the engine this diagnostic created, never another service.
        llm.llm_engine.engine_core.shutdown()


def load_router_weight(model, layer):
    from safetensors import safe_open

    root = Path(model)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    key = f"model.layers.{layer}.mlp.gate.weight"
    shard = root / index["weight_map"][key]
    if not shard.resolve().is_relative_to(root.resolve()):
        raise ValueError("Router shard escapes the reference weight directory")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if tuple(weight.shape) != (EXPERTS, 2048) or str(weight.dtype) != "torch.bfloat16":
        raise ValueError("Unexpected router parameter shape/dtype")
    return weight


def analyze(args):
    """Validate coverage first, then separate ID repeatability from arithmetic.

    Native outputs contain IDs only. Detailed snapshot weights/logits are never
    invented for the baseline; their comparisons have their own explicit scope.
    """
    import torch

    from vllm_ascend.observability.moe_routing import (
        router_reference,
        routing_metrics,
        selection_metrics,
        tensor_error,
    )

    root = relative_new(args.output)
    reference = torch.load(root / "reference-0.pt", map_location="cpu", weights_only=True)
    repeated = torch.load(root / "reference-1.pt", map_location="cpu", weights_only=True)
    probed = torch.load(root / "probe-0.pt", map_location="cpu", weights_only=True)
    expected = (4, PREFILL + DECODE, LAYERS, TOP_K)
    if any(tuple(t.shape) != expected for t in (reference, repeated, probed)):
        raise ValueError("Routing artifact shapes do not match the agreed workload")
    config = validate_model(args.model)
    router_weights = {layer: load_router_weight(args.model, layer) for layer in DETAIL_LAYERS}
    review = False
    replay_verified = True
    for sample in range(4):
        snapshot = torch.load(root / f"probe-{sample}.pt", map_location="cpu", weights_only=True)
        if len(snapshot["layers"]) != LAYERS:
            raise ValueError("Missing captured layers")
        emit("runtime_config", sample=sample, **snapshot["config"])
        for layer, trace in enumerate(snapshot["layers"]):
            if not torch.equal(trace["ids"].int(), probed[sample, :, layer].int()):
                raise ValueError("Internal snapshot and native logical IDs disagree")
            if (
                trace["device_calls"] != DECODE + 1
                or trace["prefill_rows"] != PREFILL
                or trace["decode_rows"] != DECODE
            ):
                raise ValueError("Invalid per-layer coverage")
            replay_verified &= trace["device_calls"] - trace["host_calls"] >= DECODE
        for stage, span in (("prefill", slice(0, PREFILL)), ("decode", slice(PREFILL, PREFILL + DECODE))):
            for label, candidate in (("reference_repeat", repeated), ("probe_effect", probed)):
                rows = []
                for layer in range(LAYERS):
                    row = selection_metrics(reference[sample, span, layer], candidate[sample, span, layer], EXPERTS)
                    rows.append(row)
                worst = min(range(LAYERS), key=lambda i: rows[i]["overlap_mean"])
                exact_rate = sum(row["set_equal_rate"] for row in rows) / LAYERS
                review |= exact_rate != 1.0
                emit(
                    "selection",
                    comparison=label,
                    sample=sample,
                    stage=stage,
                    tokens_per_layer=rows[0]["tokens"],
                    layers=LAYERS,
                    overlap_mean=sum(row["overlap_mean"] for row in rows) / LAYERS,
                    set_equal_rate=exact_rate,
                    worst_layer=worst,
                    worst=rows[worst],
                    weight_metrics="UNAVAILABLE_in_native_ID_only_output",
                )
            for layer in DETAIL_LAYERS:
                row = snapshot["layers"][layer]
                if not torch.isfinite(row["output"][span]).all():
                    raise ValueError("Nonfinite captured MoE output")
                cpu_logits, cpu_ids, cpu_weights, margin = router_reference(
                    row["hidden"][span], router_weights[layer], TOP_K, bool(config["norm_topk_prob"])
                )
                eager_error = tensor_error(row["logits"][span], row["eager_logits"][span])
                eager_routes = routing_metrics(
                    row["ids"][span], row["eager_ids"][span], row["weights"][span], row["eager_weights"][span], EXPERTS
                )
                review |= (
                    not eager_error["exact"]
                    or eager_routes["set_equal_rate"] != 1.0
                    or eager_routes["weight_l1_mean"] != 0
                )
                emit(
                    "arithmetic",
                    sample=sample,
                    stage=stage,
                    layer=layer,
                    parameter_dtype=row["parameter_dtype"],
                    logits_dtype=str(row["logits"].dtype),
                    capture_vs_npu_eager_logits=eager_error,
                    capture_vs_npu_eager_routes=eager_routes,
                    capture_vs_cpu_fp64_logits=tensor_error(row["logits"][span], cpu_logits),
                    capture_vs_cpu_fp64_routes=routing_metrics(
                        cpu_ids, row["ids"][span], cpu_weights, row["weights"][span], EXPERTS
                    ),
                    cpu_fp64_boundary=margin,
                    moe_output_finite=bool(torch.isfinite(row["output"][span]).all()),
                    cpu_oracle="MEASURED_not_bitwise_kernel_equivalence",
                )
    emit(
        "result",
        capture="GREEN",
        graph_replay="GREEN" if replay_verified else "UNVERIFIED",
        reference_self_check="REVIEW" if review else "IDENTICAL",
        model_quality="UNVERIFIED",
        quantization_drift="NOT_TESTED",
        routing_recovery="NOT_TESTED",
        performance="NOT_TESTED",
        decision="REVIEW_DIFFERENCES" if review or not replay_verified else "READY_FOR_CANDIDATE",
    )
    return 1 if review or not replay_verified else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "reference", "probe", "analyze"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        validate_model(args.model)
        relative_new(args.output).mkdir(parents=True, exist_ok=True)
        if args.phase == "prepare":
            make_inputs(args.model, args.inputs)
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
