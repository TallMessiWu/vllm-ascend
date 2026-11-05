# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure CPU replay tests; load numerical modules without framework initialization."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

package = types.ModuleType("kv_replay_reference_tests")
package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/quantization")]
sys.modules[package.__name__] = package
replay = importlib.import_module(package.__name__ + ".kv_replay")
codec = importlib.import_module(package.__name__ + ".kv_fp4")


class KVReplayTests(unittest.TestCase):
    def test_causal_grouped_query_reference_uses_absolute_query_positions(self):
        q = torch.zeros(2, 4, 16)
        k = torch.zeros(3, 2, 16)
        v = torch.ones_like(k)
        v[1] *= 3
        v[2] *= 8
        v[:, 1] *= 10
        # Query rows are sampled tokens at positions 0 and 2, NOT positions 0 and 1.
        actual = replay.attention_reference(q, k, v, torch.tensor([0, 2]))
        expected = torch.tensor([[1, 1, 10, 10], [4, 4, 40, 40]], dtype=torch.float32)
        torch.testing.assert_close(actual, expected[:, :, None].expand_as(q), rtol=0, atol=0)

    def test_k_only_v_only_and_both_isolate_the_changed_tensor(self):
        q = torch.zeros(2, 2, 16)
        k = torch.zeros(2, 1, 16)
        v = torch.full_like(k, 0.75)
        v[:, :, -1] = 6.0  # raw BF16 scale=1; 0.75 -> 1.0 by RNE
        before = v.clone()
        recipe = codec.FP4Recipe(16, "bf16")
        result = replay.compare_kv(q, k, v, torch.tensor([0, 1]), recipe, recipe)
        self.assertEqual([row["mode"] for row in result], ["k", "v", "kv"])
        self.assertEqual(result[0]["attention"]["rmse"], 0)
        self.assertEqual(result[1]["k"]["rmse"], 0)
        self.assertGreater(result[1]["v"]["rmse"], 0)
        self.assertEqual(result[1]["attention"], result[2]["attention"])
        self.assertEqual(len(result[2]["attention_heads"]), 2)
        self.assertAlmostEqual(result[1]["attention"]["rmse"], (15 / 16 * 0.25**2) ** 0.5)
        torch.testing.assert_close(v, before, rtol=0, atol=0)

    def test_all_formats_replay_with_distinct_fixed_k_and_v_scales(self):
        generator = torch.Generator().manual_seed(99)
        q = torch.randn(3, 4, 32, generator=generator).bfloat16()
        k = torch.randn(7, 2, 32, generator=generator).bfloat16()
        v = torch.randn(7, 2, 32, generator=generator).bfloat16()
        positions = torch.tensor([0, 3, 6])
        for group, kind in ((16, "bf16"), (32, "bf16"), (16, "e4m3"), (32, "e4m3"), (32, "e8m0")):
            kr = codec.FP4Recipe(group, kind, 0.01 if kind == "e4m3" else None)
            vr = codec.FP4Recipe(group, kind, 0.03 if kind == "e4m3" else None)
            rows = replay.compare_kv(q, k, v, positions, kr, vr)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertGreaterEqual(row["attention"]["rmse"], 0)
                self.assertEqual(len(row["k_heads"]), 2)
                self.assertEqual(len(row["attention_heads"]), 4)
            json.dumps(rows, allow_nan=False)
        with self.assertRaises(ValueError):
            replay.compare_kv(q, k, v, positions, codec.FP4Recipe(16, "bf16"), codec.FP4Recipe(32, "bf16"))

    def test_invalid_shapes_masks_values_and_empty_samples_are_rejected(self):
        q = torch.zeros(2, 4, 16)
        k = torch.zeros(3, 2, 16)
        positions = torch.tensor([0, 2])
        for qp in (positions.float(), torch.tensor([-1, 2]), torch.tensor([0, 3]), positions[:1]):
            with self.assertRaises(ValueError):
                replay.attention_reference(q, k, k, qp)
        for bad_q in (q[:0], q[:, :3], q[:, :, :7], q.double(), q + float("nan")):
            with self.assertRaises(ValueError):
                replay.attention_reference(bad_q, k, k, positions)
        with self.assertRaises(ValueError):
            replay.attention_reference(q, k, k[:2], positions)
        with self.assertRaises(ValueError):
            replay.attention_reference(q, k + float("inf"), k, positions)

    def test_error_metrics_zero_denominators_are_explicit_and_json_safe(self):
        zero = torch.zeros(2, 3)
        both = replay.error_metrics(zero, zero)
        self.assertEqual(both["relative_l2"], 0)
        self.assertEqual(both["sqnr_db"], "undefined_both_zero")
        nonzero = replay.error_metrics(zero, torch.ones_like(zero))
        self.assertEqual(nonzero["relative_l2"], "undefined_zero_reference")
        self.assertEqual(nonzero["rmse"], 1)
        json.dumps([both, nonzero], allow_nan=False)
        for bad in (zero[:0], zero[:, :1], zero + float("nan")):
            with self.assertRaises(ValueError):
                replay.error_metrics(zero, bad)

    def test_cli_snapshot_contract_compact_report_and_no_input_changes(self):
        root = Path(__file__).resolve().parents[3]
        # A private relative directory avoids /tmp and keeps all synthetic files local.
        scratch = root / ".kv-replay-test"
        scratch.mkdir(exist_ok=True)
        self.addCleanup(lambda: scratch.rmdir())
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        source = folder / "snapshot.safetensors"
        q, k = torch.zeros(2, 2, 16), torch.zeros(2, 1, 16)
        v = torch.full_like(k, 0.75)
        v[:, :, -1] = 6
        tensors = {"q": q, "k": k, "v": v, "query_positions": torch.tensor([0, 1])}
        metadata = {"contract": "qwen3-causal-gqa-post-rope-v1", "source_kind": "synthetic", "layer": "0"}
        save_file(tensors, source, metadata=metadata)
        original = source.read_bytes()
        command = [
            sys.executable,
            str(root / "examples/offline_kv_fp4_replay.py"),
            "--input",
            str(source),
            "--k-outer-scale",
            "0.02",
            "--v-outer-scale",
            "0.03",
        ]
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = folder / "outputs/kv-replay.txt"
        text = report.read_text()
        self.assertIn("source_kind=synthetic", text)
        self.assertIn("COMPLETED: 15", text)
        self.assertLess(len(text.splitlines()), 100)
        rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        self.assertEqual(len(rows), 15)
        self.assertEqual(source.read_bytes(), original)
        # Reject wrong contracts and output/input overlap, preserving previous results.
        for kwargs in ({"metadata": {**metadata, "contract": "sparse-m3"}}, {"metadata": metadata}):
            save_file(tensors, source, **kwargs)
            extra = [] if kwargs["metadata"]["contract"] == "sparse-m3" else ["--output", "snapshot.safetensors"]
            result = subprocess.run(command + extra, cwd=folder, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("RED", result.stderr)
            self.assertEqual(report.read_text(), text)
        source.write_bytes(b"invalid safetensors")
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1)
        self.assertIn("RED", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(report.read_text(), text)
        # Atomic publication breaks an output hard link instead of corrupting input.
        save_file(tensors, source, metadata=metadata)
        original = source.read_bytes()
        report.unlink()
        os.link(source, report)
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(source.read_bytes(), original)
        self.assertIn("COMPLETED", report.read_text())


if __name__ == "__main__":
    unittest.main()
