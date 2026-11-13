# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Weight-free numerical-report contracts; no real torch_npu import."""

import importlib.util
import io
import math
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples/kv_fp4_numerics.py"


class KVFP4NumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("kv_numerics_under_test", SOURCE)
        cls.tool = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.tool
        spec.loader.exec_module(cls.tool)
        cls.codec = cls.tool.load_codec()

    def test_lower_kv_mse_can_worsen_attention_with_both_sides_quantized(self):
        case = self.tool.critical_channel_case()
        rows = {row["format"]: row for row in self.tool.evaluate_case(self.codec, case)}
        ceil, mse = rows["mx4-ceil"], rows["mx4-mse"]
        # Independent arithmetic: key0=[7.5,.375x31], key1=[7.25x32].
        # Q reads only dimension0 with value4, V0=1 and V1=0.
        reference_p = 1 / (1 + math.exp(-0.25 * 4 / math.sqrt(32)))
        clipped_p = 1 / (1 + math.exp(2 * 4 / math.sqrt(32)))
        self.assertAlmostEqual(ceil["k_mse"], (0.5**2 + 31 * 0.375**2 + 32 * 0.75**2) / 64)
        self.assertAlmostEqual(mse["k_mse"], (1.5**2 + 31 * 0.125**2 + 32 * 0.75**2) / 64)
        self.assertAlmostEqual(ceil["out_rmse"], reference_p - 0.5)
        self.assertAlmostEqual(mse["out_rmse"], reference_p - clipped_p)
        self.assertLess(mse["k_mse"], ceil["k_mse"])
        self.assertGreater(mse["out_rmse"], ceil["out_rmse"])
        self.assertEqual(mse["v_mse"], 0)
        self.assertEqual(mse["k_clip_pct"], 100 / 64)
        self.assertEqual(mse["k_scale_changed_pct"], 50)
        self.assertEqual(rows["g32-b"]["out_rmse"], 0)

    def test_cpu_cli_reports_scope_and_all_formats_without_npu_or_weights(self):
        result = subprocess.run(
            [sys.executable, str(SOURCE)],
            capture_output=True,
            text=True,
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source_kind=synthetic", result.stdout)
        self.assertIn("NPU=UNVERIFIED", result.stdout)
        self.assertIn("ARITHMETIC=GREEN", result.stdout)
        self.assertIn("MODEL_QUALITY=UNVERIFIED", result.stdout)
        self.assertIn("DECISION=HOLD_E2E", result.stdout)
        self.assertIn("query-critical-channel", result.stdout)
        for name in ("g32-b", "mx4-ceil", "mx4-mse"):
            self.assertIn(f'"format": "{name}"', result.stdout)
        self.assertLess(len(result.stdout.splitlines()), 1500)

    def test_npu_mode_requires_explicit_device_without_falling_back_to_cpu(self):
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "ASCEND_RT_VISIBLE_DEVICES": ""}
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--device", "npu"], capture_output=True, text=True, env=env, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit", result.stderr)
        self.assertNotIn("ARITHMETIC=GREEN", result.stdout)

    def test_causal_mask_uses_absolute_positions_and_contiguous_gqa_mapping(self):
        q = torch.zeros(2, 4, 32, dtype=torch.bfloat16)
        k = torch.zeros(3, 2, 32, dtype=q.dtype)
        v = torch.zeros_like(k)
        v[:, 0] = torch.tensor([1, 3, 5], dtype=q.dtype)[:, None]
        v[:, 1] = torch.tensor([10, 30, 50], dtype=q.dtype)[:, None]
        _, _, output, valid = self.tool.attention(q, k, v, torch.tensor([0, 2]))
        torch.testing.assert_close(output[0, :, 0], torch.tensor([1, 1, 10, 10], dtype=torch.float64))
        torch.testing.assert_close(output[1, :, 0], torch.tensor([3, 3, 30, 30], dtype=torch.float64))
        self.assertEqual(valid.sum().item(), 16)
        for positions in (torch.tensor([-1, 2]), torch.tensor([0, 3]), torch.tensor([0.0, 2.0])):
            with self.assertRaises(ValueError):
                self.tool.attention(q, k, v, positions)
        with self.assertRaises(ValueError):
            self.tool.attention(q[:0], k, v, torch.tensor([], dtype=torch.int64))

    def test_payload_comparison_detects_scale_code_and_signed_zero_errors(self):
        reference = {
            "qdq": torch.tensor([0.0], dtype=torch.bfloat16),
            "scale": torch.tensor([1.0]),
            "code": torch.tensor([0], dtype=torch.int32),
        }
        altered = {
            "qdq": torch.tensor([-0.0], dtype=torch.bfloat16),
            "scale": torch.tensor([2.0]),
            "code": torch.tensor([1], dtype=torch.int32),
        }
        self.assertEqual(self.tool.payload_mismatches(reference, reference), {"qdq": 0, "scale": 0, "code": 0})
        self.assertEqual(self.tool.payload_mismatches(altered, reference), {"qdq": 1, "scale": 1, "code": 1})

    def test_failed_numeric_contract_never_prints_green(self):
        output = io.StringIO()
        with (
            patch.object(self.tool, "check_contracts", side_effect=ValueError("injected numerical failure")),
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            self.assertEqual(self.tool.main([]), 1)
        self.assertNotIn("ARITHMETIC=GREEN", output.getvalue())

    def test_zero_measurements_cannot_pass(self):
        output = io.StringIO()
        with patch.object(self.tool, "cases", return_value=[]), redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(self.tool.main([]), 1)
        self.assertNotIn("ARITHMETIC=GREEN", output.getvalue())

    def test_device_payload_failure_cannot_pass(self):
        output = io.StringIO()
        # Device boundary fault injection only; no actual NPU behavior is tested.
        with (
            patch.object(self.tool, "prepare_npu"),
            patch.object(self.tool, "npu_parity", return_value={"replays": 5, "replay_vs_cpu": {"scale": 1}}),
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            self.assertEqual(self.tool.main(["--device", "npu"]), 1)
        self.assertIn("ARITHMETIC=RED", output.getvalue())
        self.assertNotIn("ARITHMETIC=GREEN", output.getvalue())


if __name__ == "__main__":
    unittest.main()
