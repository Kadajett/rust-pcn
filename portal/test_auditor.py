import unittest

from portal.auditor import grade_noul, objective_metrics


class ObjectiveMetricsTests(unittest.TestCase):
    def test_continuous_and_tied_outputs_are_not_boolean_accuracy(self) -> None:
        samples = [
            {
                "schema": "river-universal-output-telemetry-v1",
                "supervision": {"label": "left_flipper", "target": 0.25},
                "response": {
                    "answers": {
                        "left_flipper": {"type": "noul", "noul": 0.0}
                    }
                },
            },
            {
                "output_type": "choice",
                "expected": "refund",
                "predicted": {
                    "choice": "refund",
                    "probabilities": {
                        "information": 1 / 3,
                        "rebooking": 1 / 3,
                        "refund": 1 / 3,
                    },
                },
                "matched": False,
                "accuracy": 0.0,
            },
            {
                "output_type": "score",
                "expected": {"score": 1.0},
                "predicted": {
                    "score": 1.0,
                    "probabilities": {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
                },
                "matched": False,
                "accuracy": 1 / 3,
            },
            {
                "output_type": "structured",
                "expected": {"status": "ready", "count": 3},
                "predicted": {"status": "blocked", "count": 3},
                "matched": False,
                "accuracy": 0.5,
                "schema_valid": True,
            },
        ]

        metrics = objective_metrics(samples)

        self.assertEqual(metrics["exact_total"], 4)
        self.assertEqual(metrics["exact_correct"], 0)
        self.assertAlmostEqual(metrics["noul_mae"], 0.25)
        self.assertAlmostEqual(metrics["noul_brier"], 0.0625)
        self.assertEqual(
            metrics["by_output_type"]["noul"],
            {
                "total": 1,
                "strict_passes": 0,
                "saturated": 1,
                "mae": 0.25,
                "brier": 0.0625,
            },
        )
        self.assertEqual(metrics["by_output_type"]["choice"]["label_correct"], 1)
        self.assertEqual(metrics["by_output_type"]["choice"]["unique_top_correct"], 0)
        self.assertEqual(metrics["by_output_type"]["choice"]["ties"], 1)
        self.assertEqual(metrics["by_output_type"]["score"]["point_exact"], 1)
        self.assertEqual(metrics["by_output_type"]["score"]["unique_top_correct"], 0)
        self.assertEqual(metrics["by_output_type"]["score"]["ties"], 1)
        self.assertAlmostEqual(metrics["by_output_type"]["score"]["mae"], 0.0)
        self.assertAlmostEqual(
            metrics["by_output_type"]["structured"]["field_accuracy"], 0.5
        )

    def test_noul_soft_targets_fail_saturation_and_use_fixed_tolerance(self) -> None:
        self.assertEqual(
            grade_noul(0.25, 0.0),
            {
                "absolute_error": 0.25,
                "squared_error": 0.0625,
                "saturated": True,
                "passed": False,
            },
        )
        self.assertTrue(grade_noul(0.25, 0.30)["passed"])
        self.assertFalse(grade_noul(0.25, 0.36)["passed"])
        self.assertTrue(grade_noul(0.0, 0.0)["passed"])
        self.assertFalse(grade_noul(0.0, 0.0)["saturated"])

    def test_choice_requires_correct_label_and_unique_top_probability(self) -> None:
        metrics = objective_metrics(
            [
                {
                    "output_type": "choice",
                    "expected": "refund",
                    "predicted": {
                        "choice": "refund",
                        "probabilities": {
                            "information": 0.1,
                            "rebooking": 0.1,
                            "refund": 0.8,
                        },
                    },
                    "matched": True,
                    "accuracy": 1.0,
                }
            ]
        )

        self.assertEqual(metrics["exact_accuracy"], 1.0)
        self.assertEqual(metrics["by_output_type"]["choice"]["label_correct"], 1)
        self.assertEqual(metrics["by_output_type"]["choice"]["unique_top_correct"], 1)
        self.assertEqual(metrics["by_output_type"]["choice"]["ties"], 0)


if __name__ == "__main__":
    unittest.main()
