import unittest

from scripts.evaluate_river_capabilities import aggregate, arithmetic_source, contrasts, fixtures, grade


class CodeCapabilityGradingTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["id"]: case for case in fixtures() if case["family"] == "code"}

    def answer(self, source):
        return {"type": "text", "text": source, "output_scope": "inherited"}

    def test_constant_return_cannot_pass_an_argument_dependent_function(self):
        for name, constant in (("code-plus", 10), ("code-minus", 4)):
            result = grade(self.cases[name], self.answer(f"def f(x): return {constant}"))
            self.assertFalse(result["pass"])
            self.assertEqual(result["executions"][-1]["result_delta"], 0)
            self.assertTrue(any(row["result_delta"] != 0 for row in result["executions"][:-1]))

    def test_correct_functions_pass_distinct_negative_zero_and_positive_arguments(self):
        for name, operator in (("code-plus", "+"), ("code-minus", "-")):
            result = grade(self.cases[name], self.answer(f"def f(x): return x {operator} 3"))
            self.assertTrue(result["pass"])
            self.assertEqual([row["input"] for row in result["executions"]], [-2, 0, 7])
            self.assertEqual(result["diff"]["result_deltas"], [0, 0, 0])

    def test_expression_only_output_is_not_a_function_implementation(self):
        with self.assertRaises(ValueError):
            grade(self.cases["code-plus"], self.answer("7 + 3"))

    def test_model_code_cannot_escape_the_arithmetic_whitelist(self):
        for source in ("import os", "def f(x): return abs(x)",
                       "def f(x): return x.real", "def f(x=7): return x+3",
                       "def f(x): return 2**x", "def f(x): return [x][0]"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                arithmetic_source(source)


class CapabilityEvidenceBoundaries(unittest.TestCase):
    def test_transport_failures_do_not_count_as_model_failures(self):
        rows = [
            {"family": "code", "actual": {"text": "def f(x): return x+3"}, "pass": True,
             "errors": [], "grading": {}, "http": {"latency_seconds": 1}},
            {"family": "code", "actual": {"text": "***"}, "pass": False,
             "errors": ["grading rejected output"], "grading": {},
             "http": {"latency_seconds": 1}},
            {"family": "code", "actual": None, "pass": False, "errors": ["HTTP 403"],
             "grading": {}, "http": {"latency_seconds": 1}},
        ]
        result = aggregate(rows)["code"]
        self.assertEqual((result["cases"], result["graded_cases"], result["unavailable_cases"]),
                         (3, 2, 1))
        self.assertEqual((result["passed"], result["failed"]), (1, 1))
        self.assertEqual(result["cases_with_errors"], 2)
        unavailable = aggregate(rows[-1:])["code"]
        self.assertEqual((unavailable["graded_cases"], unavailable["passed"], unavailable["failed"]),
                         (0, 0, 0))

    def test_observed_invalid_code_still_exposes_sensitivity(self):
        for first, second, changed in (
            ("***", "***", False),
            ("***", "@@@", True),
            (None, "***", None),
            ("***", None, None),
        ):
            with self.subTest(first=first, second=second):
                rows = []
                for identifier, baseline, text in (
                    ("code-plus", None, first), ("code-minus", "code-plus", second)
                ):
                    rows.append({
                        "id": identifier, "contrast_with": baseline, "family": "code",
                        "actual": None if text is None else {"text": text}, "pass": False,
                        "errors": ["HTTP 403" if text is None else "grading rejected output"],
                    })
                result = contrasts(rows)[0]
                self.assertIs(result["actual_changed"], changed)
                self.assertEqual(result["insensitivity_observed"], changed is False)
                self.assertFalse(result["both_semantically_pass"])

    def test_noul_endpoint_probabilities_are_scored_not_discarded(self):
        case = next(case for case in fixtures() if case["id"] == "noul-quarter")
        for probability in (0.0, 1.0):
            answer = {"type": "noul", "noul": probability, "output_scope": "request_conditioned"}
            result = grade(case, answer)
            self.assertFalse(result["pass"])
            self.assertEqual(result["brier"], (probability - 0.25) ** 2)
            self.assertEqual(result["mae"], abs(probability - 0.25))
        for probability in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                grade(case, {"type": "noul", "noul": probability,
                             "output_scope": "request_conditioned"})


if __name__ == "__main__":
    unittest.main()
