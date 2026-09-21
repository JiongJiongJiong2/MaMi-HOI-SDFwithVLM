import unittest

try:
    from scripts.audit_oakink_c0r_result import (
        exact_mcnemar,
        paired_binary,
    )
except ModuleNotFoundError:
    from audit_oakink_c0r_result import (  # noqa: E402
        exact_mcnemar,
        paired_binary,
    )


class AuditOakInkC0RResultTest(unittest.TestCase):
    def test_exact_mcnemar_no_discordance(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)

    def test_exact_mcnemar_symmetric(self):
        self.assertAlmostEqual(exact_mcnemar(6, 0), 0.03125)
        self.assertAlmostEqual(exact_mcnemar(0, 6), 0.03125)

    def test_paired_binary_counts(self):
        rows = [
            {"left": True, "right": True},
            {"left": False, "right": False},
            {"left": True, "right": False},
            {"left": False, "right": True},
        ]
        result = paired_binary(rows, "left", "right")
        self.assertEqual(result["both"], 1)
        self.assertEqual(result["neither"], 1)
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["right_only"], 1)
        self.assertEqual(result["exact_mcnemar_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
