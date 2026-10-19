"""Synthetic sampling checks; never open transcript or vocabulary artifacts."""

import argparse
from collections import Counter
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_glossary_pilot.py"
SPEC = importlib.util.spec_from_file_location("select_glossary_pilot", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def synthetic_pool():
    return [{"id": f"synthetic-{group}-{index:03}", "group": group,
             "duration_seconds": 1 + index % 12}
            for group, count in (("mixed", 120), ("mandarin", 70), ("english", 70))
            for index in range(count)]


class SelectGlossaryPilotTests(unittest.TestCase):
    def test_fixed_hash_order_and_group_order(self):
        rows = synthetic_pool()
        actual = selector.select_rows(rows)
        expected = []
        for group, count in (("mixed", 12), ("mandarin", 6), ("english", 6)):
            pool = [row for row in rows if row["group"] == group]
            pool.sort(key=lambda row: (hashlib.sha256(f"20261018:{row['id']}".encode()).hexdigest(), row["id"]))
            expected.extend(pool[:count])
        self.assertEqual(actual, expected)
        self.assertEqual(actual, selector.select_rows(list(reversed(rows))))
        self.assertEqual([row["group"] for row in actual], ["mixed"] * 12 + ["mandarin"] * 6 + ["english"] * 6)

    def test_expansion_to_200_contains_original_24_as_group_prefixes(self):
        rows = synthetic_pool()
        original = selector.select_rows(rows)
        expanded = selector.select_rows(rows, {"mixed": 100, "mandarin": 50, "english": 50})
        self.assertEqual(len(expanded), 200)
        self.assertEqual(Counter(row["group"] for row in expanded), {"mixed": 100, "mandarin": 50, "english": 50})
        self.assertLessEqual({row["id"] for row in original}, {row["id"] for row in expanded})
        for group, count in selector.DEFAULT_COUNTS.items():
            self.assertEqual([row for row in original if row["group"] == group],
                             [row for row in expanded if row["group"] == group][:count])

    def test_selection_has_no_reference_or_vocabulary_dependency(self):
        rows = synthetic_pool()
        expected_ids = [row["id"] for row in selector.select_rows(rows)]
        with_references = [dict(row, reference="arbitrary synthetic text " * (index % 5),
                                vocabulary=["unrelated", str(index)])
                           for index, row in enumerate(rows)]
        self.assertEqual(expected_ids, [row["id"] for row in selector.select_rows(with_references)])
        self.assertEqual(rows, synthetic_pool())

    def test_duration_bounds_are_inclusive_and_other_groups_are_ignored(self):
        rows = []
        for group in selector.GROUP_ORDER:
            for index, duration in enumerate((0.999, 1, 12, 12.001)):
                rows.append({"id": f"{group}-{index}", "group": group, "duration_seconds": duration})
        rows.append({"id": "other", "group": "other", "duration_seconds": 5})
        selected = selector.select_rows(rows, {group: 2 for group in selector.GROUP_ORDER})
        self.assertEqual({row["id"] for row in selected},
                         {f"{group}-{index}" for group in selector.GROUP_ORDER for index in (1, 2)})

    def test_insufficient_eligible_pool_fails(self):
        rows = synthetic_pool()
        with self.assertRaisesRegex(ValueError, "Not enough eligible english clips: 70 < 71"):
            selector.select_rows(rows, {"mixed": 12, "mandarin": 6, "english": 71})

    def test_counts_require_positive_integers_and_all_groups(self):
        rows = synthetic_pool()
        for value in (0, -1, 1.5, True, "12"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                selector.select_rows(rows, {"mixed": value, "mandarin": 6, "english": 6})
        for counts in ({"mixed": 12}, {"mixed": 12, "mandarin": 6, "english": 6, "other": 1}):
            with self.assertRaisesRegex(ValueError, "exactly"):
                selector.select_rows(rows, counts)

    def test_cli_count_type_rejects_invalid_values(self):
        self.assertEqual(selector.positive_int("100"), 100)
        for value in ("0", "-1", "1.5", "abc", ""):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                selector.positive_int(value)
        for flag in ("--mixed-count", "--mandarin-count", "--english-count"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                selector.main(["--manifest", "not-opened.jsonl", "--personal", "not-opened-personal.json",
                               "--unrelated", "not-opened-unrelated.json", "--output-dir", "not-created", flag, "0"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
