"""Synthetic-only checks for the frozen train glossary extraction rule."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_ascend_train_glossary.py"
SPEC = importlib.util.spec_from_file_location("build_ascend_train_glossary", SCRIPT)
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


def row(identifier, text, language="mixed"):
    return {"id": identifier, "reference": text, "language": language}


def sufficient_rows():
    # Twenty distinct content tokens and nineteen adjacent bigrams, each present
    # in three synthetic mixed rows. No real corpus text is used by these tests.
    text = "内容 amber birch cobalt delta ember flint grove hazel ivory jade kelp linen maple nectar opal pearl reed sable thyme umber"
    return [row(f"synthetic-{index}", text) for index in range(3)]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")


class TrainGlossaryTests(unittest.TestCase):
    def test_mixed_only_and_document_frequency_precedes_total_frequency(self):
        rows = [
            row("first", "apple berry berry pear " + "zebra " * 10),
            row("second", "apple berry pear"),
            row("third", "apple berry"),
            row("excluded", "extraneous " * 100, "en"),
        ]
        unigrams, _, vocabulary, _ = extractor.collect_candidates(rows)
        self.assertEqual([item["term"] for item in unigrams], ["berry", "apple", "pear", "zebra"])
        self.assertEqual(unigrams[0]["document_frequency"], 3)
        self.assertEqual(unigrams[0]["total_frequency"], 4)
        self.assertEqual(unigrams[0]["source_ids"], ["first", "second", "third"])
        self.assertTrue(unigrams[0]["eligible"])
        self.assertFalse(unigrams[2]["eligible"])
        self.assertIn("extraneous", vocabulary)

    def test_ties_are_lexical_independent_of_row_order(self):
        rows = [row("b", "zebra apple"), row("a", "apple zebra")]
        first, _, _, _ = extractor.collect_candidates(rows)
        second, _, _, _ = extractor.collect_candidates(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual([item["term"] for item in first], ["apple", "zebra"])

    def test_bigrams_do_not_bridge_function_words_han_or_digits(self):
        unigrams, bigrams, _, _ = extractor.collect_candidates([
            row("synthetic", "alpha and beta 中 gamma 7 delta uh epsilon,zeta")
        ])
        self.assertEqual([item["term"] for item in bigrams], ["epsilon zeta"])
        self.assertNotIn("and", {item["term"] for item in unigrams})
        self.assertNotIn("uh", {item["term"] for item in unigrams})

    def test_normalization_rejects_non_ascii_content_and_single_letters(self):
        self.assertEqual(extractor.normalize_tokens("Ａlpha中国42B, CAFÉ"), ["alpha", "中", "国", "42", "b", "café"])
        unigrams, _, _, _ = extractor.collect_candidates([row("synthetic", "Ａlpha中国42B, CAFÉ")])
        self.assertEqual([item["term"] for item in unigrams], ["alpha"])

    def test_annotations_never_become_vocabulary_or_bridge_bigrams(self):
        unigrams, bigrams, vocabulary, counts = extractor.collect_candidates([
            row("mixed", "social [noise] media [UNK] [loud [nested] noise] kept"),
            row("english-only", "alpha [noise] beta", "en"),
        ])
        self.assertEqual([item["term"] for item in unigrams], ["kept", "media", "social"])
        self.assertEqual(bigrams, [])
        self.assertEqual(vocabulary, {"social", "media", "kept", "alpha", "beta"})
        self.assertEqual(counts, {
            "annotation_span_count": 4, "affected_row_count": 2,
            "unclosed_span_count": 0, "mixed_annotation_span_count": 3,
            "mixed_affected_row_count": 1, "mixed_unclosed_span_count": 0,
        })

    def test_annotation_sanitization_handles_compatibility_and_unclosed_brackets(self):
        tokens, count, unclosed = extractor.transcript_tokens("alpha［noise］beta [UNK")
        self.assertEqual(tokens, ["alpha", extractor.ANNOTATION_BOUNDARY,
                                  "beta", extractor.ANNOTATION_BOUNDARY])
        self.assertEqual(count, 2)
        self.assertTrue(unclosed)

    def test_exact_counts_all_candidates_and_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.jsonl"
            rows = sufficient_rows()
            rows[0]["reference"] += " [UNK]"
            write_rows(source, rows)
            first = extractor.build(source, base / "first")
            second = extractor.build(source, base / "second")
            self.assertEqual(first, second)
            for name in ("correct.json", "unrelated.json", "extraction.json"):
                self.assertEqual((base / "first" / name).read_bytes(), (base / "second" / name).read_bytes())
            correct = json.loads((base / "first" / "correct.json").read_text(encoding="utf-8"))
            unrelated = json.loads((base / "first" / "unrelated.json").read_text(encoding="utf-8"))
            for terms in (correct, unrelated):
                self.assertEqual(len(terms), 30)
                self.assertEqual(len(set(terms)), 30)
                self.assertEqual(sum(len(term.split()) == 1 for term in terms), 20)
                self.assertEqual(sum(len(term.split()) == 2 for term in terms), 10)
            self.assertEqual(len(first["unigram_candidates"]), 20)
            self.assertEqual(len(first["bigram_candidates"]), 19)
            self.assertEqual(first["provenance"]["input_row_count"], 3)
            self.assertEqual(first["provenance"]["mixed_row_count"], 3)
            self.assertEqual(first["transcript_sanitization_counts"], {
                "annotation_span_count": 1, "affected_row_count": 1,
                "unclosed_span_count": 0, "mixed_annotation_span_count": 1,
                "mixed_affected_row_count": 1, "mixed_unclosed_span_count": 0,
            })
            self.assertEqual(first["provenance"]["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(first["provenance"]["extractor_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            for name, digest in first["provenance"]["output_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((base / "first" / name).read_bytes()).hexdigest())
            self.assertTrue(all(item["document_frequency"] == 3 for item in first["selected_correct"]))

    def test_controls_exclude_tokens_from_every_training_language(self):
        rows = sufficient_rows() + [row("english-only", "tor erg ria fen col tarn", "en")]
        unigrams, bigrams, vocabulary, _ = extractor.collect_candidates(rows)
        correct = extractor.select_correct(unigrams, bigrams)
        controls, assignments, audit = extractor.select_unrelated(correct, vocabulary)
        all_control_tokens = [token for term in controls for token in extractor.normalize_tokens(term)]
        self.assertFalse(set(all_control_tokens) & vocabulary)
        self.assertEqual(len(all_control_tokens), len(set(all_control_tokens)))
        self.assertEqual([len(term["term"].split()) for term in correct], [len(term.split()) for term in controls])
        for term in ("tor", "erg", "ria", "fen", "col", "tarn"):
            item = next(item for item in audit if item["term"] == term)
            self.assertFalse(item["eligible"])
            self.assertEqual(item["train_token_collisions"], [term])
        self.assertEqual(len(assignments), 30)

    def test_distractor_matching_uses_fixed_distance_and_lexical_tie_break(self):
        correct = [{"term": "test", "token_count": 1}]
        controls, assignments, audit = extractor.select_unrelated(correct, set())
        candidates = [item["term"] for item in audit if item["token_count"] == 1]
        expected = min(candidates, key=lambda term: (abs(len(term) - 4), term))
        self.assertEqual(controls, [expected])
        self.assertEqual(assignments[0]["length_distance"], [0, 0, 0])

    def test_insufficient_candidates_fail_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.jsonl"
            write_rows(source, [row("only", "alpha beta")])
            with self.assertRaisesRegex(ValueError, "Frozen rule has no fallback"):
                extractor.build(source, base / "output")
            self.assertFalse((base / "output").exists())

    def test_existing_output_is_refused_without_touching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "output"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                extractor.build(base / "nonexistent.jsonl", output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_train_schema_and_document_identity_are_enforced(self):
        invalid = [
            [row("same", "alpha"), row("same", "beta")],
            [dict(row("synthetic", "alpha"), audio="not permitted in this export")],
            [{"id": 3, "reference": "alpha", "language": "mixed"}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            for rows in invalid:
                write_rows(source, rows)
                with self.assertRaises(ValueError):
                    extractor.read_train(source)
            source.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Blank JSONL row"):
                extractor.read_train(source)


if __name__ == "__main__":
    unittest.main()
