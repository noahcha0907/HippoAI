import json
import tempfile
import unittest
from pathlib import Path

from kirby.datasets import Question, QuestionInput, assert_disjoint, choose, load_development, load_medqa, prepare
from kirby.evaluation import Generation, format_prompt, score, summarize
from kirby.medgemma import ModelConfig, validate_snapshot
from kirby.records import digest, write_json
from kirby.runner import RunConfig


def row(name="Synthetic question", answer="B"):
    options = {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    return {"question": name, "options": options, "answer_idx": answer,
            "answer": options[answer], "explanation": "SECRET_EXPLANATION"}


def save_rows(path, rows):
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")


def make_manifest(directory, count=4):
    source_path = directory / "dev.jsonl"
    save_rows(source_path, [row(f"Synthetic question {i}") for i in range(count)])
    questions, source = load_medqa(source_path, "dev")
    manifest = {"schema_version": 1, "benchmark": "medqa_english_four_options", "phase": "development",
                "sources": {"dev": source}, "question_ids": [q.id for q in questions]}
    path = directory / "development.json"
    write_json(path, manifest)
    return path, questions


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "data.jsonl"

    def test_prompt_cannot_receive_explanation_or_label_fields(self):
        save_rows(self.path, [row()])
        questions, _ = load_medqa(self.path, "dev")
        prompt = format_prompt(questions[0].input)
        self.assertIn("B. second", prompt)
        self.assertNotIn("SECRET_EXPLANATION", prompt)
        self.assertNotIn("answer_idx", prompt)
        self.assertNotIn("answer", vars(questions[0].input))

    def test_rejects_wrong_options_labels_and_inconsistent_answers(self):
        invalids = [dict(row(), options={"A": "first"}), dict(row(), answer_idx="E"),
                    dict(row(), answer="wrong"), dict(row(), question=""),
                    dict(row(), options={"A": "", "B": "second", "C": "third", "D": "fourth"}), []]
        for value in invalids:
            with self.subTest(value=value):
                save_rows(self.path, [value])
                with self.assertRaises(ValueError):
                    load_medqa(self.path, "dev")

    def test_duplicate_option_text_is_preserved_flagged_and_scored_by_key(self):
        value = row()
        value["options"]["A"] = "SECOND"
        value["options"]["D"] = "third"
        save_rows(self.path, [value])
        original_bytes = self.path.read_bytes()
        questions, source = load_medqa(self.path, "dev")
        question = questions[0]
        self.assertEqual(question.input.options, ("SECOND", "second", "third", "third"))
        self.assertEqual(question.answer, "B")
        self.assertEqual(source["count"], 1)
        self.assertEqual(source["data_quality"]["duplicate_option_records"], [
            {"line": 1, "question_id": question.id, "option_groups": [["A", "B"], ["C", "D"]]}])
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertFalse(score(question, Generation("A", 1, False), 0.1)["correct"])
        self.assertTrue(score(question, Generation("B", 1, False), 0.1)["correct"])
        # Repeated options do not exempt repeated questions or split leakage.
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint(questions, questions)
        save_rows(self.path, [value, value])
        with self.assertRaisesRegex(ValueError, "duplicate underlying question"):
            load_medqa(self.path, "dev")

    def test_reordered_options_do_not_hide_cross_split_overlap(self):
        original = row("  SYNTHETIC   question ")
        reordered = row("synthetic question", "A")
        reordered["options"]["A"], reordered["options"]["B"] = "second", "first"
        reordered["answer"] = "second"
        save_rows(self.path, [original])
        first, _ = load_medqa(self.path, "train")
        save_rows(self.path, [reordered])
        second, _ = load_medqa(self.path, "test")
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint(first, second)
        save_rows(self.path, [original, reordered])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_medqa(self.path, "dev")

    def test_sampling_is_stable_and_has_no_replacement(self):
        _, questions = make_manifest(self.root, 20)
        picked = choose(questions, 8, 17)
        self.assertEqual(picked, choose(list(reversed(questions)), 8, 17))
        self.assertEqual(len({q.id for q in picked}), 8)
        with self.assertRaises(ValueError):
            choose(questions, 21, 17)

    def test_changed_dataset_and_final_manifest_rejected(self):
        path, _ = make_manifest(self.root)
        load_development(path)
        source = self.root / "dev.jsonl"
        source.write_text(" " + source.read_text(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_development(path)
        write_json(path, {"schema_version": 1, "phase": "final"})
        with self.assertRaisesRegex(ValueError, "final evaluation is separate"):
            load_development(path)

    def test_duplicate_selected_ids_are_rejected(self):
        path, _ = make_manifest(self.root)
        manifest = json.loads(path.read_text())
        manifest["question_ids"].append(manifest["question_ids"][0])
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_development(path)

    def test_preparation_separates_final_records_and_refuses_overwrite(self):
        # Synthetic data with official split sizes; tests infrastructure, not medicine.
        paths = {}
        for split, count in (("train", 64), ("dev", 1272), ("test", 1273)):
            paths[split] = self.root / f"{split}.jsonl"
            values = [row(f"Synthetic {split} question {i}") for i in range(count)]
            values[0]["options"]["D"] = values[0]["options"]["C"]
            save_rows(paths[split], values)
        output = self.root / "plan"
        result = prepare(paths["train"], paths["dev"], paths["test"], output)
        development = json.loads((output / "development.json").read_text())
        final = json.loads((output / "final.json").read_text())
        self.assertEqual(result["development_count"], 256)
        self.assertEqual(result["duplicate_option_rows"], {"train": 1, "dev": 1, "test": 1})
        self.assertEqual(len(development["sources"]["dev"]["data_quality"]["duplicate_option_records"]), 1)
        self.assertEqual(len(final["question_ids"]), 1273)
        self.assertNotIn("test", development["sources"])
        self.assertTrue(set(final["question_ids"]).isdisjoint(development["question_ids"]))
        self.assertEqual(len(load_development(output / "development.json")[0]), 256)
        with self.assertRaises(FileExistsError):
            prepare(paths["train"], paths["dev"], paths["test"], output)


class ScoringTests(unittest.TestCase):
    question = Question("q1", "hash", "dev", QuestionInput("Question?", ("a", "b", "c", "d")), "B")

    def test_strict_answer_protocol(self):
        for text, valid, correct in ((" B\n", True, True), ("A", True, False),
                                     ("The answer is B", False, False), ("b", False, False),
                                     ("B.", True, True), (" C.\n", True, False),
                                     ("B. second", False, False), ("B. C.", False, False),
                                     ("B..", False, False), ("B .", False, False),
                                     ("AB", False, False), ("", False, False)):
            with self.subTest(text=text):
                result = score(self.question, Generation(text, 2, False), 0.1)
                self.assertEqual((result["valid"], result["correct"]), (valid, correct))

    def test_truncated_correct_letter_counts_as_incorrect(self):
        prediction = score(self.question, Generation("B", 8, True), 0.1)
        self.assertFalse(prediction["correct"])
        self.assertEqual(summarize([prediction])["truncation_rate"], 1)

    def test_invalid_answers_stay_in_denominator(self):
        predictions = [score(self.question, Generation(text, 2, False), 0.1) for text in ("B", "unclear", "A")]
        result = summarize(predictions)
        self.assertEqual((result["correct"], result["total"]), (1, 3))
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertIsNone(summarize([])["accuracy"])


class ConfigTests(unittest.TestCase):
    def test_frozen_revision_and_supported_repository(self):
        for revision in ("main", "latest", "abc", "../weights"):
            with self.assertRaises(ValueError):
                ModelConfig(revision)
        with self.assertRaises(ValueError):
            ModelConfig("a" * 40, repository="untrusted/custom-code")
        self.assertTrue(ModelConfig("a" * 40).local_files_only)

    def test_invalid_resource_limits(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RunConfig(max_seconds=value)
        with self.assertRaises(ValueError):
            ModelConfig("a" * 40, max_new_tokens=0)
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

    def test_missing_and_unsafe_shards_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "config.json", {"model_type": "gemma3"})
            for filename in ("missing.safetensors", "../outside.safetensors", "C:\\outside.safetensors"):
                with self.subTest(filename=filename):
                    write_json(root / "model.safetensors.index.json", {"weight_map": {"weight": filename}})
                    with self.assertRaisesRegex(ValueError, "shard"):
                        validate_snapshot(root)
            write_json(root / "config.json", {"model_type": "custom"})
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_snapshot(root)


if __name__ == "__main__":
    unittest.main()
