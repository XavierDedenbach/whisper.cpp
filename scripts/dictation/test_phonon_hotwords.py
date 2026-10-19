"""The prototype must send canonical merged vocabulary to the real HTTP seam."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import phonon_probe
from vocab_prompt import apply_replacements, load_all_vocabulary


class PhononHotwordTests(unittest.TestCase):
    def test_merged_dictionary_keeps_phrases_and_excludes_mishearings(self):
        with tempfile.TemporaryDirectory() as raw:
            repo, user = Path(raw) / "repo.txt", Path(raw) / "user.txt"
            repo.write_text("Phonon-2 => fun on 2\npull request\nVantage\n")
            user.write_text("Ada Lovelace\nvantage\n")
            with (
                mock.patch("vocab_prompt.repo_vocabulary_path", return_value=repo),
                mock.patch("vocab_prompt.user_vocabulary_path", return_value=user),
            ):
                words = phonon_probe.phonon_hotwords({"test": "1"})
            self.assertEqual(
                words, ["Phonon-2", "pull request", "Vantage", "Ada Lovelace"]
            )

    def test_dictionary_over_the_server_limit_is_not_silently_truncated(self):
        with mock.patch(
            "phonon_probe.load_all_vocabulary",
            return_value=([f"term{i}" for i in range(26)], []),
        ):
            with self.assertRaisesRegex(ValueError, "25"):
                phonon_probe.phonon_hotwords()

    def test_request_sends_json_hotwords_and_preserves_raw_and_corrected_text(self):
        words = ["Phonon-2", "pull request", "Smith, Jones"]
        response = subprocess.CompletedProcess(
            [], 0, json.dumps({"text": "Try Fun on 2."}), ""
        )
        with mock.patch("phonon_probe.subprocess.run", return_value=response) as run:
            result = phonon_probe.request(
                "phonon",
                Path("/tmp/audio.wav"),
                "unused",
                hotwords=words,
                replacements=[("fun on 2", "Phonon-2")],
            )
        command = run.call_args.args[0]
        field = next(value for value in command if value.startswith("hotwords="))
        self.assertEqual(json.loads(field.partition("=")[2]), words)
        self.assertIn("response_format=verbose_json", command)
        self.assertEqual(result["text"], "Try Fun on 2.")
        self.assertEqual(result["corrected_text"], "Try Phonon-2.")

    def test_default_request_uses_dictionary(self):
        with (
            mock.patch("phonon_probe.phonon_hotwords", return_value=["Phonon-2"]),
            mock.patch(
                "phonon_probe.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, '{"text":"Phonon-2"}', ""
                ),
            ) as run,
        ):
            phonon_probe.request("phonon", Path("/tmp/audio.wav"), "unused")
        self.assertIn('hotwords=["Phonon-2"]', run.call_args.args[0])

    def test_observed_phonon_mishearings_are_boundary_safe(self):
        terms, replacements = load_all_vocabulary()
        self.assertIn("Phonon-2", terms)
        self.assertNotIn("fun on 2", [term.lower() for term in terms])
        self.assertEqual(
            apply_replacements(
                "Fun on 2, phono-2, phonon 2, fun on two, phonon two.", replacements
            ),
            "Phonon-2, Phonon-2, Phonon-2, Phonon-2, Phonon-2.",
        )
        self.assertEqual(
            apply_replacements("Have fun on 20.", replacements), "Have fun on 20."
        )


if __name__ == "__main__":
    unittest.main()
