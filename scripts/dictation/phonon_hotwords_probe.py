#!/usr/bin/env python3
"""Synthetic local smoke test for Phonon dictionary transport and recognition."""

import argparse
import json
import subprocess
from pathlib import Path

from phonon_probe import phonon_hotwords, request
from temporal_audio import prepared_padded_wav
from vocab_prompt import load_all_vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    words = phonon_hotwords()
    _, replacements = load_all_vocabulary()
    phrases = [
        "I want to use Phonon two instead of Whisper for dictation.",
        "Morpheus uses Grafana to monitor the I M U and I two C sensors.",
        "The R P twenty forty board uses H I D, and the agent uses M C P.",
    ]
    report = {"active_hotwords": words, "clips": []}
    for index, phrase in enumerate(phrases):
        source, audio = (
            args.output / f"tts-{index}.wav",
            args.output / f"clip-{index}.wav",
        )
        subprocess.run(
            ["espeak", "-v", "en-us", "-s", "165", "-w", str(source), phrase],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(audio),
            ],
            check=True,
        )
        row = {"spoken_text": phrase}
        with prepared_padded_wav(audio, 0.5) as padded:
            for label, selected in (("without_hotwords", []), ("with_hotwords", words)):
                result = request(
                    "phonon", padded, "", hotwords=selected, replacements=replacements
                )
                echoed = result["response"]["x_fermion"]["hotwords"]
                if echoed != selected:
                    raise RuntimeError(f"server hotword list differs for {label}")
                row[label] = result
        report["clips"].append(row)
    report["server_confirmed_complete_dictionary"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "active_count": len(words),
                "clips": [
                    {
                        "spoken": row["spoken_text"],
                        **{
                            label: row[label]["text"]
                            for label in ("without_hotwords", "with_hotwords")
                        },
                        "corrected_with_hotwords": row["with_hotwords"][
                            "corrected_text"
                        ],
                    }
                    for row in report["clips"]
                ],
                "server_confirmed_complete_dictionary": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
