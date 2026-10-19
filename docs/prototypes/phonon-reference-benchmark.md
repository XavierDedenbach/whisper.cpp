# Prototype — Compare both engines with published human-speech references

**Critical unknown:** How Phonon-2 CPU and Whisper small.en SYCL compare when both
are judged against an independent transcript instead of each other's output.
**Scenario / real boundary:** LibriSpeech test-clean human audiobook recordings
and their published aligned transcripts from [OpenSLR 12](https://www.openslr.org/12/),
CC BY 4.0, Panayotov et al. Select the first eligible 4–9 second complete utterance
by lexical ID from each speaker, then take the first 30 speakers by lexical ID.
Freeze this selection before inference. No reference text is sent to either model.
Both engines receive the same 16 kHz mono audio and 0.5-second trailing silence.
**Budget / stop:** One 346 MB corpus download, one frozen 30-utterance batch,
at most 30 minutes. No tuning against these references, no live service changes.
**Measurement oracle:** Aggregate word error rate `(S + D + I) / reference words`,
individual substitutions/deletions/insertions, exact utterances, and request
median/p95 latency. Report raw and dictionary-corrected output separately.
Normalization lowercases and ignores punctuation while retaining internal
apostrophes. Number spelling and contraction differences remain errors.
**Acceptance oracle:** Relative candidate: Phonon's raw aggregate WER no higher
than Whisper's and warm p95 request time below two seconds. This is a small
reference benchmark, not a statistical guarantee or live-microphone acceptance.
**Valid while / revalidate when:** These model/runtime versions, selected full
utterances, dictionary, decoder profiles, hardware, and normalization remain fixed.
The published benchmark may have been used in model development; it is not a
private holdout or representative of all technical dictation.

## Harness self-test

Passed known exact, substitution, deletion, insertion, empty hypothesis, and
case/punctuation cases before inference. Five existing dictionary transport and
correction tests also pass. Ruff checks pass. Pytest is not installed in the
isolated runtime; the dictionary tests were run with their native unittest runner.

## Results

### Round 2026-10-19 — 30 speakers, 538 reference words

The downloaded archive matches the [publisher's MD5 checksum](https://www.openslr.org/resources/12/md5sum.txt):
`32fa31d27d2e1cad72775fee3f4849a9`. The sealed numeric 500 fixture was excluded.
Thirty complete recordings totaled 201.370 seconds. The selection and original
audio SHA256 values were frozen before any request; all source hashes stayed
unchanged. Both engines completed all 30 measured requests and a separate warmup.
Request order alternated by clip. No references were supplied to either model,
and no decoding settings were tuned after seeing results.

| Measurement | Phonon-2 CPU | Whisper small.en SYCL |
|---|---:|---:|
| Strict raw word error rate | 1.49% (8/538) | 2.60% (14/538) |
| Dictionary-corrected word error rate | 1.49% (8/538) | 2.60% (14/538) |
| Exact normalized utterances | 22/30 | 22/30 |
| Substitutions / deletions / insertions | 7 / 0 / 1 | 12 / 1 / 1 |
| Median warm HTTP request | 0.471 s | 1.935 s |
| p95 warm HTTP request | 0.820 s | 4.883 s |

Phonon's measured median was 4.11 times faster. These timings include HTTP client
overhead but not recording time, chunk waiting, or clipboard delivery. Background
one-minute system load changed from 6.27 to 6.61; this was not an isolated hardware
benchmark, and sequential request order cannot eliminate load/thermal effects.

The laptop used Phonon's native AVX2 C encoder and C TDT CPU loop, 12 threads,
`fermion-research` 0.2.11, and model digest
`98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`.
The separate server reported 2.83 seconds to load. Whisper used the existing
`build-sycl/bin/whisper-server`, `ggml-small.en.bin`, eight threads, beam five,
audio context 512, and the configured vocabulary prompt. Phonon received all 25
merged canonical hotwords at default strength 2.0; every measured response echoed
the same list. Both outputs used the same replacement rules, which changed no
scores on this corpus. These are practical profiles, not equal decoder settings.

### Inspectable disagreements

Each arrow below is published reference → hypothesis, not one model → the other.
All other selected clips matched after the declared normalization.

| Utterance | Phonon differences | Whisper differences |
|---|---|---|
| 1188-133604-0005 | Carpaccio's → Carpatios | It is → MCAP just; Saint → St. |
| 121-121726-0000 | None | love making → lovemaking (two scored edits) |
| 1221-135766-0002 | Prynne → Prynn | None |
| 1320-122612-0002 | Hawkeye → Hockey | None |
| 1580-141083-0000 | None | endeavour → endeavor |
| 237-126133-0002 | missus → misses | missus → Mrs. |
| 260-123286-0000 | None | fifteenth → 15th (two scored edits) |
| 2961-960-0001 | with → which | with → which |
| 3570-5694-0001 | inserted “of” before “growth” | None |
| 3729-6852-0002 | Abbe → Abbey | Abbe → Abbey |
| 5105-28233-0000 | None | fourteen / three / five → 14 / 3 / 5 |
| 5683-32865-0004 | Chelford → Shelford | None |

**Interpretation caveat:** ten of Whisper's fourteen scored edits arise from the
spelling, abbreviation, word-joining, or number-formatting cases shown above.
These are not ten lost spoken words. The published “with the Timaeus” reference
also warrants listening review because both models returned “which the Timaeus”;
no reference was silently corrected. Therefore lower strict WER alone does not
establish a meaningful recognition-quality advantage. Phonon makes real name and
word mistakes too. Neither model is treated as ground truth.

Example recording: [1320-122612-0002.wav](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/librispeech-30-20261019/1320-122612-0002.wav).
Published reference: “After proceeding a few miles the progress of Hawkeye who
led the advance became more deliberate and watchful.” Phonon substituted “Hockey”;
Whisper matched this reference apart from punctuation.

Local full evidence, including every reference, audio path, hypothesis, edit
alignment, raw response, timing, and frozen selection:
[report.json](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/librispeech-30-20261019/report.json)
and [selection.json](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/librispeech-30-20261019/selection.json).
Public corpus audio and the large raw report are kept out of Git.

Historical reproduction requires the archived one-time harness at Git commit
`a7830500` (removed from the delivered runtime tree). Use an isolated checkout,
with the temporary Phonon server on 8180 and Whisper on 8178:

```bash
cd /home/linu_x/Documents/git/whisper-phonon-2
/home/linu_x/.cache/phonon-2-prototype/venv312/bin/python scripts/dictation/phonon_reference_probe.py \
  --dataset /home/linu_x/.cache/phonon-2-prototype/librispeech/LibriSpeech/test-clean \
  --output /home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/librispeech-new-round
```

## Conclusion

The frozen numerical acceptance criterion passed: Phonon's strict raw WER was no
higher and its warm p95 was below two seconds. This is positive evidence for fast
human-speech inference, not proof of superior semantic accuracy or authorization
to replace live Whisper. Both engines now have an independent reference oracle.
The small, clean audiobook sample does not qualify microphone noise, technical
dictation, long-form chunk ownership/order, or end-to-end desktop delivery.
The next useful accuracy test is a separately recorded technical passage with a
human-verified verbatim transcript, scored with predeclared formatting equivalence;
do not tune either engine against this already-observed set.

## Disposition

Keep the benchmark on `codex/explore-phonon-2`; review pending. The temporary
Phonon server is stopped after measurement. Live Whisper services, model,
configuration, and primary checkout remain unchanged. This harness is prototype
code, not a production backend integration.

### Closure — 2026-10-19

The operator accepted the model despite the documented measurement limitations
and authorized [production integration](../plans/phonon-production-backend.md).
This reference experiment is complete; the source and reports remain recoverable
from history and the local evidence paths above.
