# Prototype — Can Phonon-2 improve LG Gram dictation?

**Critical unknown:** Whether the official Phonon-2 Linux CPU runtime can match
the current Whisper SYCL recognition while returning short chunks within two
seconds on the i7-1360P.
**Scenario / real boundary:** A warm loopback Phonon server and the existing
Whisper server transcribe the same synthetic development WAVs, including the
retained 1–150 fixture's 19 source-owned chunks. Use the existing 0.5-second
inference padding for both. No microphone, clipboard, or live service mutation.
**Budget / stop:** One installation and measurement round, at most 30 minutes
and 2 GB of downloaded dependencies/model artifacts. Stop for a reproducible
runtime failure or measured negative evidence. The sealed 1–500 holdout is excluded.
**Measurement oracle:** Client wall time per request, median/p95/maximum latency,
server-reported decode time when available, exact ordered numeric score, raw
WAV hashes, and transcripts. Measure model loading separately from warm requests.
**Acceptance oracle:** Phonon must return the 1–150 development sequence exactly,
have no failed requests, have warm p95 latency below two seconds, and have median
latency no higher than the paired Whisper baseline. Passing establishes only
synthetic offline inference feasibility, not live dictation acceptance.
**Valid while / revalidate when:** This laptop, power/load conditions, installed
runtime/model versions, and these audio inputs remain comparable. A different
model, accelerator, segmentation, or real human recording needs its own evidence.

## Harness self-test

The unchanged numeric scorer accepted exact words and mixed digit/word input,
and rejected missing, duplicate, substituted, reordered, and unrelated tokens.
Word output is accepted, so Phonon's spelling of numbers is not itself a failure.
The disposable paired harness passes Ruff check and formatting checks.

## Results

Runtime documentation: [Phonon](https://github.com/fermionresearch/phonon),
[CPU](https://github.com/fermionresearch/phonon/blob/main/docs/cpu.md),
[model](https://huggingface.co/FermionResearch/Phonon-2).

### Round 2026-10-19 — 19 paired development chunks

| Measurement | Phonon-2 CPU | Whisper `small.en` SYCL |
|---|---:|---:|
| Completed measured requests | 19/19 | 19/19 |
| Median request time | 0.421 s | 1.392 s |
| Mean request time | 0.529 s | 1.375 s |
| p95 / maximum request time | 1.004 s | 1.998 s |
| Separate first request | 0.387 s | 1.370 s |
| Exact 1–150 oracle | Fail: first deviation at 55 | Pass |
| Retained source hashes | All unchanged | All unchanged |

Phonon's median was 3.31 times faster. Its 12-thread AVX2 C encoder and C TDT
loop were selected successfully. `fermion-research` was 0.2.11, Python 3.12.3,
and PyTorch 2.14.1+cpu; the model archive digest was
`98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`, verified
by the runtime, as were all unpacked members. Reported model load took 2.66 s,
separate from download and request timing. The laptop stayed on the balanced
power profile; one-minute load rose from 3.95 to 5.13 during the comparison.

The Phonon transcript rendered 55 as “fifty nine” and 81 as “Tinky one.” Manual
inspection found the remaining values in order. This is not an exact-oracle
pass. The baseline recognized all 150 values correctly on the same run.

Requests alternated engine order by chunk, used identical padded WAVs, and
included HTTP client overhead. Whisper used its current beam-five/context-512
policy and vocabulary prompt. Phonon used default greedy decoding without
hotwords. No model-specific settings were tuned after seeing these outputs.
The run compares those two practical profiles, not equally configured decoders.

Raw report and complete transcripts (local-only):
`/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/paired-150-20261019/report.json`.
The original development capture, including its silent final tail, was retained
unchanged; this inference comparison used the 19 speech chunks only. The sealed
500 fixture was neither read nor rerun.

Historical reproduction commands below use the one-time comparison harness at
Git commit `a7830500`. That harness was removed from the delivered runtime tree;
recover it from Git history in an isolated checkout if this experiment needs to
be reproduced. The maintained production capture probe is
`scripts/dictation/numeric_continuity_probe.py`.

Reproduce the historical experiment with a new output directory and a temporary Phonon server:

```bash
cd /home/linu_x/Documents/git/whisper-phonon-2
/home/linu_x/.cache/phonon-2-prototype/venv312/bin/fermion serve phonon-2 --host 127.0.0.1 --port 8180
```

In a second terminal, while the existing Whisper server is available on 8178:

```bash
cd /home/linu_x/Documents/git/whisper-phonon-2
/home/linu_x/.cache/phonon-2-prototype/venv312/bin/python scripts/dictation/phonon_probe.py \
  --session /home/linu_x/.local/share/whisper-dictation/prototypes/numeric-continuity/performance-audioctx512/dev-implemented-150-20260905T025427.551040Z/sessions/20260905T025431.647069Z-3586577 \
  --output /home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/paired-150-new-round
```

## Conclusion

Positive evidence for CPU runtime compatibility and sub-two-second warm inference;
negative evidence against the predeclared exact-recognition acceptance criterion.
The combined replacement criterion is not satisfied. This synthetic voice fixture
does not establish real-user accuracy, noisy microphone behavior, vocabulary
handling, or end-to-end hotkey delivery. A controlled ordinary human-speech
comparison is the next useful decision input; production integration is deferred.

Company-memory retrieval returned `not_configured`; no company lesson search
result is claimed. Repository capture/recovery evidence was reused unchanged.

## Disposition

Exploration stays on `codex/explore-phonon-2`. The paired harness is retained as
prototype code only. Its separate environment and downloaded model cache are
retained for further experiments; no boot service was installed for Phonon.
The temporary Phonon process is stopped after measurement. Both production
Whisper services and their configuration remain unchanged. Review pending.

### Round 2026-10-19 — dictionary hotwords and corrections

The original speed comparison deliberately had no Phonon hotwords. The prototype
now loads the same merged shared and laptop-local vocabulary as Whisper and sends
the canonical terms in a JSON `hotwords` form field on every default Phonon
request. Multiword terms stay whole, and right-hand mishearings never become
hotwords. More than 25 active terms raises an actionable error rather than
silently discarding entries. Explicit empty lists remain available to experiments.

The shared list gained `Phonon-2`, with observed/spoken variants `fun on 2`,
`fun on two`, `phono-2`, `phonon 2`, and `phonon two`. Output retains the original
response text and separately applies the existing boundary-aware replacement
rules as `corrected_text`. These replacements do not change recognition scores
or rewrite the original comparison report.

With human authorization, 1,269 retained transcript paths were examined locally
for recurring terminology. The seven additions to this laptop's private overlay
were IMU (57 mentions), HID (33), RP2040 (16), I2C (10), Grafana (16), MCP (8), and
Morpheus (6). This gives 18 shared plus seven local terms: exactly 25. Frequencies
are evidence of usefulness, not measured spelling-error rates: correct speech
cannot be inferred reliably from transcripts alone. Raw personal transcripts and
context excerpts were not added to Git or sent to an external service. No broad
ITC-to-I2C replacement was added because that acronym can have other meanings.

Five focused tests first failed before implementation and now pass: canonical
shared/local merging, phrase preservation, exclusion of misspellings, explicit
JSON transport/default dictionary use, the 25-term bound, raw/corrected output,
and boundary-safe corrections. Ruff and diff whitespace checks pass.

The real CPU server echoed the complete 25-term list in all three synthetic
hotword requests. With the publisher's default strength, hotwords changed one
HIV misrecognition to HID, but the raw Phonon-2 phrase stayed “fun on two.” The
dictionary replacement correctly produced “I want to use Phonon-2 instead of
whisper for dictation.” Grafana and RP2040 errors remained; this is wiring proof
and a limited smoke test, not a claim of perfect vocabulary recognition.
The three hotword requests took 0.197–0.267 seconds.

Local evidence:
`/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/hotwords-20261019/report.json`
and the correction verification at
`/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/hotwords-20261019-correction/report.json`.
The separate runtime was stopped after testing. This is still an exploration
branch; production Phonon integration has not been implemented or activated.

### Closure — 2026-10-19

The operator accepted Phonon-2 and authorized production integration. The prior
measurements and their limitations remain historical evidence. Current runtime
work and delivery gates are in [the production plan](../plans/phonon-production-backend.md).
