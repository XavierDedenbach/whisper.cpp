# Prototype — Compare personal dictation against a third-model draft

**Critical unknown:** How Phonon-2 and Whisper compare on the operator's retained
wording when neither evaluated engine supplies the reference.
**Scenario / real boundary:** Ten previously selected complete personal sessions,
chosen from metadata only. Generate local Canary 180M Flash INT8 draft references
before reading existing ASR text or making candidate requests. Freeze references
and source hashes first. Keep private audio and text outside Git.
**Budget / stop:** One third recognizer, one ten-session draft and comparison
batch, under 2 GB downloads and 30 minutes. No live configuration changes,
dictionary tuning, model tuning, or external audio uploads.
**Measurement oracle:** Strict word disagreement against the frozen provisional
drafts, number-format-normalized disagreement, and warm HTTP median/p95 latency
per chunk. Report raw and dictionary-corrected results separately. Lowercase and
ignore punctuation; normalized scoring additionally expands standalone numeric
cardinals/ordinals/decimals with num2words. Keep contractions and fillers. Do not
rewrite technical terms from context. Do not infer confidence from agreement.
**Acceptance oracle:** Measurement-only: an unverified third model is not ground
truth, so no recognition-accuracy pass or replacement decision is possible.
Produce ten complete draft references, both candidate transcripts, and explicit
disagreement lists for listening review; fail the harness on missing source audio.
**Valid while / revalidate when:** Frozen inputs, versions, dictionaries, and
policies remain fixed. Independent decoder/model does not imply independent
training data or infallibility. Human listening review remains outstanding.

Canary has a separate checkpoint and autoregressive decoder. Phonon is based on
Parakeet, so using Parakeet itself as the third recognizer would bias agreement
toward Phonon. Sources: [Canary model](https://huggingface.co/nvidia/canary-180m-flash),
[Sherpa conversion/runtime](https://k2-fsa.github.io/sherpa/onnx/nemo/canary.html),
[Phonon model ancestry](https://huggingface.co/FermionResearch/Phonon-2).

## Harness self-test

Passed exact, substitution/deletion/insertion, number-equivalence, filler and
contraction preservation, and technical-identifier cases before measurement.
The initial unusable punctuation-only reference also exercised a negative gate:
comparison rejected it before any candidate request. Ruff checks pass.

## Results

The third-model archive matches its GitHub release asset SHA256:
`7a38ed8b13f014ad632b09ff8d22e0c6f1359dd046af9235d281dfae841b9ab9`.
The isolated third-recognizer environment uses Python 3.12.3, sherpa-onnx 1.13.8,
and num2words 0.5.14. Canary drafts whole sessions with four CPU threads, English
input/output, greedy decoding, and no dictionary or post-corrections. They are
machine drafts, not listening-verified references.

The initial metadata-only selection contained one recording whose Canary output
had no lexical words. That is an unscorable reference, not proof of absent speech
or a microphone fault. Its original evidence was retained. Before any candidate
inference, replace it with the next chronological eligible session; preserve all
nine other inputs and reuse their unchanged Canary drafts. `selection-v2.json`
records this sole selection repair. All ten replacement-set drafts and source
hashes were sealed before running the candidates.

Candidate inputs preserve current source-owned chunk boundaries. Only the two
older sessions with chunks longer than nine seconds are split into at-most-eight
second parts, identically for both candidates, to fit the current context-512
profile. These fixed legacy splits can themselves cut words. Both candidates get
the same 0.5-second trailing padding, vocabulary, and replacement rules; Phonon
gets 25 canonical hotwords and Whisper gets its configured prompt. Engine request
order alternates by chunk, and first-request warmups are excluded from timing.
Canary's whole-session context differs from those practical candidate chunk inputs.

Private selection and raw data are under `personal-10-20261019/` in the existing
Phonon prototype data directory, outside Git. Company-memory access remains
`not_configured`; no successful company lesson search is claimed.

### Round 2026-10-19 — ten personal recordings

The replacement set contains 363.58 seconds of retained audio, 51 candidate input
chunks, and 618 scored draft-reference words. Both candidates completed 51/51
measured HTTP requests, plus a separate warmup. The complete drafts and input
hashes were frozen before candidate inference; baseline and original source
hashes remained unchanged. Every Phonon response echoed all 25 hotwords.

- Phonon-2 CPU: median 0.602 s, p95 2.702 s per warm request. Strict draft
  disagreement 83/618 (13.43%); number-normalized disagreement 83/618 (13.43%).
- Whisper small.en SYCL: median 1.980 s, p95 6.782 s. Strict draft disagreement
  75/618 (12.14%); number-normalized disagreement 69/618 (11.17%).
- Dictionary-corrected aggregate scores equal the raw scores for both engines.
- Phonon's median was 3.29 times faster, but eight requests exceeded two seconds;
  its maximum was 3.261 s. This is not a consistent sub-two-second result.
- Phonon returned empty text on the first recording's 3.6-second legacy tail;
  Whisper returned a one-letter token plus punctuation. Keep this in listening
  review; neither an HTTP failure nor proof that spoken words were lost is claimed.

These are **disagreement rates with a provisional machine draft, not measured
word-error rates against verified speech**. The draft has spelling, word-joining,
unit, and technical-term ambiguities. Neither number normalization nor the
dictionary removes all formatting effects. Fillers and repetitions also differ
between models. Canary emitted a first-token end-of-text warning on some source
audio and used its non-end-token fallback; a nonempty draft is not a confidence
guarantee. Do not infer that a more fluent draft is more faithful.

One-minute system load rose from 9.14 to 11.98. Timing includes HTTP overhead and
0.5-second padding, but not recording/chunk wait or clipboard delivery. The
temporary Phonon model loaded in 2.33 s separately; Canary was unloaded before
timing either candidate. Production Whisper stayed loaded throughout.

Private, local-only evidence:

- [Listening review: ten full recordings, drafts, both transcripts, and edit lists](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/personal-10-20261019/review.md)
- [Frozen references and source/input hashes](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/personal-10-20261019/canary-round-2/baseline.json)
- [Complete comparison and original HTTP responses](/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-2/personal-10-20261019/canary-round-2/comparison.json)

The private review highlights disagreements shared by both candidates, as well
as technical terminology, spoken URLs, filler differences, and the ambiguous
tail. It never silently changes the reference. Personal audio, text, vocabulary
overlay, and raw responses are not committed or uploaded.

## Conclusion

The requested provisional comparison is complete. Phonon is faster at the median
and p95 in this run; Whisper agrees more closely with the third-model drafts in
aggregate. Recognition quality remains inconclusive until the actual wording is
confirmed by listening. This does not establish a replacement decision or an
accuracy regression. Record listening-confirmed references as a new version and
rescore the retained outputs; no new inference is needed for unchanged inputs.

## Disposition

Exploration only on `codex/explore-phonon-2`; listening review pending. The
temporary Phonon process was stopped, both live Whisper services remained active,
and the primary checkout was clean. Keep the isolated model/runtime cache and
private comparison data for review. No production backend change, personal
transcript publication, PR, merge, or cutover is authorized.

### Closure — 2026-10-19

The operator accepted Phonon-2 without requiring further listening qualification
and authorized [production integration](../plans/phonon-production-backend.md).
The provisional references remain unverified, not silently promoted to truth.
The one-time draft/comparison harness is archived at Git commit `a7830500`;
private audio, drafts and review sheets remain local and intact.
