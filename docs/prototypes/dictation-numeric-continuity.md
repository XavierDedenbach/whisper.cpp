# Prototype — long-dictation numeric continuity

**Report ID:** standalone-dictation-numeric-continuity
**Shapes:** assembly, feasibility
**Governing goal:** Make long dictation retain every spoken span; use ordered counting as a deterministic continuity stress test.
**Plan, if execution-scoped:** none
**Critical unknown:** Whether the observed omissions come from captured audio, the SYCL backend, decoder settings, or independent 45-second inference windows—and which general-purpose inference strategy preserves the content without special-casing numbers.
**Shared scenario/input/environment:** Locally synthesized English counting at approximately one number per second with deterministic small timing variance; `small.en`; the LG Gram's production SYCL Level Zero server; the same request and transcript assembly boundaries as dictation.
**Budget:** Local machine only; no paid services; at most four focused hours and four implementation variants before reassessment.
**Stop condition:** A development fixture containing 1–150 passes exactly, followed by one untouched 1–500 holdout run; or all general-purpose variants fail and the evidence requires a model/architecture decision.
**Exploratory measurement oracle:** Extract spoken integers from the transcript and report missing, duplicate, substituted, and out-of-order values plus inference latency.
**Acceptance oracle:** The normalized integer sequence is exactly `1..150`, each once and in order. After that passes, generate and run `1..500` exactly once as a sealed holdout; it has the same exact-sequence threshold. Human authority: chat request, 2026-09-04.
**Valid while:** Model, prompt, server build, audio cadence, and dictation chunking match the recorded run metadata.
**Revalidate when:** Model, backend, prompt/context policy, chunk duration/overlap, or audio source changes.

## Guardrails

- Do not add number-specific production corrections or expected-sequence hints.
- Do not generate, inspect, or run the 1–500 audio until the 1–150 acceptance run passes.
- Preserve raw WAVs, raw transcripts, commands, timings, and comparison reports.
- Do not restart or modify the live service during exploratory runs.

## Approaches

| Approach | Why promising | Slice | What must be real |
|---|---|---|---|
| Current independent windows | Establishes the deployed baseline | Existing server request and 45-second assembly | Production SYCL server, model, prompt, and request fields |
| Decoder/prompt correction | The retained one-chunk failure may be decoding collapse rather than capture loss | Replay identical WAV with controlled request variants | Same WAV, model, SYCL backend |
| Contextual overlapping windows | Boundary audio and prior text may prevent omissions without delaying the full session | Add bounded overlap and reconcile duplicated text | Real window boundaries and production inference |
| Final whole-session verification/recovery | A second-pass view may recover suspicious online chunks | Decode or reconcile retained session audio at stop | Real retained WAVs and model/backend |

## Shared decision contract

| Measurement | Type | Threshold / authority | Collection method |
|---|---|---|---|
| Numeric continuity | Acceptance | Exact `1..N`; user request | Deterministic transcript normalizer and comparator |
| Audio continuity | Exploratory | No missing/corrupt source interval | WAV duration, sample count, silence/energy inspection |
| Generality | Acceptance | No number-sequence production special case | Diff review and non-numeric regression tests |
| Latency | Exploratory | Record; retain warm-server UX unless correctness requires otherwise | Monotonic wall-clock timing |
| Long-session durability | Acceptance | Every input window reaches final ordered transcript or remains recoverable | Session manifest and integration test |

## Harness self-test

| Fixture | Expected | Observed | Evidence |
|---|---|---|---|
| Exact `1..10`, mixed digits/words | pass | pending | pending |
| Missing, duplicate, substitution, and reordering cases | fail with correct diagnostics | pending | pending |

## Results

### Round 2026-09-04 — retained single-WAV inspection

- Run evidence: retained session `20260904T221510.218557Z-2753419`, one 28.36-second PCM s16le/16 kHz/mono WAV, RMS 817, 355-character transcript.
- Oracle status: excluded from numeric acceptance. Later inspection established that
  this retained session contains the spoken diagnostic request, not the user's
  1–150 count. It cannot corroborate numeric omissions.
- Proxies/fakes used: none; this was real microphone audio through the production path.
- Result: useful only as a one-WAV durability observation. No inference about the
  reported 1–150 loss is drawn from this session.

## Cross-approach conclusion

- What is now known: the retained diagnostic request is a non-empty one-WAV session;
  it is not the numeric failure and is not acceptance evidence.
- What remains open: Whether prompt/decoder configuration, SYCL-specific decoding, or `small.en` capacity caused the collapse, and which general-purpose recovery path clears the exact-sequence oracle.
- Approaches retained: all pending controlled replay.
- Next route: execute the controlled 1–150 prototype, then route the winning general-purpose behavior into an implementation plan.

## Disposition

- Reports/evidence retained at: this report and local ignored prototype artifacts.
- Code/resources: temporary harness and generated development audio may be deleted after implementation; retain failure WAVs in the existing session store.
- Validity: current.
- Downstream artifact update required: plan.
- Next route: prototype.

## Standalone closure

- Conclusion: Inconclusive
- Acceptance oracle and confirming evidence: pending
- Human evidence-review authority: chat request, 2026-09-04
- Human review: pending
- Closure: open
- Next route: pending prototype evidence

### Round 2026-09-04 — controlled development fixture

- Harness self-test: exact digit/word fixtures passed; missing, duplicate, substitution, reordering, and unrelated-token fixtures were all rejected with the expected first mismatch.
- Deployed baseline (`small.en`, SYCL, production prompt, 45-second independent windows): failed at 115. It duplicated 114, omitted 119, and collapsed the final 132–150 window to `1`.
- Same-model discriminators:
  - Beam search did not recover the collapsed final window.
  - Prior-text prompting recovered some windows but propagated earlier errors into later windows.
  - Timestamp-token decoding was slower and produced hallucinations/empty fragments.
  - Whole-file CLI long-form context reset 107 onward to 1 and later omitted content.
  - 25-second and 14-second windows still lost interior values.
  - Ten-second windows advanced by six seconds (four seconds of duplicated audio), context-free decoding, and overlap alignment produced exactly `1..150`.
  - The same 10/4 layout with the production vocabulary prompt also produced exactly `1..150`.
- Passing production-prompt evidence: local report `dev-150-20260904T223820.523025Z/report.json`; 26 real requests to the production SYCL server; exact 150-token sequence.
- Oracle status: acceptance/bound for the development fixture. The 1–500 holdout remains ungenerated and unrun.
- Result: positive for short context-free inference windows with redundant boundary audio and transcript overlap alignment.

### Cross-approach conclusion 2026-09-04

- What is now known: `small.en` can preserve the full development sequence on this LG Gram when no decode spans more than ten seconds and every boundary has four seconds of independent acoustic coverage.
- Negative results: retries, beam search, long-form context, timestamp decoding, and recursive prior-text prompts do not independently satisfy the continuity contract.
- Selected production shape: retain non-overlapping durable raw chunks; prepend the prior raw chunk's last four seconds for inference; merge adjacent fragments by their longest corroborated boundary token sequence; never feed generated transcript text back into later inference.
- Generality: the selected behavior uses only audio overlap and token equality. It contains no knowledge of numbers or the expected sequence.
- Remaining gate: implement the selected shape, rerun 1–150 through that implementation, then execute the sealed 1–500 holdout once.
- Next route: implementation plan, then autonomous implementation and close-out under the user's existing full-access instruction.

### Round 2026-09-04 — real capture continuity and durable-raw replay

- The existing recorder's actual stop-then-start lifecycle was exercised against a
  seeded continuous signal through a temporary PipeWire/PulseAudio null sink. Five
  consecutive handoffs each lost source time: 50.6–60.0 ms. Evidence:
  `capture-handoff-20260904T225916.497051Z/report.json`.
- A single `parecord --raw --format=s16le --rate=16000 --channels=1` process was
  then split by the Python consumer into six independently finalized six-second
  WAVs. Correlation mapped all five boundaries with 0.000 s gap/overlap error.
  Every chunk has a retained SHA-256 identity and the temporary audio module was
  unloaded. Evidence: `capture-continuous-20260904T230105.322387Z/report.json`.
- Disposition: reject frequent recorder process rollover. Promote one continuous
  PCM capture process with application-owned durable WAV segmentation.
- The initial overlap merger had a predeclared fixture matrix and required a unique
  anchor of at least two normalized tokens. Intentional repetition,
  competing anchors, one-token matches, punctuation/case, edge artifacts, and
  no-anchor input all pass their expected preserve-or-deduplicate outcomes.
- WAV preparation fixtures cover unchanged raw bytes, exact predecessor-tail
  composition, cleanup on success, and safe fallback for first, missing, corrupt,
  and format-mismatched predecessors.
- Decisive replay retained 26 independent non-overlapping raw WAVs, prepended only
  the predecessor's final four seconds for inference, used the production prompt
  and warm `small.en` SYCL Level Zero server, and assembled via the conservative
  merger. It returned exactly 150/150 ordered tokens. All raw hashes were unchanged,
  every derived file was removed, and the slowest request was 2.24 s. Evidence:
  `dev-prototype-150-20260904T230641.023370Z/report.json`.
- That report records the model and server binary hashes, CMake cache hash, device
  selection, prompt hash/request policy, audio hash, harness hash, repository
  revision, tool versions, and exact invocation.
- This run is explicitly marked `implementation_path=false`; it cannot authorize
  the sealed holdout. The final 1–500 audio remains ungenerated and unrun.

### Updated disposition

- Prototype conclusion after this round: capture and 6/4 inference are positive;
  unconstrained interior stitching remained subject to independent review.
- Remaining gate: independently approve the revised implementation plan, wire the
  same helpers into the daemon/session store under tests, pass an
  `implementation_path=true` 1–150 run, then run the sealed 1–500 holdout once.

### Round 2026-09-04 — adversarial repetition and source-position alignment

- Independent review rejected the unconstrained interior matcher with a distinct
  repetition counterexample: `please save the file then discuss budget` followed
  by a separate `please save the file after lunch` occurrence. The old matcher
  deleted `then discuss budget`. A second analogous deployment/monitoring case was
  added before trying replacements.
- Strict suffix→prefix deduplication never deletes unique edge text and preserves
  both counterexamples, but replay of the retained development transcripts fails at
  expected value 24 and yields 188 numeric tokens. It cannot meet the exact oracle.
- Timestamped overlap-save was tested on the selected 10-second windows. It emitted
  no text-deletion decisions, but request 6 took 11.47 s and request 8 stuck the
  server socket until the 90-second watchdog restarted it. The 26-window run could
  not complete. Evidence:
  `round3-alignment-20260904T232321.074041Z/failure-report.json`.
- The selected source-position matcher uses the already-known window durations and
  five-second overlap. It accepts exactly one ≥4-token anchor only when token ordinal
  positions on both adjacent transcripts map into the same shared source interval
  within 1.75 s. Otherwise it appends both sides. It does not request timestamps and
  does not inspect expected numeric values.
- Predeclared fixtures now include both reviewer deletion cases in addition to
  intentional repetition, competing anchors, one-token refusal, punctuation/case,
  edge artifacts, no anchor, and a hard empty-fragment boundary. All pass.
- Final decisive run `dev-prototype-150-20260904T233629.079298Z` returned exactly
  150/150 using 26 independent raw WAVs, five-second predecessor overlap, and
  production `small.en` SYCL requests. Every anchor contained 4–5 tokens. Mean
  inference was 1.82 s, median 1.52 s, and maximum 5.97 s—still below the six-second
  producer interval. All raw hashes were unchanged and no temporary files leaked.
- That report includes full porcelain-v2 repository status (including untracked
  files), harness and alignment-function hashes, model/server/build/audio/input
  hashes, exact invocation, and request/prompt identity.
- The previously cited retained microphone session is excluded: inspection showed
  it contains the diagnostic request, not the 1–150 count.
- The sealed 1–500 holdout remains ungenerated and unrun.

### Updated disposition after Round 3

- Prototype conclusion: positive for continuous capture, 6/5 no-timestamp
  inference, and source-position-constrained stitching.
- Next route: final authorized Phase 0 review round; implementation begins only on
  `APPROVE`.

### Round 2026-09-04 — non-overlapping temporal ownership

- Independent Round 3 review rejected token-ordinal overlap stitching. It is not
  source time and could delete a legitimate repeated phrase after a silent pause.
  The overlap plan was superseded before production implementation.
- A replacement probe partitions the original PCM exactly once at low-energy cuts
  near eight seconds, bounded to seven–nine seconds for normal chunks. It applies
  no transcript matching or deletion. A separate audio fixture places “we should
  deploy now” on both sides of an explicit three-second silent interval; both
  occurrences remain, and concatenated chunk PCM exactly matches source PCM.
  Evidence: `repetition-temporal-20260904T234935.785071Z/report.json`.
- The first corrected 1–150 replay decoded every number but sent a final 1.18-second
  digital-silence tail to Whisper, which hallucinated `you`. The production daemon
  already has an RMS-before-inference gate, so the probe was corrected to preserve
  the silent chunk as durable audio while skipping its inference.
- Decisive prototype report
  `dev-prototype-150-20260904T235558.569815Z/report.json` passes exactly: 150
  expected tokens, 150 output tokens, no unexpected tail. Its 20 chunk payloads
  concatenate to all 4,933,076 source PCM bytes with identical SHA-256
  `3db52280ddadb177c6556ce1ae8dd102aae19bfee73310d758e3cb38cd654726`.
  All raw WAV hashes are unchanged, all inference-padding files were removed, and
  the silent final tail was retained with RMS 0 and no server request.
- Nineteen non-silent beam-five requests had 2.81-second mean, 2.62-second median,
  and 7.62-second maximum latency. Total inference remained well below the
  154.16-second capture duration; the warm SYCL server stayed healthy.
- This remains prototype evidence (`implementation_path=false`). The replacement
  acceptance harness must mechanically exercise `Dictation` and `SessionStore`
  before it may emit an implemented report or unlock the 1–500 holdout.
- The 1–500 holdout marker and artifacts remain absent.

### Current disposition

- Prototype conclusion: positive for one-process capture, exhaustive non-overlap
  temporal ownership, silence-before-inference, temporary 0.5-second trailing
  silence, and beam five on the existing `small.en` SYCL server.
- Next route: independent review of
  [dictation-temporal-segmentation.md](../plans/dictation-temporal-segmentation.md),
  then test-first implementation if approved.

### Round 2026-09-05 — implemented-path development acceptance

- A first live implementation run was rejected as contaminated after the desktop
  daemon submitted a separate 36.5-second recording to the same warm server during
  the test. Capture attestation passed, but chunk 15 ended with a spurious `100`.
- Replaying that retained WAV in isolation reproduced the terminal token. Word
  timing showed that the minimum-energy cut had selected a short pause inside the
  next spoken number rather than the longer inter-utterance pause 0.52 seconds
  earlier. A test-first hybrid selector now preserves passing minimum-energy cuts
  and substitutes a sustained adaptive-noise-floor pause only when the original
  candidate is a short dip. It contains no expected-number or transcript logic.
- The boundary regression, all 109 dictation tests, Ruff, compilation, and shell
  syntax passed. Repartitioning the contaminated capture changed only the unsafe
  boundary and its immediate successor; all 4,949,760 PCM bytes remained exact.
- Fresh isolated report `dev-implemented-150-20260905T003922.364685Z` passed the
  exact oracle: 150 expected tokens, 150 output tokens, one recorder PID, all
  4,949,760 captured bytes reconstructed exactly once, no failed chunks, and one
  atomic paste. Nineteen inferences averaged 2.259 seconds and maximum queue depth
  was one.

### Round 2026-09-05 — sealed 1–500 holdout

- The holdout marker was written before generation and the sole permitted run was
  executed against the frozen implementation. Report:
  `final-500-20260905T004221.762535Z/report.json`.
- Capture and delivery remained lossless across the 521.087-second source: one
  recorder PID, 16,689,920 captured PCM bytes exactly reconstructed from 65 durable
  WAVs, no failed chunks, and one final paste. The inference queue peaked at four.
- The exact ASR oracle failed. The transcript matched through 191, diverged at 192,
  and contained 498 numeric tokens rather than 500. Inference averaged 4.653
  seconds and peaked at 18.853 seconds, while continuous capture remained
  independent and durable.
- This is a model/independent-window recognition failure, not an empty-WAV,
  microphone reacquisition, dropped-packet, storage, ordering, or paste failure.
  The failed report and all WAVs are retained unchanged. The holdout will not be
  tuned against or rerun.

### Current disposition after the holdout

- The long-recording reliability hypothesis is positive: the implementation
  retained and delivered every source byte across both acceptance runs.
- The exact 1–500 recognition oracle remains negative for independent 7–9-second
  `small.en` windows. The immutable result is not restated as a pass.
- Xavier Dedenbach accepted the measured model limitation on 2026-09-04 after
  reviewing the exact sequence: 483 values recognized correctly, 15 substituted,
  and only 270 and 416 omitted entirely. The byte-exact capture and delivery result
  is accepted for this implementation under R-4.
- Next route: validate the complete branch, prepare the PR, and merge to `master`.

### Round 2026-09-05 — post-holdout performance qualification

The sealed 1–500 holdout was not rerun or used as a tuning fixture. Performance
work returned to the development-only 1–150 source and preserved the accepted
capture architecture, model, accelerator, prompt, and transcript assembly.

- Extending the source-owned chunk ceiling to 16 seconds retained every PCM byte
  but omitted spoken value 119. A 12-second ceiling produced the same recognition
  omission. Both variants were rejected even though their capture attestations
  passed.
- Beam size three with the accepted 8/7/9-second boundaries returned 150/150, but
  repeated same-WAV timings under concurrent laptop load did not show a stable
  advantage over beam five. Beam five remains selected.
- whisper.cpp otherwise encoded its full 30-second context for each 7–9-second
  source window. Context 512 represents 10.24 seconds at 50 tokens per second and
  safely contains the maximum 9.0-second chunk plus 0.5-second inference-only
  silence. The daemon now validates that relationship and persists the value so
  retries, restart recovery, server requests, and CLI fallback use one profile.
- Report
  `performance-audioctx512/dev-implemented-150-20260905T025427.551040Z/report.json`
  passed exactly: 150 expected and 150 output tokens, one recorder PID, all
  4,948,480 PCM bytes reconstructed exactly once, no failed chunks, one atomic
  paste, and maximum queue depth one. Nineteen inference calls ranged from 1.454
  to 6.185 seconds (3.687-second mean) under system load near 11; the 154-second
  source completed in 160.319 seconds.

The 512-token profile is selected for the LG Gram. The portable default remains
empty/full-context, so CPU, CUDA, SYCL, and machines with different chunk profiles
must opt in explicitly. The immutable holdout artifacts remain unchanged.
