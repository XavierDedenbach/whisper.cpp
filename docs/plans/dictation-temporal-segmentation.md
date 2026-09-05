# Implementation Plan: Lossless temporal segmentation for long dictation

**Status:** Active — validation complete; PR closeout pending
**Approval authority:** human pre-approval, 2026-09-04T22:40:30Z; reaffirmed by “make this your test and then close up on it”
**Activation authority:** same pre-approval; authorized phases: through-completion
**ADR(s):** [ADR-001 — Long-form dictation audio ownership](../decision/ADR_001_dictation_audio_ownership_2026-09-05.html) — Accepted by Xavier Dedenbach, 2026-09-04
**Size:** M — recorder lifecycle, durable storage, inference preparation, recovery, and real-system acceptance are coupled
**Epic / execution unit:** none
**Linear project / primary issue:** none — personal fork with no configured tracker policy
**Material cutover:** no — one local desktop service; rollback is configuration plus service restart
**Supersedes:** [dictation-overlap-continuity.md](dictation-overlap-continuity.md)
**Superseded by:** none
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp` (`XavierDedenbach/whisper.cpp`, integration branch `master`)
**Execution mode:** autonomous
**Phase 0 gate:** independent review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** current branch `codex/numeric-continuity-hardening`
**Scope:** Add an opt-in, single-process raw-PCM capture path that partitions every captured sample exactly once into durable low-energy-bounded WAVs; decode those independent chunks with optional temporary silence padding, a configured beam size, and a bounded encoder audio context; preserve ordinary ordered transcript concatenation; recover every queued chunk; prove the actual daemon path on 1–150, then run the sealed 1–500 holdout once; document, restart, validate, and merge. Out: transcript overlap/deduplication, expected-number logic, model training/change, recursive transcript prompting, whisper.cpp decoder changes, and changes to CPU/CUDA/SYCL runtime selection.

## 1. Observable outcome and invariants

On the LG Gram, Ctrl+Space starts one `parecord --raw` child and that child remains
the sole microphone owner until stop. The daemon owns segmentation: after seven
seconds it selects a low-energy cut nearest eight seconds, bounded at nine seconds.
When that minimum-energy point is only a short intra-utterance dip, a sustained
adaptive-noise-floor pause is preferred. The daemon wraps the original PCM bytes
in an independently valid durable WAV. Chunks are
transcribed while capture continues. The final partial is retained; a silent final
tail is classified before inference rather than sent to Whisper. For inference
only, the LG profile appends 0.5 seconds of digital silence, requests beam size
five, and limits encoder audio context to 512 tokens (10.24 seconds). The raw WAV
remains unchanged. Transcript assembly concatenates successful
chunks in source order and never matches or deletes text.

The result is one atomic paste when all session chunks are terminal, plus a durable
`transcript.txt`. A failure at any chunk leaves its WAV recoverable and never blocks
later chunks from being transcribed or represented in the transcript.

### Blast-radius invariants

| Contract | Required invariant |
|---|---|
| Audio ownership | Concatenating durable chunk PCM in index order reproduces every byte read from the recorder exactly once; no boundary restart, gap, overlap, or mutation |
| Text preservation | Assembly is ordered concatenation only; no transcript-derived deduplication, alignment, or expected-content logic |
| Silence | Quiet chunks stay durable and become terminal `silent`; they do not generate hallucinated text or masquerade as missing/empty capture |
| Failures | Missing/failed/corrupt chunks remain explicit markers in `transcript.txt`; later successful text remains available |
| Recovery | Startup discovers every eligible chunk, not only the first 32, and applies the session's persisted inference policy |
| Delivery | One paste occurs only after all earlier jobs and the final marker have been processed |
| Privacy | Logs contain session/chunk ids, lengths, timing, and status—not dictated text |
| Portability | Existing CPU, CUDA, SYCL, and non-`parecord` systems retain the legacy WAV recorder by default; all new behavior is opt-in |
| Holdout | 1–500 cannot be generated, inspected, or run until an actual production-path 1–150 report passes; it runs once |

### Additive configuration

| Key | Portable default | LG Gram robust profile |
|---|---:|---:|
| `CONTINUOUS_CAPTURE` | `0` | `1` |
| `STREAM_SEGMENT_TARGET_SEC` | `8` | `8` |
| `STREAM_SEGMENT_MIN_SEC` | `7` | `7` |
| `STREAM_SEGMENT_MAX_SEC` | `9` | `9` |
| `TRANSCRIPTION_TRAILING_SILENCE_SEC` | `0` | `0.5` |
| `WHISPER_BEAM_SIZE` | empty (existing greedy-first behavior) | `5` |
| `WHISPER_AUDIO_CTX` | empty (full model context) | `512` |

`CONTINUOUS_CAPTURE=1` requires `parecord`; validation fails clearly instead of
silently changing behavior. A reduced `WHISPER_AUDIO_CTX` is accepted only with
continuous capture, whose maximum padded chunk is bounded and validated.
Accelerator selection remains orthogonal and unchanged.

## 2. Phase 0 risk-reduction portfolio

| Assumption | Evidence | Result / threshold |
|---|---|---|
| One recorder can be partitioned without source-time holes | Real PipeWire/Pulse null-sink capture `capture-continuous-20260904T230105.322387Z`; six WAVs, five boundaries | Pass: all mapped boundary error 0.000 s. Stop/start control lost 50.6–60.0 ms and is rejected |
| Non-overlap temporal ownership cannot erase a repeated phrase | `repetition-temporal-20260904T234935.785071Z`; same four-word phrase on each side of explicit three-second silence | Pass: both occurrences retained; concatenated PCM hash equals source; no text matcher |
| `small.en` can decode bounded source-owned chunks | `dev-prototype-150-20260904T235558.569815Z`, production prompt/server/model, no expected-number logic in inference/assembly | Pass: exact 1–150; 150 expected and 150 output tokens; silent final tail retained and skipped by RMS |
| Inference keeps up | Same report, 154.16 s captured source | Pass: 53.33 s total, 2.81 s mean, 2.62 s median, 7.62 s max; server remained healthy |
| Raw preservation and temporary preparation are safe | Per-chunk before/after hashes, temporary-file inventory, source/chunk PCM reconstruction | Pass only if hashes match, reconstruction is exact, and no derived file remains |
| Implementation evidence cannot be forged by a flag | Existing probe inspection | Current `--implemented` label is rejected; replacement must invoke daemon capture/store and attest source hashes before unlocking holdout |

### Rejected alternatives

- Repeated stop/start recorder ownership: measured capture holes.
- Long independent 45-second decoding: model omissions and collapse.
- Audio overlap plus text matching: can delete a legitimate repeated phrase.
- Token timestamps: request latency exceeded the producer budget and one request
  held the server socket until watchdog recovery.
- Number-specific repair or training: violates generalization and the sealed holdout.

### Phase 0 review

**Verdict:** APPROVE — fresh-context reviewer `Popper`, 2026-09-05.

The review accepted the exact-partition, non-deleting architecture and found the
remaining uncertainty appropriate for test-first implementation. Mandatory test
coverage is: arbitrary odd-sized stream reads and EOF draining; exactly-once
stop/exit finalization; persisted beam, padding, and silence threshold with v1
defaults; persistence fault injection; raw-before-padding silence classification;
and an implemented-path attestation derived from actual recorder/store behavior.
No Phase 0 rerun is required. The holdout remains sealed.

## 3. Existing patterns and ownership

| Concern | Existing anchor | Planned owner |
|---|---|---|
| Recorder selection/lifecycle | `build_recorder_cmd`, `_start_recording`, `_finish_recording` | Add raw-stream command and lifecycle in `dictation.py`; leave legacy path intact |
| Temporal partitioning | Prototype low-energy splitter | New small production helper module with streaming state and WAV preparation |
| Durable audio/status | `SessionStore.ingest`, manifest v1 | Manifest v2 policy fields; v1 loads with legacy defaults |
| Inference retry | `_transcribe`, `_transcribe_server` | Configured beam is first request for this profile; existing retry/restart/CPU fallback remains bounded |
| Silence | `wav_rms`, `MIN_AUDIO_RMS` | Apply to original chunk before padding/inference |
| Transcript/delivery | `join_fragments`, `SessionPasteJob` | Preserve ordered concatenation and single final marker |
| Recovery | `SessionStore.recoverable(limit=32)` | Default `limit=None`; explicit positive limits remain available to tests/callers |
| Acceptance | `numeric_continuity_probe.py` | Null-sink playback through actual `Dictation` continuous capture and `SessionStore`; no real clipboard mutation |

## 4. Execution phases and test-first units

| Unit | Deliverable | First failing test | Green / regression gate |
|---|---|---|---|
| P0 | Accepted Phase 0 evidence | Reviewer challenges remaining invariant | Independent `APPROVE` |
| P1 | Streaming low-energy partitioner and temporary pad helper | Helper absent; exact byte partition, cut bounds, partial tail, silence and cleanup tests fail | Targeted helper tests; concatenated PCM byte equality |
| P2 | Opt-in single-process capture lifecycle | Existing timer restarts recorder | Fake/raw process tests: one PID, online chunk emission, stop drain, exact-boundary marker, partial final, unexpected-exit finalization |
| P3 | Persisted inference policy and delivery wiring | Current manifest lacks policy; raw path always decoded greedy-first | Manifest v2/v1 compatibility, beam/padding/raw-hash/exception-cleanup tests |
| P4 | Exhaustive recovery and nth-failure resilience | 40-chunk test returns 32 | 40+/100+ ordered recovery; randomized failed chunk leaves every later chunk represented |
| P5 | Actual-path 1–150 acceptance | Probe cannot attest production path | Generated one-per-second audio → null sink → actual daemon stream/store → captured paste exactly 1–150; raw/status/provenance checks all pass |
| P6 | Sealed final 1–500 and relaunch | Mechanically inaccessible before P5 | Sole run proves lossless capture/delivery; exact ASR result is recorded without alteration and receives an explicit human disposition; full suite/lint/check green; service restarted and healthy |
| P7 | Closeout into `master` | Branch-only state | Review, PR, CI, squash merge, post-merge service/check verification |

## 5. Test and recovery matrix

| Scenario | Expected behavior |
|---|---|
| Feed split across arbitrary byte boundaries | Odd bytes are retained until a full sample; final PCM is byte-identical and no sample duplicates/drops |
| Speech near target | Lowest-energy legal cut is selected; normal chunks stay 7–9 s |
| Short quiet dip competes with sustained pause | Preserve the existing low-energy cut when it lies in verified silence; otherwise choose the sustained pause and keep exact PCM ownership |
| User stop with short tail | Tail becomes one valid durable WAV; silent tail skips inference; voiced tail is decoded |
| Stop exactly at a cut | No empty WAV; ordered `SessionPasteJob` still finalizes/pastes |
| Recorder exits unexpectedly | Completed and partial audio are retained; session closes once; no automatic reconnect hides an unrecorded gap |
| Storage ingest fails at nth chunk | Gap marker is durable and later chunks continue in order |
| Transcription fails at nth chunk | WAV remains retryable; worker survives; later chunks process; transcript exposes gap |
| Daemon restarts with 100 queued chunks | All eligible chunks are scheduled in manifest order, subject only to retry limit |
| Padding success/error | Inference sees derived WAV; original hash is unchanged; derived file is removed in both paths |
| Repeated phrase separated by silence | Both utterances remain because assembly never deduplicates text |
| Legacy config / CPU / CUDA / SYCL | Existing recorder and runtime selection behavior remains unchanged |

### Commands and acceptance evidence

- Targeted and full Python tests: `python3 -m unittest discover -s scripts/dictation -p 'test_*.py'`
- Lint: `ruff check scripts/dictation`
- Runtime validation: `bash scripts/dictation/check.sh`
- Development oracle: implemented live-capture `1..150`, exact scorer, one number
  per approximately one second with deterministic small jitter.
- Final oracle: same implemented path and untouched generator with `1..500`, run
  only after the development report's attestation and score are verified. Exact
  recognition is reported separately from byte-exact capture/delivery; a non-exact
  ASR result requires explicit human disposition and may never be rewritten as a
  perfect result.

No assertion, fixture, or holdout expectation may be weakened to obtain a pass.

### Execution evidence and gate result

- The latest automated suite passed: 122 tests in 44.656 seconds;
  Ruff, Python compilation, shell syntax, and diff whitespace checks also
  passed. The earlier 109-test implementation run remains historical evidence.
- Final independent review found and closed two portability/lifecycle gaps:
  reduced audio context is now continuous-mode-only, and an unreaped rejected
  continuous-recorder child blocks startup retries. Both fixes were added
  test-first, and both delta reviewers returned `APPROVE` with no findings.
- The isolated implemented-path development run
  `dev-implemented-150-20260905T003922.364685Z` passed exactly: 150 expected and
  150 output tokens, one recorder PID, 4,949,760 captured bytes reconstructed
  byte-for-byte from durable WAVs, no failed chunks, and one atomic paste. Nineteen
  requests averaged 2.259 seconds; maximum queue depth was one.
- The sole sealed holdout run `final-500-20260905T004221.762535Z` failed its exact
  ASR oracle: the matched prefix ended at 191 and the transcript contained 498
  numeric tokens. It nevertheless passed every capture/durability check: one
  recorder PID, all 16,689,920 input bytes reconstructed exactly once, 65 durable
  chunks, no failed chunks, and one atomic paste. Maximum queue depth was four.
- The holdout marker was consumed before generation. The failed holdout is retained
  unchanged and will not be tuned against or rerun. The desktop daemon and warm
  SYCL server were restored after the run.
- The post-holdout performance run
  `performance-audioctx512/dev-implemented-150-20260905T025427.551040Z` kept the
  accepted 8/7/9-second segmentation, 0.5-second padding, `small.en` model, and
  beam five while limiting encoder context to 512. It passed 150/150 exactly,
  retained all 4,948,480 PCM bytes exactly once, had no failed chunks, pasted
  once, and held maximum queue depth to one. Under a system load near 11, its 19
  inference calls ranged from 1.454 to 6.185 seconds (3.687-second mean), and the
  154-second recording completed in 160.319 seconds. The sealed holdout was not
  rerun.

The initial P6 verdict was RED under the pre-run exact-ASR threshold. Xavier
Dedenbach explicitly accepted the observed result on 2026-09-04 as model-level
error outside the capture harness's responsibility and authorized push/closeout.
That decision is recorded as R-4; it does not alter the report. P6 is accepted and
P7 may proceed.

## 6. Rollout, rollback, and documentation

The portable template documents all new keys with conservative defaults. Only the
LG Gram's local config enables continuous capture, 8/7/9 segmentation, 0.5-second
padding, beam five, and a 512-token encoder audio context. Restart both user
services after merge, verify server device/health, daemon PID, hotkey indicator,
recent error-free journal, and `check.sh`.

Rollback is reversible: set `CONTINUOUS_CAPTURE=0`, clear beam/padding/audio-context
overrides, restore `MAX_RECORD_SEC=45`, and restart the dictation daemon. Retained
manifest-v2 sessions remain readable because recovery consumes persisted policy;
old manifest-v1 sessions retain legacy behavior.

## 7. Done contract

- Independent Phase 0 review approves this replacement plan.
- Every production change lands test-first; targeted and full suites plus Ruff pass.
- Actual continuous capture retains all source bytes once, uses one recorder process,
  and never reports a normal later chunk as an empty handoff WAV.
- Exact implemented 1–150 passes before the holdout marker exists.
- The first and only implemented 1–500 run preserves all captured PCM exactly
  once, retains every chunk, completes ordered delivery, and records the exact ASR
  score. Any ASR deviation has explicit human disposition in `docs/RISKS.md`.
- Portability defaults, README, and LG local profile are correct.
- The restarted service is healthy on the merged `master` revision.
- PR is merged into `master`; no unresolved review or CI failures remain.
