# Implementation Plan: Continuous capture with corroborated overlap

**Status:** Superseded after Phase 0 review
**Approval authority:** pre-approval by human, 2026-09-04T22:40:30Z (auto-approved)
**Activation authority:** pre-approval by human, 2026-09-04T22:40:30Z (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (estimate-size, 2026-09-04)
**Epic / execution unit:** none
**Linear project:** none — personal fork; no repository/workspace tracker policy
**Primary Linear issue:** none — local personal-fork execution; `gen-tickets` no-op
**Material cutover:** no — local desktop dictation process only; no production data, traffic, tenant, or customer exposure
**Cutover plan dependency:** none
**Routine deployment phase:** none; the explicitly requested local service restart is acceptance verification on this laptop
**Supersedes:** none
**Superseded by:** [dictation-temporal-segmentation.md](dictation-temporal-segmentation.md)
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp` (`XavierDedenbach/whisper.cpp`, integration branch `master`)
**Execution mode:** autonomous
**Phase 0 gate:** independent-review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** current feature branch `codex/numeric-continuity-hardening`
**Scope:** Add an opt-in single-process PCM capture path with application-owned durable WAV segmentation, raw-audio overlap, and corroborated fragment stitching; remove the 32-chunk recovery ceiling; retain the existing model/prompt/accelerator selection; verify 1–150; run the sealed 1–500 holdout once; update LG Gram local settings; restart, validate, and merge. Out: number-specific correction, model training, model change, recursive transcript prompting, core whisper.cpp decoder changes, and changes to CUDA/SYCL/CPU selection.

## 1. Observable outcome and invariants

### End-to-end outcome

Before it was superseded, this plan proposed that one `parecord` process emit a continuous signed-16/16 kHz/mono PCM stream. The daemon would finalize non-overlapping six-second WAVs without restarting that process. For inference only, each chunk after the first would be prefixed with five seconds from its durable predecessor, producing at most eleven seconds of audio. Adjacent transcripts would be merged only at one unique ≥4-token anchor whose ordinal positions map into the same shared source-time interval; uncertain boundaries would preserve both sides. Its planned oracle was exact 1–150 development recognition followed by exact 1–500 holdout recognition. This overlap-and-stitch path was not implemented; the replacement temporal-segmentation plan records the actual holdout result and its human disposition.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Durable audio | Raw recorder WAVs are retained independently | Real continuous-capture probe + session-store preservation/failure tests | One recorder stream; independently valid WAVs; raw bytes remain unchanged; overlap WAV is temporary and reproducible |
| Atomic delivery | One final clipboard paste after ordered terminal chunks | Existing atomic-session integration tests | Assembled text is overlap-deduplicated before the same single paste |
| Failure recovery | Retained queued/failed WAVs rebuild `transcript.txt` | Recovery tests | Recover every eligible chunk, including sessions over 32 chunks |
| Gaps | Failed/missing chunks remain visible markers | Existing failure transcript tests | Never stitch across a non-complete or non-contiguous boundary |
| Vocabulary | Static vocabulary prompt applies to every request | Probe with production prompt | Preserve it; do not feed transcript output back as prompt |
| Portability | CPU/CUDA/SYCL and overlap-disabled configs work | Runtime selection suite + overlap=0 characterization | Additive config only; default overlap remains 0 and legacy file recorder remains available for existing installs/non-`parecord` capture |
| Privacy | Successful dictated content is absent from journal | Log assertions/manual journal check | Log lengths/timings only, never transcript text |

## Optional prototype evidence

- **Form:** separate prerequisite artifact
- **Status:** evidence accepted against the user-defined development oracle
- **Uncertainty resolved:** whether retained audio, decoder retries/context, or boundary strategy can preserve repetitive long-form content with `small.en` on SYCL
- **Target environment/boundary:** production warm SYCL server on the LG Gram, generated 16 kHz mono PCM speech, production vocabulary prompt
- **Budget:** local-only; no paid resources
- **Stop/exit criteria:** exact 1–150 before implementation; untouched exact 1–500 after implementation
- **Report:** [dictation-numeric-continuity.md](../prototypes/dictation-numeric-continuity.md)

| Variant | Evidence | Disposition |
|---|---|---|
| 45-second independent decode | Duplicate/omission and final-window collapse | Reject |
| Beam/timestamps/whole-file/prior-text context | Incorrect, slow, or error-propagating | Reject |
| 14-second window / 4-second overlap | Interior omissions | Reject |
| 10-second window advanced six seconds | Exact 1–150, both without and with production prompt | Promote through TDD |

- **Artifact disposition:** maintain the generic generator/oracle as an opt-in hardware integration test; generated WAVs/reports remain local and ignored.
**Findings update:** implement one persistent raw PCM recorder, six-second durable chunks, and five-second predecessor tails; do not use generated text as model context and do not restart the recorder at chunk boundaries.

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| A0: Capture can create six-second durable chunks without losing source time | Process handoffs create unrecoverable holes before inference | Existing stop/start; one continuous PCM stream with durable segmentation | Correlate independently finalized chunks against a seeded signal for both real capture lifecycles | Inspect stop→ready wall-clock gaps and compare a single continuous capture | No absolute boundary error above 20 ms; reject any lifecycle that fails |
| A1: 10/4 acoustic redundancy clears model collapse without number-aware logic | Approach cannot satisfy the user's oracle | Audio overlap; stronger model | Exact development probe and diff inspection | Production-prompt and no-prompt runs | Both runs exactly `1..150`; no expected-sequence logic in production |
| A2: SYCL inference keeps up with six seconds of new audio | Queue grows without bound on long sessions | Short windows; model change | Per-request timings in accepted probe | Service journal under final run | Every accepted development request <6 s; no stuck request |
| A3: Source-position-constrained alignment removes edge artifacts without deleting distinct repetitions | Text loss moves from ASR to assembly | Known window geometry + token ordinal; token timestamps; strict refusal | Predeclared repetition/deletion/ambiguity/punctuation/edge/no-anchor fixtures plus exact 1–150 | Strict suffix-prefix and timestamped overlap-save | Deduplicate only one unique ≥4-token anchor mapped into the same five-second source interval; otherwise preserve both fragments; exact 1–150 |
| A4: Temporary overlapped input is reconstructible after restart | Crash loses context or corrupts retained WAVs | Rebuild from predecessor; persist derived WAV | Byte-preservation, policy-version, and cleanup tests | Inject missing/corrupt/mismatched predecessor | Raw bytes unchanged; safe current-only fallback; no temp leak; v1 means overlap disabled |
| A5: More frequent chunks remain fully recoverable | A crash after ~3 minutes strands later audio | Exhaustive startup recovery | Queue 40+ chunks, recover and complete all | Random nth failure matrix | Every eligible chunk returned in order and transcript rebuilt; no implicit cap |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| A1 | Same model/backend/prompt; exact sequence; general algorithm | Prototype runs `dev-150-20260904T223604.483340Z` (no prompt) and `dev-150-20260904T223820.523025Z` (production prompt) each returned exactly 150 ordered tokens | Supported | Synthetic eSpeak voice is a proxy for human speech; the reported human count was not retained as an identifiable WAV | High | Confirm through production helper path |
| A2 | Warm server request latency stays below audio advance | Production-prompt accepted run: all 26 requests completed; slowest observed request 4.35 s for six seconds of new audio | Supported | One 153-second fixture; 500 holdout will exercise sustained run | Medium-high | Observe queue/timings on holdout |
| A3 | Edge hallucinations removable; repetitions retained | Initial prototype used an unconstrained interior anchor | Superseded/unsupported after Round 2 counterexample | It can delete a distinct repeated phrase | Low | Round 3 position/time discriminator |
| A4 | Derived input can be rebuilt and raw WAVs preserved | Repository inspection confirms predecessor raw WAVs are durable and jobs run in manifest order | Partial | Production overlap helper not yet present | Medium | RED/GREEN WAV composition/fallback/cleanup tests |
| A5 | No hidden recovery ceiling | `SessionStore.recoverable(limit=32)` and daemon startup explicitly cap recovery at 32 | Contradicted in current code; fix is bounded | Static evidence is exact | High | RED/GREEN 40+ chunk test and remove startup cap |

**Round summary:** The selected decoding shape passes the bound development oracle and stays ahead of recording. Implementation risk is limited to WAV composition, conservative overlap assembly, and recovery pagination; each has a direct test-first unit.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN — fresh reviewer `Halley`, 2026-09-04

Required rerun scope: measure real six-second recorder handoff continuity; specify and exercise ambiguous merger fixtures; persist overlap policy with backward compatibility; select exhaustive recovery; attach complete run provenance; then validate 1–150 through independently retained raw chunks and the actual production helpers. The 1–500 holdout remains excluded.

> **Phase 0 status — revision in progress (Round 1).**

#### Round 2

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| A0 | Real recorder lifecycle; independently valid WAVs; source-time continuity | `capture-handoff-20260904T225916.497051Z`: stop→start lost 50.6–60.0 ms at all five boundaries. `capture-continuous-20260904T230105.322387Z`: one raw `parecord` stream split into six WAVs mapped all five boundaries at exactly 0.000 s error | Stop/start rejected; continuous segmentation supported | Temporary null sink removes microphone acoustics but exercises the real PipeWire/PulseAudio process and byte stream | High | Implement the passing single-process shape; preserve legacy mode when opt-in is disabled |
| A1 | Independently retained raw chunks; production prompt/server/model; exact general stitcher | `dev-prototype-150-20260904T230641.023370Z`: 26 non-overlapping raw WAVs; 25 predecessor-tail inputs; exact 150/150 | Supported | eSpeak remains a voice proxy; the human report motivates the defect but has no correctly identified retained numeric WAV | High | Repeat through wired daemon/store helpers before holdout |
| A2 | Warm SYCL remains ahead of six seconds of new audio | Same decisive run: maximum request 2.24 s, total 49.08 s | Supported | One 154-second development run | High | Observe queue depth/timing on implemented acceptance |
| A3 | Ambiguity does not cause deletion | Predeclared fixtures pass: suffix-prefix, intentional repetition, one-token refusal, competing-anchor refusal, punctuation/case, edge artifacts, no anchor; accepted run's anchors were all unique and 2–4 tokens | Supported | Text fixtures cannot cover all prose ambiguity; conservative no-merge fallback preserves content | Medium-high | Production unit tests must use the same fixture matrix and enforce no merge across status gaps |
| A4 | Raw immutability; derived cleanup/fallback; restart-stable policy | Composition fixtures pass success/first/missing/corrupt/mismatch and cleanup. Decisive run reports unchanged hashes for all 26 raw WAVs and zero temp leaks | Supported for helper shape | Manifest policy is a design assertion until store tests are GREEN | Medium | Persist manifest v2 `transcription_overlap_seconds` + stitch policy; interpret v1 as overlap 0; test recovery |
| A5 | Startup recovery is exhaustive | Current cap is exact static evidence; selected behavior is `limit=None` exhaustive with explicit limits retained only for callers/tests | Design selected; current code still fails | No runtime evidence until implementation | Medium | RED/GREEN 40+ chunk recovery and random nth failure matrix |

##### Provenance

The decisive report contains the exact command; source-audio, harness, model,
server-binary, and CMake-cache SHA-256 values; Git revision/dirty state; active
systemd server status; configured Level Zero/SYCL device; prompt hash and complete
request policy; tool versions; per-input hashes/durations/text/timings; stitch
anchors; and cleanup/raw-integrity results. The capture reports retain source and
chunk identities plus process commands and correlation mappings. The final holdout
marker remains absent.

##### Selected contracts before implementation

- Recording: continuous PCM mode is enabled only when overlap is configured and a
  compatible `parecord` raw stream is available. Default overlap `0` keeps the
  existing recorder path on CPU/CUDA/SYCL laptops and other recorder binaries.
- Manifest: version 2 persists `recording_chunk_seconds`,
  `transcription_overlap_seconds`, and stitch policy `source-position-anchor-v1`; version 1
  loads as overlap disabled.
- Stitching: merge only contiguous complete chunk indices from a manifest carrying
  the overlap policy. Require one unique normalized anchor of at least four tokens
  whose ordinal positions in the adjacent known-duration windows map into the same
  shared source-time interval within 1.75 s. Ambiguity/no anchor appends both sides.
  A failed/missing chunk resets the merge run.
- Recovery: startup discovery is exhaustive (`limit=None`). An explicit positive
  limit remains available for focused callers, but the daemon supplies no cap.
- Holdout: prototype reports cannot unlock 1–500. Only a passing report marked
  `implementation_path=true` may create the one-shot holdout marker.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN — fresh reviewer `Kuhn`, 2026-09-04

The reviewer accepted continuous capture, 6/4 inference, throughput, composition,
recovery direction, and holdout seal, but produced a deterministic distinct-repeat
counterexample for the unconstrained interior matcher. Required rerun scope is A3,
its assembled 1–150 result, and corrected provenance only.

> **Phase 0 status — Round 2 revision complete; Round 3 follows.**

#### Round 3

##### Evidence inventory

| Candidate | Safety discriminator | Development oracle | Throughput/liveness | Disposition |
|---|---|---|---|---|
| Strict suffix→prefix, no unique-token deletion | Preserves both reviewer counterexamples and all unmatched text | Fails at expected value 24; 188 numeric tokens due duplicated/error edges | Reuses fast no-timestamp responses | Reject: safe but not accurate enough |
| Token-timestamp overlap-save with fixed source-time ownership | Has no text deletion/matching mechanism | Could not complete: only 7/26 windows returned | One request took 11.47 s; window 8 left the server socket stuck for 90 s and triggered the existing watchdog restart | Reject: violates six-second producer budget and liveness |
| Source-position-constrained unique anchor | Preserves both distinct-repeat counterexamples; passes intentional repetition, competing-anchor, one-token, punctuation/case, edge-artifact, no-anchor, and empty-boundary fixtures | Final candidate `dev-prototype-150-20260904T233629.079298Z`: six-second chunks/five-second overlap, exact 150/150, all anchors 4–5 tokens | Mean 1.82 s, median 1.52 s, maximum 5.97 s (<6 s); no timestamp mode; server stayed healthy | Select for TDD |

The selected matcher compares only adjacent fragments. It estimates each matched
token's position from token ordinal and actual inference-window duration, translates
the predecessor positions into the known five-second shared interval, and accepts a
cut only when exactly one ≥4-token candidate maps on both sides within the
predeclared 1.75-second tolerance. This rejects the reviewer's distinct occurrences
because their predecessor phrases are outside the shared source interval. It is a
content-agnostic geometry rule and contains no expected-number knowledge.

The decisive report records full `git status --porcelain=v2 --untracked-files=all`,
the harness SHA-256, the selected alignment function SHA-256
`38ec2b63c0d2a3dd0064f452ae05b518df5434436c5f6e43bd48756ac1de563b`
(harness `f59ecc6ea85fae570ecaccc6f4c7320e4ae5f52a9e15453230e9350221162215`),
model/server/build/audio/input hashes, exact command, prompt/request policy, raw
immutability, and zero temporary leaks. The retained session
`20260904T221510.218557Z-2753419` is explicitly excluded from numeric evidence: it
contains the diagnostic request, not the 1–150 count. Timestamp failure evidence is
`round3-alignment-20260904T232321.074041Z/failure-report.json`.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REPLAN — independent reviewer `Carson`, 2026-09-04

The reviewer accepted the single-recorder capture direction and the 6/5 decode
feasibility evidence, but rejected token-ordinal overlap stitching. Token ordinal is
not source time, so a legitimate phrase repeated after a silent boundary can be
mistaken for duplicated overlap and deleted. No production implementation was
started from this plan. The replacement plan uses exhaustive, non-overlapping
source-time ownership and performs no transcript matching or deletion.

> **Phase 0 status — evidence ready for final authorized review round.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Rolling recorder | `_arm_max_timer`, `_roll_recording_locked`, `_stage_chunk` | `MAX_RECORD_SEC` currently stops then restarts the recorder | Add opt-in one-process raw PCM reader that finalizes every `MAX_RECORD_SEC`; retain legacy path at overlap 0 | `dictation.py` + local config |
| Inference input | `_deliver_chunk` calls `_transcribe(job.wav_path)` | Raw path is passed directly | Build/deallocate temporary predecessor-tail input around `_transcribe` using persisted session policy | `session_store.py` helper + `dictation.py` call site |
| Transcript rebuild | `join_fragments`, `completed_text`, `_rebuild` | Ordered concatenation with gap markers | Merge only contiguous complete fragments when manifest overlap >0 | `session_store.py` |
| Recovery | `recoverable(limit=32)` at daemon init | Durable attempts/status transitions | Make default exhaustive (`None`) and preserve optional explicit positive limit | `session_store.py`, `dictation.py` |
| Real-system oracle | prototype probe + production server | Deterministic TTS, exact sequence score | Reuse production overlap/merge helpers; add raw-chunk mode and final seal | `numeric_continuity_probe.py` |
| Portability/docs | `config.env`, README table/check flow | Optional environment keys | Add overlap key; leave default 0; document 6/5 robust profile | config/README |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0 | Independent review of prototype-fed risk evidence | plan grant | plan/prototype | — | n/a | reviewer `APPROVE` | S |
| P1 | Continuous PCM capture and exact durable six-second segmentation | §1/A0 | `dictation.py`, runtime integration tests | P0 | New continuous-stream lifecycle/partial-final tests fail because only process rollover exists | targeted capture tests + full dictation suite | S |
| P2 | Versioned conservative overlap assembly | §1/A3/A4 | `session_store.py`, `test_session_store.py` | P1 | New manifest-v2/fixture/gap tests fail under `join_fragments` | targeted store suite + full dictation suite | S |
| P3 | Temporary predecessor-tail composition, raw preservation, fallback/cleanup | §1/A4 | `session_store.py`, `dictation.py`, tests | P2 | New waveform/path lifecycle tests fail because raw path is used directly | targeted waveform/delivery tests + full suite | S |
| P4 | Exhaustive crash recovery beyond 32 chunks | §1/A5 | store/daemon/tests | P3 | 40-chunk default recovery test returns only 32 | all 40 returned/processed in order; nth-failure matrix | S |
| P5 | Implemented development acceptance and docs/config | user oracle | probe, config template, README, local config | P1–P4 | Prototype is not authorized as implementation evidence | wired raw-6/overlap-4 path exactly `1..150`; all regressions pass | S |
| P6 | Sealed final holdout and live relaunch | user final-test authority | local evidence, systemd/check | P5 | n/a; one-shot acceptance gate | sole `1..500` run exact; service/check green; journal contains no transcript | S |

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** repository exposes no Python coverage configuration or threshold. Baseline is the complete `unittest` dictation suite plus Ruff; no percentage will be invented.
- **Coverage completion gate:** all requirement-linked tests plus the complete suite pass; no test/assertion/exclusion is weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| One recorder stream emits exact durable chunks and a final partial | L4/process-boundary runtime integration | continuous-stream mode/helper absent | targeted fake-stream lifecycle tests then full discovery | exact boundaries, clean EOF, user stop, unexpected exit, no empty chunk |
| Merge corroborated overlap, preserve repetition | L2/unit `test_session_store.py` | targeted new tests show duplicates or lost repeated words | targeted module then full discovery | Every merge branch incl. no match |
| Compose predecessor tail without changing raw | L2/real WAV `test_session_store.py` | helper absent/current path lacks prefix | targeted waveform assertions | Success, missing/corrupt/mismatch, cleanup |
| Use derived path for inference and delete it | L4 boundary + real WAV | delivery mock receives raw path | runtime integration test receives temporary path and observes cleanup | success and exception paths |
| Recover >32 | L2/store | default recovery returns 32 of 40 | returns all 40 ordered | limit=None and explicit limit |
| Exact long continuity | L2/real model/server/TTS | retained baseline report fails at 115 | probe raw-6/overlap-4 returns exact 150 | full numeric oracle |
| Holdout generalization | L2/real model/server/TTS | not run before implemented GREEN | one sealed exact 500 run | full holdout oracle |

### Realism target

Level 2: production binaries, model, SYCL device, HTTP server, prompt, 16 kHz PCM, production WAV/assembly helpers, and synthetic speech. The user's human-mic report supplies defect motivation but is not treated as retained acceptance evidence. Implemented acceptance routes deterministic playback through PipeWire and the actual continuous recorder path.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| 150 exact | TTS → durable raw chunks → overlap WAV → HTTP SYCL → store assembly → oracle | LG Gram | implemented `numeric_continuity_probe.py --implemented --durable-chunks --chunk-seconds 6 --overlap-seconds 5` |
| Atomic paste remains | chunk worker → store → clipboard dispatch | test fake X11 + service | existing atomic-session tests + `check.sh` |
| Live server stays ahead | warm server + six-second cadence | LG Gram | per-request report and journal timings |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| Continuous capture boundary | exactly six seconds in one PCM stream | two valid contiguous raw WAVs; recorder PID unchanged | runtime integration | fake PCM process + real capture probe evidence | targeted test |
| Stop on partial chunk | user stop before next boundary | partial valid WAV finalized once; no empty successor | runtime integration | fake PCM process | targeted test |
| Unexpected stream exit | recorder dies mid-session | completed/partial raw audio retained, manifest terminalized, user notified | runtime integration | fake PCM process | targeted test |
| First chunk | no predecessor | transcribe raw only | unit/integration | temp WAV | targeted test |
| Missing/corrupt/mismatched predecessor | overlap unavailable | log/fallback to current raw; retain both | unit | temp WAV | targeted test |
| Transcriber exception | temporary file exists | cleanup derived file; raw remains retryable | integration | mock error + real WAV | targeted test |
| No textual anchor | ASR overlap differs | join without deleting either fragment | unit | strings | targeted test |
| Edge hallucination before/after anchor | partial word decoded badly | discard only uncorroborated overlap edge | unit | strings | targeted test |
| Intentional repeated phrase | repeated tokens are real | longest multi-token anchor preserves repetition | unit | strings | targeted test |
| Failed/missing middle chunk | non-contiguous complete runs | never merge across marker; later text retained | unit/integration | session store | existing + new test |
| 40+ queued chunks after crash | startup recovery cap | every eligible WAV queued in order | integration | temp store | targeted test |
| Overlap disabled | existing laptop/profile | byte-for-byte old raw inference and normal joining | regression | unit | targeted + full suite |
| CPU/CUDA/SYCL selection | different laptop backend | selection unchanged; overlap config orthogonal | regression | fake runtimes | full runtime suite |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| None | Final quality and sustained-duration gates use deterministic local playback and exact machine scoring | n/a | n/a | n/a |

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| `numeric_continuity_probe.py` | Generate/score deterministic long speech and seal holdout | High regression value for model/backend/config changes | validation | retain as opt-in integration tool |
| Local generated WAVs/reports | Raw evidence | Reproducibility on this machine | after merge | retain under user data, not Git |
| Temporary overlap WAV | Inference-only composition | None after request | every request finally block | delete always |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Disposition |
|---|---|---|---|
| Continuous raw capture fails on a non-`parecord` host | startup capability/error | leave overlap at 0 and use unchanged legacy WAV recorder; document capability | compatibility fallback, not silent mode change |
| Production-helper 150 fails | exact mismatch report | compare prototype/helper boundaries and fix general composition/assembly | amend and rerun P4 |
| Any request takes ≥6 s repeatedly | timings/queue growth | evaluate 8/4 or stronger/faster model in a new prototype | replan; do not silently accept backlog |
| Exact alignment deletes real repeated speech | focused regression | require timed alignment or alternate decoder metadata | replan |
| Sealed 500 holdout fails | immutable final report | preserve failed holdout; create a new development fixture and a new sealed holdout before further tuning | stop merge and reopen evidence route |
| SYCL/server fails | report/journal | existing SYCL restart then CPU fallback retains WAV | existing recovery contract |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Use counting as deterministic test | P5/probe | exact 1–150 report |
| Iterate without seeing final test | final marker + P6 ordering | final marker absent until P5 pass; one final report |
| Final count to 500 | P6 | exact 1–500 report |
| Do not lose middle chunks | P1–P5 | continuous capture, overlap assembly, gap tests, 150 exact |
| Recover long recordings after crash | P3–P4 | retained raw WAVs + 40+ recovery test |
| Preserve same model and acceleration | P5 | report says `small.en`, SYCL server; runtime selection suite |
| Relaunch ready service | P6 | active units + `check.sh` |

## 9. Primary Linear issue

- **Identity:** none — this personal fork has no Linear tracking policy or linked issue
- **Reconciliation state:** no-op
- **Desired title:** n/a
- **High-level description:** n/a

### Adapted children/subtasks

None; progress milestones are printed locally as not posted.

## 10. Execution checklist and outcomes

- [x] Required prototype evidence accepted and folded into plan
- [x] No-ADR authority and size recorded
- [x] Human approval packet includes explicit no-tracker identity
- [ ] Phase 0 independent review approved
- [ ] Pattern inventory reconciled after Phase 0
- [ ] Every behavior-changing unit has RED then GREEN evidence
- [ ] Production-equivalent 1–150 passes
- [ ] Sealed 1–500 holdout passes on its sole run
- [ ] Full tests and Ruff pass without weakening
- [ ] Blast-radius invariants pass
- [ ] Local service is restarted and healthy
- [ ] Validation outcomes recorded
- [ ] PR merged to `master`
