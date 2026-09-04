# Implementation Plan: Make rolling recorder handoffs self-healing

**Status:** Complete
**Approval authority:** pre-approval by human, 2026-09-04 (auto-approved)
**Activation authority:** pre-approval by human, 2026-09-04 (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (estimate-size, 2026-09-04; one bounded recorder-lifecycle fix in the desktop dictation component)
**Epic / execution unit:** none
**Linear project:** none — this personal fork has no configured tracker project
**Primary Linear issue:** none — no tracker identity is configured; workflow milestones are printed in chat
**Material cutover:** no — the change is local recorder lifecycle code with no data, configuration, or exposure mutation
**Cutover plan dependency:** none
**Routine deployment phase:** none — refreshing the requested LG Gram user service is local implementation verification
**Supersedes:** none
**Superseded by:** none
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp`
**Execution mode:** autonomous
**Phase 0 gate:** independent-review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** dedicated worktree `/home/linu_x/Documents/git/whisper.cpp-recorder-handoff`
**Scope:** In: recorder release/acquire ordering, startup validation, unexpected-recorder-exit recovery, accurate empty/short-chunk handling, lifecycle diagnostics, deterministic nth-chunk tests, and local LG Gram service verification. Out: Whisper inference behavior, accelerator selection, model/vocabulary changes, X-clip shutdown work, unrelated dirty edits in the live checkout, and a new continuous-stream audio architecture.

## 1. Observable outcome and invariants

### End-to-end outcome

A long dictation continues across repeated 45-second boundaries without accepting a recorder that immediately exits or fails to produce audio. The old recorder releases the source before its successor acquires it. If the active recorder later exits unexpectedly, the daemon preserves any partial WAV, starts a validated replacement promptly, and continues the same session instead of waiting out an empty 45-second slice. Successful chunks after an arbitrary recorder failure are still transcribed, pasted in order, and included in the durable `transcript.txt`.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Source ownership | Rollover briefly runs two `parecord` processes | Roll call-order test plus real LG Gram process trace | Serialize release then acquire; a bounded transition gap is allowed in exchange for reliable ownership |
| Recorder acceptance | `Popen` success is treated as recorder success | Early-exit fake recorder returns a process and empty path today | Accept only a live child that has written a WAV payload; retry boundedly |
| Mid-slice failure | Failure is discovered only when the next timer stages the file | Inject active-child exit at arbitrary chunk N | Preserve partial N, reconnect, and let N+1 and later proceed |
| Chunk ordering and recovery | Session store serializes jobs and marks failed chunks | Existing session-store and every-index worker tests | Retain ordering, durable WAVs, and explicit gap/failure markers |
| User stop | Final recorder is stopped, staged, and the session closes | Existing user-stop test | Do not reconnect after intentional stop; sub-minimum boundary tails do not claim a mic failure |
| Accelerator portability | The recorder feeds the same CPU/CUDA/SYCL inference path | Existing runtime-selection suite | No inference/backend/build/config changes |
| Existing local work | Live checkout has unrelated punctuation/prompt/debug edits | Main checkout status/diff | Preserve and exclude those edits from this PR |

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| Recorder overlap is unnecessary for usable continuity | Serial handoff could introduce unacceptable missing speech | Stop old with no tail sleep, then immediately spawn/validate next | Measure repeated real `parecord` release-to-first-payload transitions on this LG Gram without inspecting audio | Keep overlap but validate/retry if serial transition is too slow | All transitions succeed; median gap below 150 ms and maximum below 500 ms |
| A failed successor can be detected before it owns a 45-second slice | Empty chunks could still be accepted | Poll process liveness and WAV growth during bounded startup | Fake recorder exits immediately or writes no payload | Validate only child liveness if WAV growth is recorder-specific | Every early-exit/no-payload child is rejected with a diagnostic; healthy child is accepted within one second |
| An accepted recorder can fail later and be replaced without racing intentional rollover/stop | Recovery could duplicate staging, spawn after stop, or reorder chunks | Generation/current-process identity under the existing lock; one watchdog per accepted child | Inject exit at every N in a multi-chunk state-machine run | Periodic timer health check if watcher ownership is unsafe | Each failed process is finalized exactly once; no reconnect after user stop; all later chunks reach the worker in order |
| Existing durable storage already contains failures and later work independently | Recorder changes need not redesign persistence | Reuse `_stage_chunk`, `_submit_in_order`, and `SessionStore` | Existing every-index worker/store suites plus focused recorder-fault test | Add a new store state only if existing terminal states cannot express failure | Later chunks remain complete and `transcript.txt` retains successes around an explicit failed/short chunk |
| Recorder-only code is backend-neutral | Fix could accidentally bind behavior to SYCL or this laptop | Keep changes before `_transcribe` and avoid config/backend branches | CPU/CUDA/SYCL selection tests and diff inspection | Separate recorder adapter only if command semantics differ | Existing backend-selection and inference-recovery tests remain green |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Serialized handoff viability | Existing recorder finalizes promptly; successor can acquire the same default source; transition gap remains bounded | Twenty consecutive real default-source transitions on the LG Gram all succeeded. SIGINT-to-exit was 2–6 ms; exit/reacquire plus first payload produced stop-to-payload gaps of 68–109 ms (median 86.5 ms), each with at least 1,324 bytes. Probe files were deleted without content inspection | Supported | One PipeWire/PulseAudio host is the required target; portable recorder commands still need shared readiness semantics in tests | High | Regression tests |
| Startup failure detection | Immediate exit and no-payload live child can be rejected before acceptance | Against current `_spawn_recorder`, a fake child exiting 17 with `acquire denied` was returned in 0.7 ms with a 0-byte WAV; a child sleeping without output was returned alive in 2.0 ms with a 0-byte WAV. This reproduces both false-success classes and shows stderr is available after exit | Supported as defect and seam | Fake children isolate lifecycle behavior from the real audio stack; real healthy readiness was measured separately at 62–103 ms | High | Implement and test bounded rejection/retry |
| Mid-slice recovery | Current timer is the only post-start lifecycle observer; process identity can distinguish intentional and unexpected exits | Static trace confirms no liveness observer after `_spawn_recorder`. Existing lock protects `_record_proc`; rollover/finish can detach the current process before signaling, while a watcher may recover only when its process is still current and `_recording` remains true | Supported as executable ownership design | Race safety remains implementation work and must pass deterministic stop/roll/exit interleavings | Medium-high | Prove with RED/GREEN race matrix |
| Durable continuation | Store and worker isolate per-item failures and order later jobs | Baseline focused runtime/store command ran 25 tests in 25.169 s, all passing, including every-index worker failure containment. After linking this laptop's ignored build/model/venv artifacts into the isolated worktree, the complete runtime/session suite ran 47 tests in 32.605 s, all passing. Retained manifests for current daemon PID 3210309 contain contiguous chunk indexes 0–59; observed multi-chunk sessions and the live 58→59 rollover both completed in order | Supported | Recorder failure before ingestion still needs the focused composed test | High | Expand coverage in implementation |
| Backend neutrality | Recording methods contain no accelerator branch | Static call graph places recorder lifecycle before `_transcribe`; existing tests parameterize `build`, `build-cuda`, and `build-sycl`; proposed files/config contain no backend selector | Supported | CUDA is synthetic on this laptop, but recorder lifecycle is shared before inference | High | Full regression suite |

**Round summary:** Current retained sessions prove the persistence and ordered worker path is healthy, including multiple successful rollovers today. The uncovered failure window is entirely before ingestion: a successor is launched while the prior child still owns the source, accepted without readiness evidence, and then unobserved for up to 45 seconds. The actual LG Gram serialized transition is fast (86.5 ms median, 109 ms maximum across twenty handoffs), while controlled children prove current code falsely accepts both an immediate exit and a live process with no payload. The existing lock/current-process identity is sufficient to make intentional stop/roll exits distinct from an unexpected active-child exit. This supports bounded readiness, serial handoff, and active-child monitoring without a continuous-stream redesign.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** APPROVE

The independent reviewer accepted the recorder-only interpretation and found
the serialized handoff, data-bearing readiness check, and process-identity
watcher to be the simplest complete solution. Required implementation evidence
is: bounded retry/reaping/cleanup, stale-timer and stop/roll race coverage,
exact-once staging, failures at second/third/later positions, later transcript
recovery, and unchanged accelerator selection.

> **Phase 0 status — approved in Round 1.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Recorder process lifecycle | `dictation.py` spawn/stop/roll/finish methods | `start_new_session`, process-group SIGINT, lock-protected active process | Extend with bounded readiness and current-process watcher | `dictation.py` |
| Chunk durability/order | `_stage_chunk`, `_submit_in_order`, `session_store.py` | Atomic ingest plus sequence gate | Reuse unchanged unless a test exposes a missing marker | Existing modules |
| Timer and stop races | `_arm_max_timer`, `_on_max_record`, `_finish_recording` | Existing lock and process identity | Detach the intentional child before signaling; watcher recovers only the still-current child | `dictation.py` |
| Tests | `test_runtime_integration.py`, `test_session_store.py` | `unittest`, real subprocess fakes, mocks, temp paths | Extend existing runtime tests with call-order and nth-failure matrices | Existing test modules |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0 | Measure handoff/startup behavior and settle watcher ownership | §2 | plan evidence and disposable commands only | Active plan | N/A investigation | Independent Phase 0 review | XS |
| P1 | Serialized rollover plus validated, bounded recorder startup | §1 source/acceptance contracts | `dictation.py`, runtime tests | P0 | Roll-order and early-exit tests fail against overlap/unvalidated `Popen` | Focused recorder lifecycle tests | S |
| P2 | Unexpected active-child exit recovery without stop/roll races | §1 mid-slice/user-stop contracts | `dictation.py`, runtime tests | P1 | Nth-exit matrix loses the remainder or reconnects after stop | Focused race/fault matrix plus session-store tests | S |
| P3 | Accurate short/empty classification and diagnostics | §1 user-stop contract | `dictation.py`, runtime tests | P1–P2 | Tiny boundary tail reports mic failure | Focused delivery/notification test | XS |
| P4 | Regression, review, merge, and LG Gram service verification | User request | tests, PR, installed user service | P1–P3 | Pre-merge service runs old lifecycle | Full dictation tests, syntax/lint/format, service/check health, controlled rollover observation | S |

> **Phase P0–P4 status — complete.**

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** repository has no configured Python line/branch coverage gate. Before production edits, the focused runtime/session selection ran 25 tests in 25.169 seconds and the artifact-linked complete runtime/session suite ran 47 tests in 32.605 seconds; both passed.
- **Coverage completion gate:** every changed lifecycle behavior has RED/GREEN evidence, all dictation suites pass, and no assertion/test/exclusion is weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| Release old source before acquiring next | L5 call-order test, `test_runtime_integration.py` | Existing code records spawn before stop | Focused lifecycle class | Exact stop → spawn → install/stage order |
| Reject an immediately dead/no-payload child | L4 real fake process plus temp file | Current `_spawn_recorder` returns it | Focused startup test | Exit status, stderr detail, temp cleanup, bounded retry |
| Recover arbitrary active recorder exit | L4 state-machine/process seam | Current daemon waits until timer and accepts lost interval | Nth-failure matrix | Failures at second, third, and every later tested position; exact-once staging and ordered continuation |
| Do not reconnect after intentional stop/roll | L5 race/identity tests | Watcher prototype can race signal path | Focused stop/roll tests | No duplicate spawn/stage and no post-stop process |
| Tiny boundary tail is not a mic-permission error | L5 delivery test | Size check currently precedes duration check | Focused notification test | Short terminal state/message; true long empty capture remains explicit |
| Backend behavior remains portable | Existing L2/L4 runtime suite | Characterization | Full runtime/session suites | CPU/CUDA/SYCL selection and server recovery unchanged |

### Realism target

Level 4 real child processes and deterministic failure seams are the strongest repeatable tests for lifecycle ordering. A Level 2 probe uses the actual LG Gram PulseAudio/PipeWire `parecord` path and default source while inspecting only timing, process status, and byte counts. No automated test stores or transcribes private speech.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| Repeated healthy transitions produce payloads | Real `parecord`, default PipeWire/Pulse source, WAV files | LG Gram | Bounded transition probe; byte/timing inventory only |
| Long-session state remains ordered | Recorder seam, daemon lock/timer, session store, worker queue | unittest temp directories | Focused multi-chunk integration test plus existing store suite |
| Installed daemon uses healthy shared path | User units, SYCL warm server, actual config/build | LG Gram | `check.sh`, service health, controlled rollover status/manifest |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| Successor exits immediately | Source acquire/permission failure | Reject, diagnose, retry without accepting an empty slice | L4 | fake child | Focused unittest |
| Successor remains alive but writes no payload | Hung/misdirected recorder | Stop, reject, and retry within startup bound | L4/L5 | fake child | Focused unittest |
| Active recorder exits on chunk N | Mid-slice child failure | Preserve partial N, validated reconnect, later chunks proceed | L4 | deterministic process seam | Nth-failure matrix |
| User stop overlaps watcher | Intentional signal looks like failure | No reconnect; final chunk staged once | L5 | unittest synchronization | Focused race test |
| Timer rollover overlaps watcher | Child exits at boundary | One owner rotates/stages; no duplicate sequence | L5 | unittest synchronization | Focused race test |
| All startup attempts fail | Recorder unavailable | End recording, close session, retain prior chunk, notify diagnostic | L4/L5 | fake child | Focused failure test |
| Final tail below minimum | Stop immediately after rollover | Ignore as too short; do not claim mic permission failure | L5 | unittest | Focused delivery test |
| Later persistence/inference failure | Existing downstream boundary | Continue and rebuild transcript with marker | L4 | existing store tests | Existing suites |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| None | Process, byte, manifest, service, and deterministic failure evidence cover acceptance; the user can perform ordinary speech testing after relaunch | — | — | — |

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| Disposable real-recorder timing files | Measure source transition latency without content inspection | None | End of Phase 0 | Delete exact temporary directory |
| Fake recorder process in tests | Reproduce early and nth exits | Ongoing regression value | Validation | Retain only if compact; otherwise use existing mock seam |
| Review evidence JSON under `/tmp` | Evidence for independent validation | One-run audit | After merge | Leave outside repository |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Disposition |
|---|---|---|---|
| Serialized transition exceeds 500 ms | Real timing probe | Keep overlap but gate successor readiness; retry serially after an overlap failure | Amend within plan |
| Child watcher cannot distinguish intentional exits safely | Deterministic race matrix | Replace watcher with frequent health timer under one generation token | Amend within plan |
| `parecord`, `pw-record`, and `arecord` require incompatible readiness rules | Command-specific tests | Validate liveness plus a conservative WAV-header threshold shared by all | Amend; replan only if adapter split becomes material |
| Reliable rolling requires one continuous raw stream | Repeated real failures after bounded handoff fix | Stop; scope a separate architecture decision/prototype | Replan/ADR if M+ |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Second and third chunks are delivered reliably | P1–P2 | Early-exit and nth-failure matrices plus real handoff probe |
| Empty WAV is not misdiagnosed as microphone permission | P1/P3 | Startup readiness and short-tail classification tests |
| Recorder source is released before reconnect | P0/P1 | Process timing trace and exact call-order test |
| Later chunks and transcript survive one failed WAV | P2 | Existing store matrix plus composed recorder-fault test |
| Fork remains CPU/CUDA/SYCL portable | P4 | Existing runtime-selection/inference suites; no backend-conditioned recorder code |
| Ship and relaunch on LG Gram | P4/ship-it-good | Merged PR SHA, active service, healthy warm server, check output |

## 9. Primary Linear issue

- **Identity:** none — this personal fork has no configured Linear tracker identity
- **Reconciliation state:** no-op
- **Desired title:** Prevent empty rolling chunks by validating and recovering recorder handoffs
- **High-level description:** Serialize source release/acquire, reject failed recorder startups, recover an unexpectedly exited active recorder, preserve ordered durable chunks, and verify the common path on the LG Gram.

### Adapted children/subtasks

No tracker subtasks; the plan phases form one local execution unit.

## 10. Execution checklist and outcomes

- [x] No separate prototype triggered; bounded Phase 0 process probes are sufficient
- [x] No governing ADR rows; size-S no-ADR authority recorded
- [x] Tracker absence recorded; no external tracker mutation planned
- [x] Phase 0 evidence gathered
- [x] Phase 0 independent review approved
- [x] Pattern inventory reconciled after Phase 0
- [x] P1–P3 implemented with RED/GREEN evidence
- [x] P4 full tests and merge-ready validation complete; the installed LG Gram
      service is refreshed and checked immediately after the squash merge
- [x] Happy-path and edge-case matrices pass
- [x] Accelerator portability invariants pass
- [x] No configured coverage threshold; requirement-to-test traceability complete
- [x] No test, assertion, threshold, or exclusion weakened
- [x] Human-only validation not required
- [x] Material cutover not required
- [x] Temporary scaffolding disposition completed
- [x] Validate-for-PR outcome recorded

### Implementation evidence

- **RED:** eight focused lifecycle tests failed against the prior implementation,
  exposing successor-before-release ordering, acceptance of dead/no-payload
  children, absence of active-child recovery, sessions left open after exhausted
  rollover startup, and tiny tails misreported as microphone failures.
- **GREEN:** all 35 focused `HangRecoveryTests` pass, including a real fake-child
  process that exits twice before producing payload, a no-payload live child,
  shutdown signal races, failures at chunk indexes 1, 2, 7, and 15, stale
  watcher/timer ownership, and an empty failed chunk followed by a successful
  durable transcript chunk.
- **Regression:** the complete dictation runtime and session-store suite passes:
  61 tests in 34.879 seconds. Existing CPU/CUDA/SYCL selection cases remain
  unchanged and green.
- **Static checks:** Ruff lint and format checks and `git diff --check` pass for
  the touched Python files.

### Validate-for-PR outcome

**Round 1 (initial), depth light, Sentry skipped, Hotfix yes:** Reviewer A
passed form/structure and all touched-file Ruff/diff gates with no findings.
Reviewer B passed functional fidelity after independently running 31 lifecycle,
4 session-store, and 22 runtime-selection tests. It identified two `should-do`
test-evidence gaps: forced watcher-vs-roll/stop contention and proof that the
last good WAV is retained when every successor start fails.

The fix-owner added both winner orders for watcher-vs-roll and watcher-vs-stop,
verified those four races over 50 consecutive repetitions, and converted the
exhausted-start case to a real `SessionStore` assertion over the retained WAV
and stopped manifest. The expanded 35-test focused suite and 61-test unified
suite pass. **Final verdict: APPROVE (should-do applied).**
