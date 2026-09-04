# Implementation Plan: Make long dictation sessions recoverable

**Status:** Complete
**Approval authority:** pre-approval by human, 2026-09-03 (auto-approved)
**Activation authority:** pre-approval by human, 2026-09-03 (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (estimate-size, 2026-09-03; one bounded reliability change in the desktop dictation component)
**Epic / execution unit:** none
**Linear project:** none — this personal fork has no configured tracker project
**Primary Linear issue:** none — no tracker identity is configured; workflow milestones are printed in chat
**Material cutover:** no — session persistence is local and additive; rollback is the prior daemon plus preserved session files
**Cutover plan dependency:** none
**Routine deployment phase:** none — the requested user-service relaunch is an implementation verification on the LG Gram
**Supersedes:** none
**Superseded by:** none
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp`
**Execution mode:** autonomous
**Phase 0 gate:** independent-review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** dedicated worktree `/home/linu_x/Documents/git/whisper.cpp-worktrees/dictation-long-session-recovery`
**Scope:** In: durable per-session WAV chunks, ordered transcript fragments/final transcript, per-chunk fault containment, restart recovery, graceful queue drain, configuration/docs/tests, and LG Gram relaunch. Out: model/backend changes, vocabulary/hallucination tuning already dirty in the live checkout, cloud storage, speaker diarization, audio editing, and automatic destructive retention.

**Post-completion amendment:** `dictation-session-atomic-delivery.md` supersedes
this plan's incremental live-paste behavior only. Persistence and recovery remain
unchanged; live sessions now paste their ordered successful fragments once at
stop.

## 1. Observable outcome and invariants

### End-to-end outcome

A 10–12 minute dictation can roll through approximately fourteen to sixteen 45-second chunks. Every finalized chunk is stored under one durable session directory before transcription. A failure, corruption, or missing WAV at arbitrary chunk N is recorded without stopping later chunks. Successful chunk transcripts are assembled in sequence into `transcript.txt`; missing or failed chunks are represented explicitly. Restarting the daemon discovers incomplete sessions, retries recoverable chunks without pasting stale text into the focused application, and regenerates the transcript from durable fragments.

### Persistence and recovery contract

- Session manifests are serialized under one process lock and committed with a same-directory temporary file, file `fsync`, `os.replace`, and parent-directory `fsync`; the last committed manifest is authoritative and orphan temporary files are ignored.
- Chunk states are `recording`, `queued`, `transcribing`, `complete`, `no_text`, `silent`, `ignored`, `failed`, `corrupt`, `missing`, or `exhausted`. `complete`, `silent`, `ignored`, `missing`, and `exhausted` are terminal.
- On startup, queued, interrupted, and retryable failed chunks are discovered and requeued. Missing audio becomes `missing`; repeated failures retain an explicit marker and the original audio for manual recovery.
- Recovery is capped at 32 chunks per daemon start. Recovered work never invokes interactive paste; a subsequent live recording pastes its assembled successful fragments once after finalization. Priority scheduling beyond the existing FIFO is deferred under R-1.
- SIGTERM stops/finalizes the active recorder, marks the session stopped, and waits at most five seconds for ready queue work. In-flight work remains recoverable. The systemd unit grants a 15-second stop bound; process-faithful stress simulation is deferred under R-3.
- Session audio is retained by default. This plan performs no automatic destructive retention; operators may delete completed session directories after verifying `transcript.txt`.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Recording continuity | At `MAX_RECORD_SEC`, the next recorder starts before the old one stops | Existing roll-order test plus a 16-chunk state-machine test | Preserve continuous rolling; persistence must not introduce a capture gap |
| Interactive paste | Successful live chunks are assembled in sequence | Session-atomic delivery tests | Paste once after finalization; recovered historical chunks never paste |
| Accelerator selection | CPU, CUDA, and SYCL use the same daemon | Existing runtime integration suite and live health check | No backend/build selection change |
| Server recovery | Server restart/retry and CPU fallback are bounded | Existing hang-recovery tests | A failed inference becomes a durable chunk failure rather than deleted audio |
| Local-only privacy | Audio currently remains local | Filesystem tests inspect paths/modes | Sessions remain local; no network destination beyond loopback whisper-server |
| Existing local work | Live checkout has unrelated uncommitted prompt/hallucination changes | Worktree/status comparison | Do not overwrite, stage, or include those changes in this execution unit |

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| A filesystem manifest plus numbered files can recover ordering without a database | Restart could reorder or lose chunks | Atomic JSON replacement; numbered chunk metadata and text fragments | Crash/reload a synthetic session after arbitrary state transitions | JSONL append log if whole-manifest replacement proves unsafe | Reload preserves every committed chunk and deterministic order after interrupted writes |
| One bad chunk can be isolated without stopping the worker | Later episode audio could remain queued forever | Catch exceptions around the entire queue item; persist terminal failure state | Inject empty, missing, corrupt, and exception failures at every index in a 16-chunk run | Supervisor/restart worker if containment cannot be local | Every later chunk reaches a terminal state and worker remains alive |
| Recovery can avoid duplicate interactive paste | Restart could type old episode text into the wrong window | Queue recovered work with `paste=False`; transcript file is source of truth | Restart fixture with pending chunks and mocked insertion | Manual recovery command only | Recovered chunks update files and never call insertion |
| A 12-minute session is operationally small | Persistence could create unreasonable disk pressure | 16 kHz mono S16 is about 23 MB per 12 minutes | Measure generated fixture size and document retention behavior | Shorter chunks/compression in a later plan | Default keeps audio explicitly; one 12-minute session remains under 30 MB excluding transcript |
| Recovery states and retries can be bounded | Old sessions could retry forever or starve live dictation | Persist attempts; three-attempt limit; 32-item recovery cap; live-priority queue | Restart/state matrix plus scheduling probe | Manual recovery-only command if automatic scheduling is unsafe | Exhausted/missing work terminates, recoverable work retries, and a live item schedules first |
| SIGTERM can hand off active/queued work before systemd kills the cgroup | Restart could corrupt the current chunk or strand queue state | Main-process-first SIGTERM, five-second drain, 15-second service bound | Process-faithful child with recording/transcribing/queued states | Immediate durable handoff with no drain | Exit under five seconds with recorder state queued, in-flight state recoverable, and every audio file present |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Filesystem recovery | Atomic metadata, stable numbering, missing-file handling | Disposable same-filesystem probe committed a 16-chunk JSON manifest with mode `0600`, left an invalid temporary file to model interruption, and reloaded all 16 committed chunks; numbered fragments rebuilt 16 ordered lines with `[missing audio: chunk 0007]` and `text-15` last | Supported | Exercises ext-family filesystem and exact atomic-replace/file layout primitives; production integration remains P1/P2 | High | None for Phase 0 |
| Per-chunk containment | Empty result vs exception/corruption behavior | Controlled 16-chunk simulation continued after empty chunk 7 but deleted its WAV; corrupt chunk killed the sole worker and blocked chunk 8 | Contradicted for corruption and recovery | Synthetic WAVs exercised the actual worker/delivery methods | High | Prove replacement behavior |
| No duplicate recovery paste | Recovered work must update files without interacting with focus | Existing insertion is isolated in `_insert_chunk`; the recovery queue can carry a `paste=False` flag and use the same delivery path while suppressing only that side effect | Supported | Interface-seam proof from current code; P2 tests must prove it under restart | Medium | Expand coverage in implementation tests |
| Disk bound | 720 seconds × 16,000 samples/s × 2 bytes ≈ 23.0 MB | A real 16 kHz mono S16 WAV fixture measured 23,040,044 bytes (21.97 MiB); target filesystem has 213 GiB free | Supported | One-session measurement excludes long-term accumulation, so default retention stays explicit/no automatic deletion | High | None for Phase 0 |

**Round summary:** Existing rolling is fast enough and an ordinary empty inference does not stop later work, but recoverability fails because every WAV is deleted and the worker has no outer exception boundary. Same-filesystem probes support atomic whole-manifest replacement, deterministic fragment rebuild, private file modes, and a 21.97 MiB twelve-minute footprint. The smallest credible design remains a local session store integrated with the existing serial queue; JSONL or a database would add lifecycle complexity without improving the bounded single-daemon case.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN

> **Phase 0 round 1 status — superseded by Round 2 evidence and explicit human approval.**

The reviewer accepted the local session-store direction but required production-shaped evidence for concurrent atomic updates, every-index failure containment, restart/no-paste and retry bounds, plus SIGTERM handoff. It also required the explicit state contract above, service-unit ownership, reproducible commands, and the corrected fourteen-to-sixteen chunk duration statement.

#### Round 2

##### Evidence inventory

The disposable probe was removed during clean-up-slop after its useful
assertions were promoted into maintained tests. Its observed results are
preserved below.

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Filesystem recovery | Serialized concurrent mutation; interruption before/after replacement; private modes; deterministic reload | Production-shaped atomic store probe retained all eight overlapping updates; SIGKILL before replace reloaded revision 0, SIGKILL after replace reloaded revision 1; manifest mode `0600` | Supported | Disposable implementation uses the exact planned lock/fsync/replace/fsync sequence on the target filesystem | High | Confirm in maintained P1 tests |
| Per-chunk containment | Four failure types at every possible N; later progress; failed audio preservation | 64-case matrix covered empty, missing, corrupt, and raised-exception failures at all 16 positions; the worker remained alive until normal sentinel, every later chunk completed, and every non-missing failed WAV remained | Supported | Prototype worker mirrors the planned outer guard but not the current production class | High | Promote matrix to maintained P2 tests |
| No duplicate recovery paste | Restart recovers historical work without focus side effects; live work remains interactive | Restart fixture recovered queued, transcribing, and retryable failed chunks with zero recovery paste calls; subsequent live work produced one paste call | Supported | Insertion is represented by a counted seam rather than X11 | High for policy seam | Confirm through `Dictation` integration test |
| Recovery states and retry bounds | Missing/exhausted terminal states; eligible retry set; scheduling fairness | Fixture recovered indexes 1–3, marked missing index 5, left attempt-exhausted index 4 terminal, generated six transcript lines, and scheduled live work before queued recovery | Supported | One session fixture; P2 adds multiple-session discovery and 32-item cap | Medium-high | Expand in implementation tests |
| Service shutdown/restart | Recording, queued, and transcribing states survive SIGTERM within service bound | Process child received SIGTERM and exited 0 in 0.032 s; recording became queued, queued remained queued, transcribing remained recoverable, and all three WAVs remained | Supported | Synthetic recorder/inference states rather than actual PulseAudio/SYCL; state handoff is the discriminating property | High for repository implementation | Confirm real service restart in P4 |
| Rolling order | Persistence reservation does not move recorder stop before next start | Probe order was `reserve-next`, `spawn-next`, `stop-old`, `queue-old` | Supported | Event model; existing production roll-order test covers real method calls | Medium-high | Extend existing test with durable paths |
| Disk bound | Twelve-minute S16 audio remains manageable | Round 1 measured 23,040,044 bytes (21.97 MiB), below the fixed 30 MB threshold | Supported | Exact capture format, one session | High | None |

**Round summary:** The expanded probe clears the snapshot, failure-containment, restart/no-paste, retry/scheduling, shutdown-handoff, roll-order, and disk assumptions with rerunnable evidence. The whole-manifest approach remains simpler than JSONL because one lock serializes all daemon mutations and the bounded file is small.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** PROCEED BY HUMAN DIRECTION

The independent reviewer requested a third, more elaborate prototype covering
multi-session queue pressure, process-faithful recorder shutdown, and additional
atomic interruption boundaries. On 2026-09-03 the human explicitly classified
those additions as nice-to-have and directed implementation to proceed without
them. The accepted Round 2 evidence covers the required product outcome: durable
chunks, arbitrary per-chunk fault containment, ordered transcript recovery, and
bounded service handoff.

> **Phase 0 status — approved by explicit human direction.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Rolling capture | `dictation.py` timer, recorder, staging, worker methods | Serial queue and global ordering gate | Extend rather than replace | `dictation.py` |
| Local configuration | `config.env`, `load_config()` | Shell-style machine-local overrides | Add session root and document retained-audio policy | `config.env`, README |
| Atomic response validation | `validate-server-response.py` | Small single-purpose Python helper | Use the same small-module style for session persistence | New `session_store.py` |
| Recovery entrypoint | Daemon startup/run path | No current recovery hook | Scan incomplete sessions before listener starts; enqueue without paste | `dictation.py` and session store |
| Tests | `test_runtime_integration.py` | `unittest`, temp directories, mocks, fake commands | Extend with filesystem/state-machine integration fixtures | Existing test module plus `test_session_store.py` |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0 | Validate atomic store and recovery-state assumptions | §2 A1–A4 | tests/temporary fixtures only | Plan active | Reload/interrupted-write tests fail because no store exists | Phase 0 evidence plus independent review | S |
| P1 | Durable session store with atomic manifest/fragments and deterministic transcript rebuild | §1 outcome | new `session_store.py`, tests | P0 | Store/reload/missing-WAV tests fail | Focused session-store tests and Ruff | S |
| P2 | Recorder/worker integration, per-item exception containment, stop/drain lifecycle, startup recovery without paste | §1 outcome/invariants | `dictation.py`, `whisper-dictation.service`, tests | P1 | 16-chunk fault matrix and restart tests fail | Fault matrix, worker-survival, restart/SIGTERM handoff, existing integration suite | M |
| P3 | User-facing configuration, recovery location/status, and operational documentation | §1 local privacy | `config.env`, README | P1–P2 | Documentation/config assertions fail | README/config checks plus diff validation | S |
| P4 | Production-equivalent LG Gram service-ready verification | User request | installed user units, real SYCL artifacts, durable local session fixture | P1–P3 | Existing health check does not prove persistence path | Full suite with real artifact links and validated user-unit template; ship-it-good performs the requested post-merge relaunch | S |

> **Phase P0–P4 status — complete.**

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** no repository line/branch coverage configuration exists. `env -u WHISPER_REPO_ROOT -u WHISPER_BUILD_DIR -u WHISPER_ACCELERATOR /home/linu_x/Documents/git/whisper.cpp/scripts/dictation/.venv/bin/python -m unittest scripts.dictation.test_runtime_integration` produced 30 passes plus one known worktree-only failure because the clean worktree lacks untracked model/build artifacts; the same base suite passes in the live checkout.
- **Coverage completion gate:** every behavior row below has a test, all existing tests pass in an environment with the current model/build artifacts, and no test or assertion is weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| Atomic durable session creation/reload | L4 filesystem integration, `test_runtime_integration.py` | Focused session-store class absent | Focused unittest class | Session id, manifest, modes, numbering |
| Final transcript survives one missing WAV | L4 filesystem integration | No transcript artifact exists | Focused missing-file/rebuild test | Ordered text plus explicit missing marker |
| Arbitrary chunk failure does not stop later chunks | L4 actual worker with synthetic WAVs | Corrupt chunk kills worker | Parameterized 16-chunk fault test | Every N and later terminal states |
| Restart resumes incomplete chunks without paste | L4 actual recovery/store with mocked inference | No recovery scan exists | Restart fixture | Pending/transcribing/failed states, retry budget, no insertion |
| User stop drains/finalizes | L4 queue/lifecycle integration | Worker is daemon-only and never joined | Stop/finalization fixture | Last chunk and final transcript |
| Existing dictation behavior | Existing L2/L4 suite | Baseline recorded above | Full unittest + live `check.sh` | No regression across CPU/CUDA/SYCL selection and server recovery |

### Realism target

Level 4 filesystem and process-faithful tests cover deterministic fault injection without waiting twelve minutes. Level 2 uses the real LG Gram service, model, SYCL server, and synthetic WAV chunks before relaunch. Live speech is useful acceptance feedback but is not the only evidence path.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| Sixteen chunks become one transcript | Session store, worker, transcript builder | Temporary filesystem, synthetic valid WAVs | Focused integration unittest |
| Existing model/backend remains healthy | Daemon scripts, SYCL binary/server, model | LG Gram | `bash scripts/dictation/check.sh` after restart |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| Empty inference at N | Backend returns no text | Preserve WAV/state; continue N+1 | L4 | Temp fixture | Parameterized unittest |
| Exception at N | Delivery/transcription raises | Mark failed; worker survives | L4 | Temp fixture | Parameterized unittest |
| Corrupt WAV at N | `wave.Error` | Mark corrupt; worker survives; continue | L4 | Temp fixture | Parameterized unittest |
| Missing WAV on restart | File deleted externally | Mark missing; retain other fragments and finalize transcript with marker | L4 | Temp fixture | Recovery unittest |
| Daemon exits mid-transcription | Manifest says transcribing | Requeue within three-attempt limit without paste | L4 | Temp fixture | Restart unittest |
| Manifest write interrupted | Temporary file remains | Last committed manifest loads | L4 | Temp fixture | Atomic-write unittest |
| Stop during queued work | Recording ended, worker busy | Drain and create final transcript | L4 | Temp fixture | Lifecycle unittest |
| Many stale sessions plus new speech | Recovery backlog | Queue at most 32 recovery chunks and prioritize live work | L4 | Temp fixture | Scheduling unittest |
| SIGTERM with active recorder/inference/queue | Service restart | Finalize recording, persist handoff within five seconds, recover on next start | L3/L4 | Child process plus LG user service | Process/service lifecycle tests |
| Backend unavailable | Restart and CPU fallback fail | Keep audio and failed state; later recovery remains possible | L4 | Temp fixture | Existing plus new failure test |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| None | All required acceptance evidence has an automated or service-level path | — | — | — |

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| Disposable Phase 0 probe and evidence JSON | Production-shaped atomic/fault/restart/SIGTERM discrimination | Evidence only; maintained tests replace it | Phase 0 approval | Deleted during clean-up-slop after evidence was recorded in this plan |
| Synthetic 16-chunk WAV/session fixture | Fast deterministic long-session and fault testing | High regression value | Validation | Promote into maintained tests |
| Worktree-local links to existing model/build/venv | Run unchanged integration checks without copying large artifacts | None | After merge | Leave untracked; remove with worktree |
| Fault-injection hooks | Exercise failures deterministically | Only valuable in tests | P2 refactor | Keep dependency injection in object boundaries; no production-active fault switch |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Amend plan / replace plan / supersede ADR |
|---|---|---|---|
| Whole-manifest atomic rewrite loses committed state | Interrupted-write test | Switch to append-only JSONL events plus derived snapshot | Amend plan |
| Graceful drain blocks hotkey listener or shutdown | Lifecycle timing test | Finalizer marker and bounded shutdown worker | Amend plan |
| Recovery requires pasting to preserve transcript | Restart test | Keep filesystem transcript authoritative and recovered paste disabled | Amend plan |
| Required behavior needs a database/service | Phase 0 evidence | Stop; architecture scope exceeds size S | Replace plan and require ADR |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Ten-to-twelve-minute episode | P2 synthetic sixteen-chunk run | Ordered transcript and terminal manifest states |
| Arbitrary Nth-chunk failure | P2 fault matrix | Later chunks complete for every injected index |
| WAV recovery | P1/P2 durable session store | Failed WAV remains; startup retry and missing-file marker tests |
| `transcript.txt` despite loss | P1 deterministic rebuild | Fragment-order/missing-WAV test |
| Preserve backend portability | P4 existing suite and health check | CPU/CUDA/SYCL selection tests plus LG SYCL check |
| Relaunch for testing | P4 user units | Active/success status and strict check output |

## 9. Primary Linear issue

- **Identity:** none — this personal fork has no configured Linear tracker identity
- **Reconciliation state:** no-op
- **Desired title:** Make long Whisper dictation sessions recoverable
- **High-level description:** Persist numbered WAV chunks and transcript fragments, contain arbitrary chunk failures, recover incomplete sessions after restart, and verify a sixteen-chunk episode without changing accelerator selection.

### Adapted children/subtasks

No tracker subtasks; the plan phases are one local execution unit.

## 10. Execution checklist and outcomes

- [x] Disposable Phase 0 prototype executed; no separate prerequisite prototype was triggered
- [x] No governing ADR acceptance rows; size-S no-ADR authority recorded
- [x] Tracker absence recorded; no external tracker mutation planned
- [x] Phase 0 evidence gathered
- [x] Phase 0 approved by explicit human direction after independent review
- [x] Pattern inventory reconciled after Phase 0
- [x] P1 session store complete with RED/GREEN evidence
- [x] P2 daemon recovery integration complete with RED/GREEN evidence
- [x] P3 configuration and documentation complete
- [x] P4 service-ready verification complete; requested live relaunch is the post-merge ship-it-good handoff
- [x] Happy-path integration passes
- [x] Edge-case matrix passes
- [x] Blast-radius invariants pass
- [x] No configured coverage threshold; requirement-to-test traceability complete
- [x] No test, assertion, threshold, or exclusion weakened
- [x] Human-only validation not required
- [x] Material cutover not required
- [x] Scaffolding disposition completed
- [x] Validation outcomes recorded

## 11. Validation outcome

- `ruff check` and `ruff format --check` passed on all four touched Python files.
- 33 runtime integration tests passed using the LG Gram's real CPU/SYCL build,
  model, and virtual-environment artifacts.
- Four durable-session tests passed, including failure at every chunk position
  from 0 through 15 and recovery without interactive paste.
- The substituted systemd user-unit template passed `systemd-analyze verify`.
- Validate-for-PR completed at light depth after one delta round with final
  verdict `APPROVE (risks accepted: R-1, R-2, R-3)`.
