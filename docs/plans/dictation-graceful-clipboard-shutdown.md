# Implementation Plan: Graceful Dictation Clipboard Shutdown

**Status:** Complete
**Approval authority:** pre-approval by user, 2026-09-03 (auto-approved)
**Activation authority:** pre-approval by user, 2026-09-03 (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (estimate-size, 2026-09-03; one component, two production files plus focused tests, low-risk lifecycle change)
**Epic / execution unit:** none
**Linear project:** none
**Primary Linear issue:** none — this personal fork has no configured Linear tracker identity
**Material cutover:** no — local daemon lifecycle only; no production data, configuration, exposure, or coordinated rollout
**Cutover plan dependency:** none
**Routine deployment phase:** none
**Supersedes:** none
**Superseded by:** none
**Target repo:** XavierDedenbach/whisper.cpp fork
**Execution mode:** autonomous
**Phase 0 gate:** independent-review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** dedicated worktree
**Scope:** Make the dictation daemon release its tray and clipboard helper promptly on SIGTERM without changing recording, transcription, accelerator selection, or durable chunk recovery. Preserve unrelated live-checkout edits.

## 1. Observable outcome and invariants

### End-to-end outcome

`systemctl --user restart whisper-dictation.service` stops the prior daemon without the 15-second timeout or SIGKILL, including after clipboard insertion, and the relaunched daemon becomes ready with the configured SYCL server path.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Clipboard insertion | `xclip` owns CLIPBOARD and `xdotool` sends Ctrl+V | Focused command/lifecycle tests | Run `xclip` in the foreground as an owned child, verify ownership, and retain it until replacement or shutdown |
| Durable shutdown | SIGTERM finalizes active audio and drains queued work for up to five seconds | Existing unit assertion plus focused shutdown test | Also stop tray and clipboard resources after the durable handoff |
| Recorder safety | `KillMode=mixed` lets Python finalize the recorder before systemd kills residual children | Existing service-unit test | No unit kill-mode change |
| Backend portability | Runtime selects CPU, CUDA, or SYCL independently of insertion | Existing runtime-selection suite | No backend/build/config changes |

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| Python remains alive because detached tray resources are not stopped | Clipboard-only work would not fix restart latency | Explicit `TrayIndicator.stop`; force interpreter exit | Inspect pystray thread construction and current shutdown path | Reproduce service stop after clipboard is absent | pystray uses non-daemon thread and shutdown omits `stop`, or another bounded cause is found |
| A directly owned foreground `xclip` can preserve reliable paste readiness | Paste could race or lose clipboard contents | `xclip -quiet` plus ownership probe; `-loops`; process-group tracking | Exercise foreground xclip on an isolated X selection and read it immediately | Read xclip manual/source behavior | Exact bytes are available and the direct process remains controllable |
| Keeping `KillMode=mixed` protects active audio while explicit cleanup removes the timeout | Unit-only fix could truncate the last WAV | Inspect recorder process/session and systemd policy | Existing long-session shutdown tests | Production service restart | No kill-mode change; resource cleanup occurs after recording finalization and bounded queue drain |
| Aggregate cleanup fits the service deadline | A worst-case child could still trigger systemd SIGKILL | One overall work deadline plus bounded clipboard/tray teardown | Sum declared worst-case waits and test the budget constants | Two installed service restarts | Failure-path budget is below 14 seconds, leaving at least one second below `TimeoutStopSec=15` |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Detached tray blocks interpreter exit | `run_detached` creates non-daemon backend/setup threads; current `TrayIndicator.stop()` must release both | Installed pystray source and an actual-display probe showed the current `run_detached()`/`stop()` pair takes about 5 seconds and leaves a non-daemon setup waiter | Refuted for the original remedy | Cause is supported, but simply calling the existing stop method is insufficient | High | Replace `run_detached()` with one owned runner around `Icon.run()` and prove bounded join |
| Foreground xclip is directly manageable and ready after stdin closes | `-quiet` avoids forking; selection bytes can be served; terminate remains available | Local X11 probe on the isolated SECONDARY selection successfully returned exact bytes from foreground `xclip`; manual states `-quiet` runs foreground while default `-silent` forks | Supported | Probe did not type into a real application; production-equivalent restart will validate composed lifecycle | High | Prove command order, readiness, replacement, and termination in tests |
| Explicit cleanup is safer than `KillMode=control-group` | Recorder is a cgroup child but separate process group; mixed mode preserves app-owned finalization | Recorder uses `start_new_session=True`; SIGTERM handler calls `_finish_recording` then drains; service unit intentionally asserts `KillMode=mixed` | Supported | Static process-policy evidence; existing recovery suite covers finalization semantics | High | Preserve unit and run regression suite |

**Round summary:** The timeout has two bounded lifecycle causes: pystray's detached non-daemon Xorg thread is not stopped, and default xclip forks an unowned clipboard process. The clipboard approach is viable, but the proposed call to the existing tray stop path is insufficient because `run_detached()` leaks a second setup waiter.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN

The reviewer required an owned `Icon.run()` runner, real recorder finalization/reaping evidence, and a declared aggregate shutdown budget. No production edit was made before this verdict.

> **Phase 0 round 1 status — revisions required and carried into Round 2.**

#### Round 2

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Owned tray runner exits cleanly | One non-daemon runner calls `Icon.run()`; `Icon.stop()` releases backend/setup work; join stays below 0.5 seconds | Actual-display dependency probe used installed pystray 0.19.5 with one named runner; stop plus join completed in 0.000 seconds and left no new non-daemon thread | Supported | Dependency-level probe mirrors the intended implementation but is not yet integrated into `TrayIndicator` | High | Prove integrated start/stop with focused test and rerun actual-display probe |
| Foreground xclip is directly manageable and ready after stdin closes | `-quiet` avoids forking; selection bytes can be served; terminate remains available | Round 1 isolated X11 probe and xclip(1) manual evidence remain valid | Supported | Probe uses SECONDARY to avoid disturbing the user's CLIPBOARD; command semantics are otherwise identical | High | Prove command order, readiness, replacement, and cleanup in tests |
| Recorder can finalize and reap before Python cleanup | Controlled child follows the production process-group/SIGINT path; resulting WAV is readable; PID is gone | Real-process probe launched a separate-session recorder that wrote WAV frames and handled SIGINT; `graceful_stop_recorder()` returned in 0.088 seconds, produced a readable 1004-byte WAV, and reaped the PID | Supported | Synthetic recorder exercises the same signal/process/WAV boundary, not PipeWire itself; existing retained-audio behavior and installed restart cover the real recorder | High | Add permanent real-process regression plus ordering assertion |
| Aggregate shutdown fits `TimeoutStopSec=15` | Recorder worst case is bounded; queue drain shares an overall deadline; clipboard/tray teardown is independently bounded | Design fixes durable-work deadline at 12 seconds from shutdown entry; clipboard TERM/KILL totals at most 0.5 seconds and tray join at most 0.5 seconds, yielding a declared bound below 13 seconds plus ordinary instruction overhead | Supported as executable budget | Constants and deterministic budget assertion remain implementation work; real installed restarts remain final evidence | High | Add budget test and verify two restarts below 14 seconds |

**Round summary:** The corrected tray lifecycle is demonstrated on the actual display, the existing recorder helper finalizes/reaps a real controlled WAV child, and a shared 12-second durable-work deadline plus at most one second of owned-resource cleanup leaves more than one second of margin below systemd's 15-second timeout.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN

The reviewer accepted foreground xclip and the normal tray path, but required production-recorder evidence, explicit synchronization with concurrent finalization, one propagated deadline, and a bounded tray failure fallback.

> **Phase 0 round 2 status — revisions required and carried into Round 3.**

#### Round 3

##### Normative shutdown contract

- `SHUTDOWN_BUDGET_SEC = 13.0`, strictly below the service's 15-second stop timeout.
- `SHUTDOWN_RESOURCE_RESERVE_SEC = 1.0`; durable recording/queue work receives the first 12 seconds from one absolute deadline captured at shutdown entry.
- `RECORDER_STOP_FLUSH_MSEC` is clamped to 500 ms. Recorder flush, SIGINT wait, SIGKILL/reap fallback, WAV stability, and queue drain all cap waits against the same durable-work deadline; the kill/reap path reserves its final 250 ms before optional stability waiting.
- One `_finalize_lock` serializes `_roll_recording()` and `_finish_recording()`. Every finalizer receives a 12-second work deadline, shutdown publishes its shared deadline before a timed lock acquisition, and shutdown proceeds to resource cleanup when that deadline is exhausted.
- Clipboard cleanup allows up to 250 ms after TERM and 250 ms after KILL, capped by the aggregate time remaining after lock acquisition and durable work. The directly retained `Popen` handle prevents PID-reuse signaling.
- The tray runner and bounded stop helper are daemon threads. Startup uses an explicit backend-ready/error handshake; normal stop invokes `Icon.stop()` and joins the runner, while shutdown caps the 500 ms tray allowance by the aggregate time remaining.

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Owned tray runner exits or cannot hold shutdown | Normal `Icon.stop()` reaps runner; delayed backend is bounded; no non-daemon thread remains | Actual-display installed-pystray probe stopped and joined the owned runner in 0.000 seconds with no residual non-daemon thread. A delayed-backend prototype returned after 0.500 seconds with only daemon runner/stopper threads remaining | Supported | Delayed path is a protocol-faithful thread prototype; integrated class tests remain implementation work | High | Add integrated normal, repeated, startup-failure, and delayed-stop tests |
| Foreground xclip is directly manageable and ready after stdin closes | `-quiet` avoids forking; selection bytes can be served; retained handle is safe to terminate | Round 1 isolated X11 probe and xclip(1) manual evidence remain valid | Supported | Probe uses SECONDARY to avoid disturbing user clipboard; exact command semantics are shared | High | Add command-order and failure-path tests |
| Recorder finalizes/reaps before Python cleanup | Real installed recorder follows production arguments/process group; overlap race is understood; lock contract closes it | Installed `parecord` probe with production 16 kHz mono WAV arguments stopped through `graceful_stop_recorder()` in 0.202 seconds, produced a readable 5164-byte WAV `(16000 Hz, 1 channel, 2-byte samples, 2560 frames)`, and left no PID. A controlled concurrency probe reproduced the current defect: shutdown returned while an already-running finalizer was blocked (`shutdown_returned_before_finalizer=True`) | Supported with defect reproduced | Real recorder proves signal/WAV/reap behavior; controlled overlap uses a barrier to deterministically expose coordination rather than relying on timing | High | Implement `_finalize_lock` and permanent real-process/overlap tests |
| Aggregate shutdown fits `TimeoutStopSec=15` | One entry deadline bounds every wait; configuration cannot expand it; resource cleanup has reserved time | Fixed budget prototype asserts 13.0-second total, 12.0-second durable-work window, and 1.0-second cleanup reserve composed of xclip TERM 0.25 + KILL 0.25 + tray stop/join 0.5. The current unbounded/serial overlap is removed by the normative contract above | Supported as executable contract | Deterministic production-constant test and installed restart timing remain implementation/final validation | High | Add fake-clock failure-path test and verify two installed restarts below 14 seconds |

**Round summary:** Round 3 adds the missing installed `parecord` evidence, deterministically reproduces the concurrent-finalizer defect, makes synchronization and deadline propagation normative, clamps configuration, and bounds a blocked tray backend without leaving a non-daemon thread capable of holding Python alive.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** APPROVE

The independent reviewer approved the foreground clipboard owner, owned daemon tray runner, finalization lock, and single 13-second shutdown contract. Remaining command-order, overlap, deadline, and installed-restart checks are implementation acceptance work.

> **Phase 0 status — approved in Round 3.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Clipboard insertion | `dictation.py::_insert`; xclip(1) | Default xclip forks and xdotool pastes | Add one owned foreground clipboard process with bounded readiness and cleanup | `dictation.py` |
| Shutdown | `dictation.py::shutdown`, `main`; `dictation_indicator.py` | Durable handoff then `SystemExit`; tray exposes `stop()` | Stop owned UI/clipboard resources in a guaranteed cleanup block | `dictation.py` |
| Service policy | `whisper-dictation.service`; integration test | `KillMode=mixed`, 15-second bound | Preserve policy | Characterization only |
| Regression location | `test_runtime_integration.py::HangRecoveryTests` | Existing process/restart and unit tests | Extend focused lifecycle tests | Test module |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0 | Validate tray/xclip lifecycle and recorder-safety assumptions | §2 | plan and local process probes | Active plan | N/A investigation | Independent Phase 0 review | XS |
| P1 | Owned foreground xclip lifecycle with readiness, replacement, and bounded cleanup | §1 clipboard invariant | `dictation.py`, runtime tests | P0 | New test expects `Popen(... -quiet ...)`, ownership probe, and retained handle | Focused lifecycle tests plus runtime suite | S |
| P2 | Shared-deadline recorder synchronization plus owned tray/clipboard cleanup after durable handoff | §1 shutdown invariant and Round 3 contract | `dictation.py`, `dictation_indicator.py`, runtime tests | P1 | New overlap/budget/tray tests expose early return, unbounded cleanup, and non-daemon thread | Real recorder/overlap tests, integrated actual-display probe, focused shutdown tests, and full dictation suite | S |
| P3 | Prove service-ready lifecycle and define the post-merge local activation check | User ship request | focused service policy/tests; post-merge local user service and journal | P1–P2 | Prior journal shows 15-second timeout/SIGKILL | Pre-merge lifecycle suite; ship-it-good handoff installs merged code and verifies two restarts | XS |

> **Phase P0–P3 status — complete; post-merge installation/restart remains the ship-it-good handoff.**

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** no configured Python line/branch coverage gate. Combined runtime/session suite ran 47 tests: 46 passed; one known artifact-free-worktree failure because ignored build/model/venv assets were absent. The focused `HangRecoveryTests` class passed 21/21.
- **Coverage completion gate:** all focused lifecycle tests and existing `HangRecoveryTests` pass; the full suite passes when the worktree uses the live ignored build/model/venv assets; no assertion or test is weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| xclip remains a directly owned foreground process | L5 command seam, `test_runtime_integration.py` | Focused test fails because `_insert` uses blocking/default-fork `subprocess.run` | Focused unittest then full runtime suite | Exact `Popen`, `-quiet`, stdin, readiness, and xdotool order |
| replacing clipboard ownership reaps the prior process | L5 process seam | Focused test fails because no handle exists | Focused unittest | terminate/wait and kill-on-timeout paths |
| shutdown always stops tray and clipboard after durable drain | L5 component integration | Focused test fails because shutdown omits both | Focused unittest plus session suite | idle and owned-process cases |
| service restart no longer times out | L1 installed user service | Existing journal contains `stop-sigterm timed out` and SIGKILL | Two timed restarts plus bounded journal query | old PID exits cleanly; service active/ready |

### Realism target

Level 5 controlled process seams prove deterministic command order and failure cleanup; Level 1 verification uses the installed LG Gram service, X11 session, tray backend, and current SYCL configuration after merge.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| Paste owns clipboard and service exits cleanly | Python daemon, pystray, xclip, systemd | LG Gram user session | Focused tests followed by two service restarts and journal inspection |
| Backend remains healthy | launcher, warm server, SYCL build/model | LG Gram | `scripts/dictation/check.sh` and service status |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| xclip unavailable | executable lookup fails | Delivery fails before sending any text event; durable transcript remains | L5 | unittest mocks | Focused unittest |
| xclip cannot take ownership | process exits or readiness probe fails | Process is reaped; delivery fails before UI mutation | L5 | unittest mocks | Focused unittest |
| previous owner still running | consecutive dictation | Prior child terminates before replacement | L5 | unittest mocks | Focused unittest |
| owner ignores SIGTERM | subprocess cleanup timeout | Escalate only that child to kill, bounded | L5 | unittest mocks | Focused unittest |
| shutdown during active recording | recorder and queued work | Existing finalization/drain precedes resource cleanup | L5 + existing suite | unittest | Focused ordering assertion and regression suite |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| None | Installed service and journal checks are machine-readable | — | — | — |

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| Worktree links to live ignored build/model/venv assets | Run the complete existing suite without copying large artifacts | None | Validation | Remove with worktree |
| `/tmp` validation evidence JSON | Reviewer input | One-run audit | After merge | Leave outside repository |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Disposition |
|---|---|---|---|
| Foreground xclip readiness is unreliable | Exact-byte readiness test or real paste fails | Use a small owned wrapper/handshake while retaining process ownership | Amend plan |
| Tray stop itself blocks beyond its 0.5-second allocation | focused timing/service evidence | Return from daemon stop helper; process exit remains authoritative cleanup | Fix within plan |
| Explicit resource cleanup races queued final paste | ordering test or retained chunk remains pending | Serialize final insert/cleanup and leave chunk recoverable | Amend plan |
| Backend selection changes | runtime suite | Revert unrelated behavior; no accelerator-specific branch is allowed | Fix within plan |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Fix the low-risk xclip shutdown timeout | P1–P3 | Lifecycle tests and clean service restarts |
| Do not affect transcription or missing-audio recovery | Blast-radius invariants, P2 | Existing 47-test dictation suite and preserved `KillMode=mixed` |
| Ship to the fork while preserving live edits | dedicated branch/worktree | PR diff and post-merge live checkout audit |

## 9. Primary Linear issue

- **Identity:** none — this personal fork has no configured Linear tracker identity
- **Reconciliation state:** no-op
- **Desired title:** Stop dictation clipboard and tray children cleanly
- **High-level description:** Own xclip directly, stop clipboard/tray resources after durable shutdown, retain recorder-safe systemd policy, and verify on the LG Gram.

### Adapted children/subtasks

No tracker subtasks; this is one local execution unit.

## 10. Execution checklist and outcomes

- [x] No prototype triggered; local source/process evidence is sufficient
- [x] No governing ADR; size-S no-ADR authority recorded
- [x] Desired no-tracker identity included in approval packet
- [x] Phase 0 evidence independently approved in Round 3
- [x] P1–P3 completed with RED/GREEN evidence
- [x] Happy path and edge matrix pass
- [x] Blast-radius invariants pass
- [x] No configured coverage threshold; requirement-to-test traceability complete
- [x] No test/assertion/threshold/exclusion weakened
- [x] Human-only validation not required
- [x] Material cutover not required
- [x] Scaffolding disposition decided; worktree links remain untracked until cleanup
- [x] Validation outcomes recorded

## 11. Execution evidence

| Unit | RED evidence before production edit | GREEN / regression evidence | Outcome |
|---|---|---|---|
| P1 foreground clipboard owner | Focused class failed because `_insert` never called `Popen`, no owned-process cleanup API existed, and readiness failure was unobservable | Exact command/order, stdin close, byte readiness, retained handle, TERM/KILL escalation, and side-effect-free ownership failure pass | Complete |
| P2 finalizer synchronization and bounded cleanup | Focused class reproduced shutdown returning before a blocked finalizer; validate-for-PR Round 1 RED raised `TypeError` when the fake blocked lock lacked the old context-manager protocol | Concurrent-finalizer behavior remains covered; fake-clock blocked-lock coverage proves a 12-second timed acquisition and gives tray only the final 250 ms after clipboard cleanup | Complete |
| P2 tray lifecycle | Initial owned-runner tests failed; validate-for-PR Round 1 RED showed a runner `OSError` delayed past 50 ms was reported as successful startup | Normal ownership, immediate and delayed startup failure, and delayed stop behavior pass through an explicit ready/error handshake; actual-display stop completed in 0.002–0.003 seconds with no residual non-daemon thread | Complete |
| P3 service readiness | Existing journal contains 15-second stop timeouts and SIGKILL of Python/xclip/parecord | Service-unit invariant remains `KillMode=mixed`; post-merge two-restart journal check is defined in §5 | Ready for ship handoff |
| Regression | Baseline focused suite was 21/21; artifact-linked full baseline was 46/47 with one missing-artifact failure before links | Validate-for-PR Round 1 focused suite passes 35/35, including consecutive owner replacement, wrong-byte readiness timeout, and unavailable-xclip side-effect-free failure. Current artifact-free runtime module passes 56/57 with the previously documented missing-build/model failure; Ruff lint/format and `git diff --check` pass | Complete |

No test, assertion, configured threshold, or exclusion was weakened. Review evidence is `/tmp/whisper-dictation-clipboard-shutdown-evidence.json`.

## 12. Validation outcome

Validate-for-PR completed at light/hotfix depth. The initial functional review found the 250 ms direct-typing partial-write hazard; the fix removed that non-atomic fallback and retained explicit durable failure. Reviewer A and Reviewer B both passed the delta, the complete 88-test dictation suite passed, Ruff lint/format passed, and the real X11 probe delivered 200/200 characters exactly.
