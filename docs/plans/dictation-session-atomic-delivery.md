# Implementation Plan: Atomic Long-Session Dictation Delivery

**Status:** Complete
**Approval authority:** pre-approval by user, 2026-09-04 (auto-approved; explicit full-access and proceed instructions)
**Activation authority:** pre-approval by user, 2026-09-04 (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (one daemon component and focused tests; no public or accelerator interface change)
**Epic / execution unit:** none
**Linear project:** none
**Primary Linear issue:** none — this personal fork has no configured tracker
**Material cutover:** no — local user service only; no data/configuration migration or external coordination
**Cutover plan dependency:** none
**Routine deployment phase:** none; the LG Gram user-service restart is local operational validation
**Supersedes:** none
**Superseded by:** none
**Target repo:** XavierDedenbach/whisper.cpp fork
**Execution mode:** autonomous
**Phase 0 gate:** human (approved by the user's explicit instruction to investigate and fix the live defect)
**Maximum Phase 0 rounds:** 1
**Authorized phases:** through-completion
**Context strategy:** existing dedicated Xclip lifecycle worktree
**Scope:** Preserve rolling recorder handoff, durable WAVs, online transcription, vocabulary contents/replacements, and CPU/CUDA/SYCL selection. Change only interactive delivery: assemble completed fragments in sequence and paste once after the final slice. Finish bounded foreground-Xclip lifecycle fixes and add observable delivery outcomes. Historical recovery remains file-only and never pastes. The pre-existing prompt-cue formatting edit is a separate execution unit governed by `dictation-unfinished-vocabulary-prompt.md`.

## 1. Observable outcome and invariants

### End-to-end outcome

A three-to-twelve-minute recording produces all valid slice WAVs and an ordered `transcript.txt`, then dispatches that complete text to the focused field exactly once after stop. No middle text can disappear because of a target application's mid-recording focus change or re-render.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Recorder | Validated recorder rotates every 45 seconds | rollover/failure matrix | None |
| Inference | One FIFO worker uses configured server/CLI | runtime and semantic-retry tests | None in this execution unit; prompt-cue formatting is governed separately |
| Durable recovery | Every WAV and terminal fragment survives independently | session-store suite | None; recovery still suppresses UI insertion |
| Interactive insertion | Every slice invokes an unverified paste | new RED long-session test | One verified-ready clipboard paste per stopped session |
| Portability | Build/runtime selects CPU, CUDA, or SYCL | accelerator-selection suite | Delivery remains backend-independent |

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| Audio/inference, not delivery, loses middle slices | Wrong subsystem would be changed | manifests, retained WAVs, journal | Audit every multi-slice deployed session | inspect recorder failure/empty-WAV logs | Proceed only if post-handoff multi-slice WAVs are valid and complete |
| Session-atomic insertion removes the observed loss mode | Long-session text could still be overwritten mid-recording | final-only paste from durable fragments | Four-slice component test with one insertion | Xvfb real xclip/Text probe | Exactly one ordered insertion after finalization |
| Foreground xclip can make readiness and cleanup bounded | Final paste could still race or block restart | retained `Popen`, exact-byte probe, timeout | X11 selection probe and focused lifecycle tests | fail before UI mutation when ownership is unavailable | Exact bytes visible before paste; command and shutdown waits bounded |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Middle slices exist before delivery | recorder handoff succeeds; FIFO transcribes every retained WAV | 93 manifests audited. Every deployed multi-slice session after the handoff fix has complete, non-empty WAVs/fragments; the reported 144.8-second session has four complete chunks. Journal contains no new empty-WAV/recorder failure | Supported | Files and journal prove daemon state, not target-widget state | High | None |
| Current success signal is not delivery evidence | xdotool return and target mutation are not checked | code audit: `_deliver_chunk` marks storage complete, invokes default-fork xclip and unchecked xdotool, then reports `Typed`; no session reconciliation exists | Supported | Generic applications expose no portable read-back API | High | Replace incremental side effects with one final transaction and honest logging |
| Xclip readiness can be established | foreground owner serves exact expected bytes | 700-cycle Xvfb/Tk probe had 0 wrong pastes for both legacy and readiness paths; xclip manual confirms default forks and `-quiet` remains foreground | Supported for readiness; legacy race not proven | Xvfb cannot model Codex's controlled input re-renders | High for chosen final-only design | Keep exact-byte readiness and bounded child ownership |

**Round summary:** Recording and transcription are healthy; the unobservable repeated UI side effect is the remaining loss boundary. One final paste removes all mid-session target-state races without delaying earlier inference.

##### Review

**Gate:** human
**Verdict:** APPROVE — user explicitly rejected the deployed behavior and instructed investigation and repair.

> **Phase 0 status — approved.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Ordered persistence | `session_store.py`, long-session plan | manifest indices and rebuilt transcript | expose completed text in manifest order | session store |
| Finalization ordering | `_stage_chunk`, `_submit_in_order`, worker FIFO | sequence gate | mark final live job; paste after it is processed | dictation daemon |
| Clipboard lifecycle | Xclip shutdown branch and plan | foreground owner plus byte probe | retain, bound xdotool, report failure | dictation daemon |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P1 | Ordered completed-session text and final-only delivery | §1 | `session_store.py`, `dictation.py`, tests | Phase 0 | four chunks currently invoke four inserts before stop | one ordered insert after the final queue item, including ignored/failed tail | S |
| P2 | Bounded truthful clipboard transaction | §1 | `dictation.py`, tests | P1 and Xclip branch | xdotool call has no timeout/result contract | readiness, return-code, timeout, shutdown-race tests | S |
| P3 | Reconcile diagnostics/docs and activate | user closeout request | README, plans, live checkout/service | P1-P2 | current README promises incremental paste | lint, full suite, Xvfb path, merge, service restart/health | XS |

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline:** no repository coverage threshold/tool is configured; retain requirement-to-test traceability and do not weaken tests.
- **Coverage completion gate:** focused tests, complete dictation test suite, Ruff, and `git diff --check` pass.

| Behavior/requirement | Test level and path | RED expectation | GREEN/regression | Coverage expectation |
|---|---|---|---|---|
| Four slices paste once in order | L4 actual session store + mocked inference/UI seam | first slice currently inserts immediately | component test and session-store suite | all fragments, exact order, one call |
| Failed final ingest still flushes prior text | L4 store/queue seam | no final delivery marker exists | focused queue test | partial completion remains deliverable |
| xdotool cannot hold shutdown | L5 process seam | no timeout and result ignored | timeout/nonzero/shutdown tests | bounded failure is visible |
| Portability unchanged | L2 fake CPU/CUDA/SYCL launchers | characterization | full runtime selection suite | all backend variants |

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| Long-session transaction | queue, store, inference seam, insertion seam | temp filesystem | focused four-slice test |
| Real X11 clipboard | foreground xclip, xdotool, editable widget | Xvfb/Tk | repeated exact-text probe |
| Installed service | systemd, SYCL server, local X11 daemon | LG Gram | restart, `/health`, journal ready line |

### Edge-case and failure matrix

| Scenario | Expected behavior | Verification |
|---|---|---|
| final slice is short/silent | paste all earlier completed fragments once | focused test |
| arbitrary middle transcription fails | transcript records gap; remaining completed text still dispatches once | fault test |
| final WAV persistence fails | final marker still dispatches prior completed fragments | queue/store test |
| xclip unavailable/wrong bytes | fail before any text event; retain transcript | lifecycle tests |
| xdotool hangs/nonzero | bounded failure, transcript path retained and reported | lifecycle tests |
| restart before final dispatch | recovery updates files and never types historical text | existing recovery tests |

### Human-only validation

None required for implementation readiness. The user will perform natural long-form speech after the machine-verifiable local restart.

## 6. Temporary scaffolding

Xvfb/Tk probe is disposable and stays outside the repository. Any worktree symlinks used for ignored build/model assets are removed during cleanup.

## 7. Fallbacks and replan triggers

If one final foreground-clipboard paste fails in the X11 editable-widget probe, use a directly owned X11 insertion helper before deployment. If final-only delivery exceeds the user's stop-latency target, retain online inference and measure only the final partial slice before changing model/backend.

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| no missing middle chunks | P1 | four-slice and random-failure tests |
| recover audio/transcript | P1 | session-store and restart tests |
| fix remaining Xclip work | P2 | lifecycle/shutdown tests |
| CPU/CUDA/SYCL portability | P3 | runtime-selection suite |

## 9. Primary Linear issue

- **Identity:** none — no tracker configured
- **Reconciliation state:** no-op
- **Desired title:** Deliver long dictation sessions atomically
- **High-level description:** Preserve rolling capture and online inference while replacing unreliable incremental UI pastes with one ordered final transaction.

### Adapted children/subtasks

None; this is one local execution unit.

## 10. Execution checklist and outcomes

- [x] No ADR/prototype required; deployed files and bounded X11 probe discriminate the subsystem
- [x] Phase 0 evidence gathered and human-authorized
- [x] P1-P3 RED/GREEN evidence recorded
- [x] Happy path, edge matrix, and blast-radius invariants pass
- [x] No test/assertion weakened
- [x] Validation outcomes recorded

### Implementation evidence

- **P1:** the four-slice test failed while chunks pasted incrementally, then passed with one ordered final dispatch; short, missing, and failed final slices also flush earlier completed fragments.
- **P2:** timeout and nonzero-result tests failed against the unbounded `xdotool` call, then passed with bounded execution; worker, recorder, Xclip, and concurrent-finalizer failure tests remain green.
- **P3:** the complete 88-test dictation suite, the focused 65-test store/recovery suite, Ruff lint/format checks, `git diff --check`, and repeated Xvfb clipboard probes pass. CPU, CUDA, and SYCL selection tests remain green.

### Validation outcome

Validate-for-PR completed at light/hotfix depth with one delta. Reviewer A and Reviewer B both passed after the direct-typing partial-write hazard was removed and the prompt-only edit was separated. The final review has no blocking, should-do, or nit findings and accepts no risks.
