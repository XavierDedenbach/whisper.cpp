# Implementation Plan: Recover semantic failures through the warm Whisper server

**Status:** Active (auto-approved)
**Approval authority:** pre-approval by human, 2026-09-03T22:10:01Z (auto-approved)
**Activation authority:** pre-approval by human, 2026-09-03T22:10:01Z (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (scope analysis, 2026-09-03)
**Epic / execution unit:** none
**Linear project:** none — this personal fork has no configured tracker
**Primary Linear issue:** none — this personal fork has no configured tracker
**Material cutover:** no — the change is a reversible request/retry policy in the existing local daemon
**Cutover plan dependency:** none
**Routine deployment phase:** none — the user separately requested a post-merge relaunch of the LG Gram user services
**Supersedes:** none
**Superseded by:** none
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp`
**Execution mode:** autonomous
**Phase 0 gate:** independent-review (pre-approved)
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** dedicated worktree `/home/linu_x/Documents/git/whisper.cpp-worktrees/dictation-sycl-semantic-retry`
**Scope:** Correct the server decoding profile, retry semantically invalid punctuation-only results through the selected warm server before restarting it or using the CPU CLI, add the observed `End of video` silence hallucination, preserve CPU/CUDA/SYCL portability, test and merge to the fork integration branch (`master`). Existing unrelated live-checkout edits remain outside this execution unit.

## 1. Observable outcome and invariants

### End-to-end outcome

For a non-silent dictation chunk, the daemon requests ordinary text decoding without emitted timestamps. An empty or punctuation-only HTTP-success response is treated as an inference failure, retried once on the same warm selected server with beam search, then retried after at most one server restart before the existing CPU CLI fallback. Valid first-pass text remains the fast path. The request policy works with whichever CPU, CUDA, or SYCL server the machine configuration selected.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Backend selection | Config chooses server/CLI and CPU/CUDA/SYCL build | Existing runtime-selection suite | No accelerator hard-coding; only server request/retry policy changes |
| Fast path | One successful server request returns immediately | Focused valid-response test | Add no-timestamp fields; no retry/restart/CPU call |
| Transport recovery | HTTP/connection failure restarts server, then may use CPU CLI | Existing hang-recovery tests | Preserve bounded restart and CPU fallback |
| Durable delivery | Each chunk reaches one terminal state and later chunks continue | Existing session/runtime suites | A semantic retry may replace invalid punctuation; no duplicate paste |
| Silence handling | Low-RMS known phrases are ignored | Hallucination unit test | Add the observed `End of video` phrase only |

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| Timestamp-emitting decode mode causes the punctuation collapse | Retry policy would mask another capture/model problem | Existing warm server; fresh server; SYCL CLI | Replay retained failing WAV with old and `no_timestamps=true` form fields | CPU CLI and fresh SYCL server comparison | Same WAV fails under old fields and returns speech under no-timestamp mode; restart alone does not cure it |
| Same-server semantic retry is safe and useful | Extra latency without recovery, or forced CPU use | Beam search; nonzero-temperature sampling; bounded restart; existing CPU fallback | Compare materially distinct profiles over all retained punctuation failures and scan all retained chunks under corrected greedy | Exact call-order matrix | Valid first pass makes one call; semantic failure retries a genuinely different selected-server profile before restart/CPU; all-failure path remains bounded |
| Generic request fields preserve machine portability | Fork could regress CPU/CUDA laptops or Spark | Server HTTP API fields are backend-independent; build selection remains unchanged | Command-construction assertions and existing runtime-selection suite | Real LG Gram SYCL service check | No code branches on accelerator; all existing selection/recovery tests pass |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Token-timestamp root cause | WAV is valid/non-silent; process restart is insufficient; no-timestamp decode recovers speech on the selected accelerator | Retained 28.6 s chunk `20260903T214221.535413Z-3075591/chunk-0004.wav` (RMS 508) returned only `,` on the existing warm server and a fresh SYCL server with server defaults. SYCL CLI recovered the full transcript. The same warm server recovered it in 2.13 s with `no_timestamps=true`, `token_timestamps=false`, `best_of=2`; a second retained 28.9 s/RMS 541 chunk showed the same original failure | Supported as initially stated; causal wording challenged by review | Two real continuation chunks on this LG Gram's SYCL backend; CPU/CUDA are covered by backend-independent request construction rather than equivalent failure samples | Medium pending correction | Distinguish `no_timestamps` from `token_timestamps` and expand retained corpus |
| Retry ordering and bounded fallback | Semantic HTTP success must retry SYCL/server path; transport failure still restarts; CPU remains final fallback | Current `_transcribe` bounds transport recovery to initial server call, one restart, and CPU fallback. The clean target worktree returns punctuation as successful text and would paste it; only unrelated dirty live-checkout work filters it after transcription | Partial | Repository control-flow evidence establishes current behavior; exact new ordering requires TDD tests | High for implementation design | Prove in implementation tests |
| Portability | Request controls do not select an accelerator and existing build selection remains untouched | `dictation.py`, `runtime-env.sh`, and existing runtime-selection tests select build/backend before the HTTP request. Whisper server accepts the tested form fields on the active SYCL binary | Supported | Real HTTP proof is SYCL-only, but fields belong to the common server API and no build scripts/config are changing | Medium-high | Run full selection suite and verify no accelerator-conditioned code |

**Round summary:** Real retained audio isolates the defect to timestamp-emitting server decoding, not recording, prompt, process staleness, or SYCL computation generally. Correcting the ordinary request is the smallest fix. Punctuation-only output on already-qualified non-silent audio is a narrow semantic-error signal suitable for one same-server robust retry before the existing restart/CPU ladder.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** REVISE_AND_RERUN

The reviewer accepted `no_timestamps=true` as the ordinary fix but found that `best_of=5` at temperature zero is normally the same greedy strategy, requested a materially distinct retry profile, corrected causal wording, all four observed punctuation failures, exact reproducibility, and a complete transition matrix.

> **Phase 0 round 1 status — revisions required and carried into Round 2.**

#### Round 2

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| Timestamp-emitting decode mode causes the observed collapse | Old request reproduces on warm and fresh server; `no_timestamps` is the discriminating control; retained audio is valid | Four non-silent retained failures reproduced exactly under the production prompt: WAV SHA prefixes `3c1f3be6` (28.640 s, RMS 507.7) → `,`; `62bd70db` (28.920 s, RMS 541.0) → `,`; `f9fcd2cf` (21.120 s, RMS 552.2) → `,`; `3badc750` (28.540 s, RMS 699.1) → `-`. Corrected greedy recovered non-punctuation text from all four in 1.249–2.125 s. A separate server process (PID 3153752, port 18179) replayed SHA `3c1f3be6` with the old profile and returned `{"text":",\n"}` in 2.244 s | Supported | Four real failures plus a fresh process on the LG Gram SYCL runtime. Full transcripts are intentionally omitted from this public-fork plan; exact output SHA-256 values were captured in the terminal evidence | High | None for Phase 0 |
| A materially distinct defensive retry is available | Retry changes search strategy; output stays usable; cost is acceptable only on rare semantic failure | On all four failures: corrected greedy recovered in 1.249–2.125 s; `beam_size=5`, temperature 0 recovered in 2.883–4.635 s with distinct output hashes on all four; temperature 0.2 plus `best_of=5` took 2.144–4.754 s and duplicated greedy output on three of four | Supported for beam search | No retained chunk fails under corrected greedy, so beam recovery is defensive rather than demonstrated against a corrected-profile failure. It is materially distinct by server source and output hashes, and runs only after invalid text | Medium-high with bounded residual risk | Prove fixed call order and semantics in tests |
| Corrected greedy is safe across retained speech | Ordinary request avoids semantic regressions in the available corpus | All 27 retained WAV chunks were replayed through the active SYCL server with temperature 0, `no_timestamps=true`, `token_timestamps=false`, production prompt/carry/suppression. Result: 27/27 non-empty, non-punctuation; maximum observed latency 6.866 s | Supported | One user's same-day corpus on one SYCL machine; existing tests and shared server source cover non-SYCL selection | High for observed defect and local activation | Continue with regression suite |
| Retry/fallback ordering is bounded | Every semantic/transport combination has one restart maximum and CPU last | Transition table below fixes normal versus beam profile, restart count, maximum server calls, and CPU behavior | Supported as executable design | Control-flow proof remains implementation work | High | Prove matrix with exact call-count tests |
| Request policy is portable | Fields are parsed by common server code before backend execution; accelerator selection remains outside request construction | Real CPU/SYCL acceptance, shared `examples/server/server.cpp` parsing/strategy selection, existing 37-test runtime suite, and planned backend-parameterized command assertions | Supported | No CUDA hardware in this worktree; common server source is the bounded proxy | Medium-high | Run all selection tests after implementation |

**Reproducible request template:** `curl -sS --connect-timeout 2 --max-time 100 http://127.0.0.1:8178/inference -F file=@<retained.wav> -F response_format=json -F language=en -F temperature=0.0 -F no_timestamps=true -F token_timestamps=false` plus the installed production prompt, `carry_initial_prompt=true`, and `suppress_nst=true`. The old control omits both timestamp fields; the beam variant adds `beam_size=5`. Input SHA-256 values above bind the private retained audio without committing it.

##### Bounded transition table

| Initial corrected-greedy result | Same-process action | Restart action | Post-restart action | Final fallback | Maximum server calls / restarts |
|---|---|---|---|---|---|
| Valid text (contains a non-punctuation character) | Return immediately | None | None | None | 1 / 0 |
| Empty or punctuation-only text | Run beam-5 once; return if valid | If beam is invalid or transport-fails, restart once | Run beam-5 once; return if valid | CPU CLI, if available | 3 / 1 |
| Transport/HTTP failure (`None`) | None | Restart once | Run corrected greedy once; if semantic-invalid, run beam-5 once | CPU CLI, if restart fails or both post-restart decodes fail | 3 / 1 |

Server mode is accelerator-neutral: “selected server” means whichever CPU, CUDA, or SYCL build the existing machine config starts. Empty/punctuation validity is checked inside `_transcribe` before delivery; text such as `Wait, what?` is valid because it contains word characters.

**Round summary:** Round 2 corrects the causal claim to no-timestamp decode mode, binds all four observed failures by audio hash and exact punctuation response, demonstrates the failure after a fresh process start, rejects `best_of=5` as a false differentiator, selects beam-5 as a demonstrably distinct defensive profile, and bounds the full recovery ladder to three server calls and one restart.

##### Review

**Gate:** independent-review (pre-approved)
**Verdict:** APPROVE

The fresh reviewer found the corrected mode-level claim, four-failure and fresh-process evidence, beam-versus-sampling portfolio, 27-chunk corpus scan, finite transition table, clean baseline, and shared backend contract sufficient. The remaining call-order and portability checks belong to implementation tests.

> **Phase 0 status — approved in Round 2.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Server form request | `dictation.py::_transcribe_server` | One curl form command with prompt/language options | Extend with common no-timestamp fields and optional `beam_size` | `dictation.py` |
| Recovery ladder | `dictation.py::_transcribe`; `HangRecoveryTests` | Initial server call → restart → server call → CPU | Extend with punctuation validation and same-server robust retry, retain bounded restart/CPU ordering | `dictation.py`, existing test module |
| Hallucination vocabulary | `_HALLUCINATION_PHRASES`, `is_likely_hallucination` | Exact/prefix normalized phrase matching | Add `end of video` | `dictation.py`, existing test module |
| Operator behavior | `scripts/dictation/README.md` | Documents warm server and CPU fallback | Document semantic selected-server retry and common decode profile | README |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0 | Validate root cause, retry seam, and portability assumptions | §2 | retained WAVs, source, plan only | Plan active | N/A investigation | Independent Phase 0 review | XS |
| P1 | No-timestamp server request and configurable decoding strength | §1 fast path | `dictation.py`, runtime tests | P0 | Command assertion lacks fields/parameter | Focused command test and runtime suite | XS |
| P2 | Bounded punctuation-only semantic retry through selected server, restart, then CPU | §1 recovery | `dictation.py`, runtime tests | P1 | Retry-order tests return comma or call CPU too early | Focused success/failure matrix and runtime suite | S |
| P3 | `End of video` silence phrase and operator documentation | User-observed failure | `dictation.py`, tests, README | P1–P2 | Phrase test returns false; docs omit retry semantics | Focused phrase test, lint, full dictation tests | XS |
| P4 | Production-equivalent LG Gram verification | User request | real SYCL server/service and retained WAV | P1–P3 | Current service still runs pre-merge code | Post-merge install/restart, health/check script, retained-WAV inference | S |

> **Phase P0–P3 status — complete. P4 is pending post-merge service relaunch.**

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** repository has no configured Python line/branch coverage gate. Before production edits, the 11-test `HangRecoveryTests` suite passed; the combined runtime/session suite passed 36/37 in the artifact-free worktree, with the sole failure caused by absent untracked build/model artifacts rather than code.
- **Coverage completion gate:** all focused and existing dictation tests pass; no assertion, test, threshold, or exclusion is weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| Common no-timestamp request | L5 subprocess command seam, `test_runtime_integration.py` | Focused unittest fails because timestamp fields and `beam_size` option are absent | Focused unittest plus full runtime module | Exact fields/default and robust profile |
| Semantic retry succeeds without restart | L5 controlled server seam | First comma is returned instead of retrying | Focused retry-order unittest | Two server calls; second robust; no restart/CPU |
| Semantic retry survives server restart | L5 controlled server seam | Repeated punctuation returns early | Focused retry-order unittest | Restart then one robust selected-server call |
| Exhausted server path uses CPU | L5 controlled server/CLI seams | Comma never reaches CPU | Focused fallback unittest | Bounded calls and exact CPU binary |
| Valid first pass remains fast | L5 controlled server seam | Characterization | Focused unittest | One normal call only |
| Observed phrase is ignored at low RMS | L5 pure-function test | `End of video.` is not classified | Focused unittest | Case/punctuation normalization |
| Real retained speech recovers | L2 same binary/service and real WAV | Old profile returns comma | Post-merge curl/service check | Non-punctuation transcript through SYCL |

### Realism target

Deterministic L5 tests are appropriate for retry ordering and command construction, where real process failure is expensive and nondeterministic. L2 validation uses the actual LG Gram SYCL binary, model, service, and retained failing WAV to prove the composed path.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| Existing runtime selection and durable delivery remain healthy | Dictation daemon, store, server/CLI selection | Worktree using live build/model artifacts | Full dictation unittest suite |
| No-timestamp request recovers retained speech | Real server, model, Iris Xe, retained WAV | LG Gram user service | HTTP inference plus service health/check script |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| Valid first response | Server returns words | Return immediately | L5 | unittest mocks | Focused unittest |
| Punctuation-only first response | HTTP 200 JSON contains comma | Retry same selected server with robust profile | L5/L2 | unittest + LG Gram | Focused unittest and retained WAV |
| Robust retry also punctuation | Semantic failure repeats | Restart server and retry selected server once | L5 | unittest mocks | Focused unittest |
| Transport error | Curl timeout/nonzero | Restart server, then retry | L5 | unittest mocks | Existing and updated hang tests |
| Selected server remains invalid/unavailable | All bounded server attempts fail | Use portable CPU CLI if present | L5 | unittest mocks | Focused fallback unittest |
| CPU unavailable | Server and fallback unavailable | Return empty; durable chunk records no text/failure per existing behavior | L5 | unittest mocks | Existing recovery path test |
| Silence phrase | Low-RMS inference says `End of video.` | Ignore without paste | L5 | pure function/delivery tests | Focused unittest |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| None | Real retained audio and service checks provide the required acceptance evidence | — | — | — |

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| Worktree links to live build/model/venv | Run real existing artifacts without copying | None | Validation | Untracked; remove with worktree later |
| Retained failed WAVs under user data | Reproduce production inference failure | Diagnostic/recovery value belongs to existing session retention | User-controlled retention | Do not modify or commit |
| Review evidence JSON under `/tmp` | Human-readable validation artifact without dirtying repo | One-run audit only | After merge | Leave outside repository |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Disposition |
|---|---|---|---|
| No-timestamp fields regress normal speech | Focused/real integration failure | Revert ordinary profile and apply no-timestamp only as semantic retry | Amend plan |
| Punctuation is valid intended dictation | Product behavior ambiguity | Keep rejection limited to server-mode non-silent chunks; document residual edge | Amend plan if tests expose a real use case |
| Server API differs across maintained backends | CPU/CUDA/SYCL test failure | Detect capability or use backend-neutral CLI flags | Replan only if common endpoint contract is false |
| Retry ladder becomes unbounded | Call-count test | Consolidate attempts into explicit fixed sequence | Fix within plan |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Retry through fast selected inference path | P2 | Same-server robust-retry test and retained-WAV SYCL result |
| Explain and prevent comma-only loss | P1–P2 | No-timestamp command assertion plus punctuation semantic-failure matrix |
| CPU only after selected-server recovery | P2 | Exact call-order tests |
| Preserve CUDA/SYCL/CPU fork operation | P1–P4 | No accelerator branch plus existing selection suite and LG SYCL check |
| Merge and relaunch | ship-it-good/P4 | Merged PR verification, service active/healthy, check script |

## 9. Primary Linear issue

- **Identity:** none — this personal fork has no configured Linear tracker identity
- **Reconciliation state:** no-op
- **Desired title:** Recover punctuation-only Whisper server failures without premature CPU fallback
- **High-level description:** Use backend-neutral no-timestamp decoding, retry punctuation-only semantic failures on the selected warm server, preserve bounded restart/CPU fallback, and verify on the LG Gram SYCL service.

### Adapted children/subtasks

No tracker subtasks; the plan phases are one local execution unit.

## 10. Execution checklist and outcomes

- [x] No separate prototype triggered; retained production WAV evidence is sufficient Phase 0 input
- [x] No governing ADR rows; size-S no-ADR authority recorded
- [x] Tracker absence recorded; no external tracker mutation planned
- [x] Phase 0 evidence gathered
- [x] Phase 0 independent review approved in Round 2
- [x] Pattern inventory reconciled after Phase 0
- [x] P1–P3 implemented with RED/GREEN evidence
- [ ] P4 service-ready verification complete
- [x] Happy-path implementation integration passes
- [x] Edge-case matrix passes
- [x] Blast-radius invariants pass
- [x] No configured coverage threshold; requirement-to-test traceability complete
- [x] No test, assertion, threshold, or exclusion weakened
- [x] Human-only validation not required
- [x] Material cutover not required
- [ ] Scaffolding disposition completed
- [ ] Validation outcomes recorded

## 11. Execution evidence

| Unit | RED evidence before production edit | GREEN / regression evidence | Outcome |
|---|---|---|---|
| P1 request profile | Focused `HangRecoveryTests` run: `test_server_request_uses_no_timestamp_greedy_profile` failed because both fields were absent; beam-profile test raised the expected missing-keyword `TypeError` | Focused class passes; command assertions run for `build`, `build-cuda`, and `build-sycl` and verify beam is absent on the fast path | Complete |
| P2 semantic recovery | Same 20-test RED run returned comma from semantic cases, pasted loud punctuation, and never reached beam/restart/CPU in the required order (7 failures, 1 expected missing-API error) | 21 focused tests pass. Call-order assertions prove normal → beam → one restart → beam → CPU, transport → restart → normal → beam, valid one-call fast path, and empty/punctuation handling | Complete |
| P3 phrase/docs | RED assertion showed `is_likely_hallucination("End of video.")` was false | Phrase assertion passes; README documents backend-neutral selected-server recovery | Complete |
| Regression | Pre-edit focused baseline was 11/11 | Combined runtime/session suite is 47/47; Ruff check and format check pass; `git diff --check` passes | Complete |
| Real implementation path | Old server profile reproduced punctuation on all four observed failures | Worktree `Dictation._transcribe` against the active real SYCL server recovered retained SHA `3c1f3be6` in 1.721 s with 45 words and non-punctuation output hash `57c9514e…` | Complete |

No test, assertion, configured threshold, or exclusion was weakened. P4 will record the merged commit, installed live code, service health, and final retained-WAV check.

## 12. Validate-for-PR outcome

- **Settings:** size S, depth light, Sentry N/A, Hotfix no.
- **Initial round:** Form A PASS; Functional B FAIL on the incomplete hand-written punctuation set; unified verdict `CHANGES_REQUIRED`.
- **Fix-owner:** replaced the allow-list with Unicode `P*` classification. Focused representative-character tests produced 12 RED subtest failures before the production edit and passed afterward; all 47 tests and lint/format gates passed.
- **Delta round:** Form A PASS; Functional B PASS. The prior blocker is resolved for `()`, `[]`, `/`, `¿?`, `。`, and `،`; no new findings.
- **Final verdict:** `APPROVE` after one delta round. No accepted risks, follow-ups, Sentry work, or cutover-owned probes.
- **Review evidence:** `/tmp/whisper-dictation-sycl-semantic-retry-evidence.json`.
