# Implementation Plan: Unfinished Whisper Vocabulary Prompt

**Status:** Complete
**Approval authority:** pre-approval by user, 2026-09-04 (auto-approved; resolve and close the remaining local dictation edits)
**Activation authority:** pre-approval by user, 2026-09-04 (auto-approved); Authorized phases: through-completion
**ADR(s):** none — No-ADR authority: size S per estimate-size
**Size:** S (one prompt formatter, its default, and focused tests)
**Epic / execution unit:** none
**Linear project:** none
**Primary Linear issue:** none — this personal fork has no configured tracker
**Material cutover:** no
**Cutover plan dependency:** none
**Routine deployment phase:** none; the local service restart is validation
**Target repo:** XavierDedenbach/whisper.cpp fork
**Execution mode:** autonomous
**Phase 0 gate:** human (approved by the user's instruction to resolve the retained local edits)
**Authorized phases:** through-completion
**Scope:** Format the generated vocabulary prompt as an unfinished cue by replacing a trailing prefix period with a colon and omitting generated terminal punctuation. Preserve explicit `WHISPER_PROMPT` text verbatim, vocabulary contents, replacements, model/backend selection, and inference retry behavior.

## 1. Observable outcome and invariants

Generated prompts have the deterministic form `Technical dictation: term, term` and do not end with punctuation that can prime a punctuation-only continuation. An explicit prompt remains exactly user-controlled.

| Contract | Invariant |
|---|---|
| Vocabulary | Shared and machine-local term loading is unchanged |
| Replacements | Spoken-form replacement behavior is unchanged |
| Explicit prompt | `WHISPER_PROMPT` is returned verbatim after surrounding-whitespace normalization already performed by config loading |
| Runtime | CPU, CUDA, SYCL, server/CLI, retries, and model stay unchanged |

## 2. Phase 0 evidence

The retained local diff was isolated to `build_whisper_prompt`, its default prefix, and one test. The punctuation-only output incident makes prompt termination a plausible contributor, while the formatter can be tested deterministically without changing inference or accelerator code. The user explicitly authorized closing this retained edit.

**Gate:** human
**Verdict:** APPROVE

## 3. Execution units

| Unit | Deliverable | First failing test | Green verification |
|---|---|---|---|
| P1 | Generated prompt is an unfinished cue | original formatter ends generated text with `.` | exact prompt-shape test |
| P2 | Explicit prompt remains user-owned | proposed formatter stripped its trailing period | verbatim explicit-prompt test |
| P3 | Portable runtime unchanged | characterization | CPU/CUDA/SYCL selection and full dictation suite |

## 4. Test strategy and evidence

- **RED:** the unfinished-cue test failed against the original period-terminated formatter.
- **GREEN:** generated prefix and vocabulary terms now produce an exact unfinished cue; an explicit prompt retains its final period.
- **Blast radius:** run the complete dictation suite, Ruff lint/format, and CPU/CUDA/SYCL runtime-selection tests with the delivery execution unit.

## 5. Human-only validation

None required. Natural dictation after restart is operational observation, not an implementation gate.

## 6. Execution checklist

- [x] Prompt scope separated from atomic-delivery invariants
- [x] Generated-cue RED/GREEN evidence recorded
- [x] Explicit prompt preservation covered
- [x] Complete suite and validation review recorded

## 7. Validation outcome

The complete 88-test dictation suite passed. Reviewer B verified generated unfinished-cue formatting, verbatim explicit prompts, and unchanged CPU/CUDA/SYCL selection in the final delta; Reviewer A's lint, format, and whitespace gates passed. No risks or follow-ups were accepted.
