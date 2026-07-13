# Implementation Plan: Dictation accuracy via small.en + warm large-v3-turbo

**Status:** Complete  
**Approval authority:** Human — chat request 2026-07-12 to implement Option 1 and Option 2; `/implement-plan` in automated fashion  
**Activation authority:** Same chat request — `Authorized phases: through-completion`  
**ADR(s):** no-ADR — operational upgrade of existing local dictation daemon (model selection, CLI flags, optional warm `whisper-server` sidecar). No new durable cross-repo interface; stays inside `scripts/dictation/` + user systemd/config.  
**Epic / execution unit:** none  
**Execution issue:** none (local fork; not Linear-tracked)  
**Supersedes:** none  
**Superseded by:** none  
**Target repo:** `/home/spark-x/git/whisper.cpp`  
**Execution mode:** autonomous  
**Phase 0 gate:** independent-review  
**Maximum Phase 0 rounds:** 3  
**Authorized phases:** through-completion  
**Context strategy:** current context (dictation-only; preserve unrelated working-tree edits)  
**Scope:**

**In**
- Upgrade default dictation model path from `tiny.en` → `small.en` (Option 1; `small.en-q8_0` is not offered by `download-ggml-model.sh`)
- Expose accuracy knobs: `WHISPER_PROMPT`, suppress non-speech (`-sns`), language `en`
- Add warm-model path via local `whisper-server` + HTTP `/inference` (Option 2)
- Download/install `large-v3-turbo-q8_0` and make it selectable (`WHISPER_MODEL` + `WHISPER_BACKEND=server`)
- systemd unit(s) / install / check / README updates for the new flow
- Keep singleton lock and single-autostart (systemd only) behavior from prior fix

**Out**
- Parakeet / Faster-Whisper / Option 3
- Changing core whisper.cpp C++ API
- Commits, push, or PR (not authorized)
- Mic hardware replacement

---

## 1. Observable outcome and invariants

### End-to-end outcome

After Ctrl+Space → speak → Ctrl+Space:

1. With `WHISPER_BACKEND=cli` + `WHISPER_MODEL=small.en`, transcription uses the larger model and accuracy flags; paste happens once.
2. With `WHISPER_BACKEND=server` + `WHISPER_MODEL=large-v3-turbo-q8_0`, a warm `whisper-server` handles inference; stop-to-paste latency is dominated by decode, not cold model load.
3. `check.sh` reports model file presence, backend health, and at most one dictation daemon.

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Hotkey UX | Ctrl+Space toggle start/stop | Manual / existing daemon | Unchanged |
| Paste path | clipboard `ctrl+v` or `xdotool type` | Unit-level insert unchanged | Unchanged |
| Singleton | One daemon via flock | Second process exits 0 | Unchanged |
| Autostart | systemd only | No `.desktop` dual-start | Unchanged |
| Config file | `~/.config/whisper-dictation/config.env` | New keys optional with defaults | Additive keys only |
| Default install model | `base.en-q5_1` in template | Template/default → `small.en` | Intentional upgrade |

---

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| A1: CUDA build can run `small.en-q8_0` and `large-v3-turbo-q8_0` on samples/jfk.wav with usable latency | Turbo/small unusable for dictation | `whisper-cli` timed runs; GPU vs `-ng` | Wall-clock encode+decode on jfk.wav; confirm GPU used | `nvidia-smi` during run | small cold&lt;3s decode; turbo warm&lt;2s preferred; both produce non-empty English text |
| A2: Cold CLI load of turbo dominates UX vs warm server | Option 2 unnecessary | Time cold `whisper-cli` vs warm `/inference` | Same WAV, two timings | Repeat 3× | Warm path ≥2× faster than cold CLI for turbo, or warm&lt;1.5s while cold&gt;3s |
| A3: `whisper-server` `/inference` multipart API returns parseable text for our WAV | Server path blocked | curl multipart from README | JSON/text contains transcript | Response format variants | Non-empty transcript matching jfk gist |
| A4: Daemon can fall back to CLI if server down without breaking paste | Server-only brittle | Design probe + dry API failure | Document contract; later test | N/A in Phase 0 | Fallback contract accepted: try server → CLI on connection error |
| A5: Download scripts yield expected filenames for `small.en` and `large-v3-turbo-q8_0` | Install/docs wrong | `download-ggml-model.sh` dry-run / list | Files exist after download | models/README | Exact `ggml-<name>.bin` paths; note `small.en-q8_0` unavailable |

### Phase 0 evidence and review

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| A1 | CUDA loads; both models decode EN under latency bar | `whisper-cli` on `samples/jfk.wav`: tiny REAL 0.72s; small.en REAL 0.92s + good punctuation; turbo-q8_0 REAL 1.05s×2. Log: `use gpu=1`, CUDA0 GB10, turbo load 267ms + encode 166ms + total 541ms | Supported | jfk.wav is clean studio speech (proxy for desk mic); latency bar cleared with margin | High | None |
| A2 | Cold load dominates; warm helps UX | Cold turbo CLI ~1.05s wall; warm `whisper-server :8099` `/inference` REAL 0.24 / 0.16 / 0.15s (text+json). Warm ≈4–7× faster | Supported | Short clip; longer utterances still benefit from warm residency. Cold already usable (~1s) on GB10 — Option 2 is latency polish + residency, not hard requirement | High | None |
| A3 | `/inference` multipart returns parseable text | curl `-F file=@jfk.wav -F response_format=json` → `{"text":" And so, my fellow Americans..."}`; `response_format=text` returns plain text | Supported | Exercised json+text; not srt/vtt | High | None |
| A4 | Daemon can detect server down and fall back | curl to `127.0.0.1:18099` → exit 7 connection refused in 0ms | Supported | Contract-level only; daemon fallback implemented in P2 | Medium | Accept with documented risk — implement try/except in P2 |
| A5 | Download names map to files | `small.en-q8_0` → Invalid (script lists `small.en`, `small.en-q5_1`, `small-q8_0` multilingual). Downloaded `ggml-small.en.bin` (466M) and `ggml-large-v3-turbo-q8_0.bin` (834M) successfully | Supported (after amend) | Plan amended to `small.en` | High | None |

**Round summary:** All material assumptions cleared. Amend model default to `small.en` (not `small.en-q8_0`). Warm server is worthwhile (~0.15s vs ~1.05s cold) even though cold turbo is already acceptable on this GPU. Proceed Option 1 (CLI small.en + flags) then Option 2 (server + turbo selectable, CLI fallback).

##### Review

**Gate:** independent-review  
**Verdict:** APPROVE  
**Reviewer:** fresh-context phase0-review subagent (Round 1)  
**Required before P1:** amend P1 deliverable to `small.en` (done). P2 must L2-test server→CLI fallback and pass prompt/sns/lang on HTTP.

> **Phase 0 status — APPROVED (Round 1).**

---

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Transcribe invocation | `scripts/dictation/dictation.py` `_transcribe` | subprocess `whisper-cli` `-m -f -nt -np -t` | extend with flags + optional HTTP | dictation.py |
| Config | `config.env`, `load_config()` | KEY=VAL env file | add `WHISPER_PROMPT`, `WHISPER_BACKEND`, `WHISPER_SERVER_URL`, keep `WHISPER_MODEL` | config.env + user config |
| Install / model download | `install.sh` `download_model` | `download-ggml-model.sh` | download small by default; optional turbo | install.sh |
| Autostart | `whisper-dictation.service` | single ExecStart launcher | add companion `whisper-server` unit or ExecStartPre/side service | systemd user units |
| Health check | `check.sh` | CLI + model file | check selected model + server if backend=server | check.sh |
| Server API | `examples/server/README.md` | curl multipart `/inference` | reuse as-is | no C++ changes |
| Singleton | `acquire_singleton_lock` | flock in runtime dir | keep | unchanged |

---

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | Verification | Effort |
|---|---|---|---|---|---|---|
| P0 | Evidence inventory + independent review APPROVE | Activation | docs/plans/… | — | Reviewer APPROVE | S |
| P1 | Download models; Option 1 CLI path with flags; defaults → small.en | Option 1 | models/, config.env, dictation.py, install.sh, README, check.sh | P0 | jfk.wav CLI; check.sh; config keys | M |
| P2 | Warm server backend + systemd sidecar; turbo selectable | Option 2 | dictation.py, new service unit, install/uninstall, check/README | P1 | timed warm vs cold; server down → CLI fallback | M |
| P3 | Apply user config, restart services, record validation | through-completion | `~/.config/…`, systemd | P2 | check.sh green for backend; human dictation note | S |

> **Phase 1 status — complete.** Models on disk; CLI flags + defaults → `small.en` in template; user config set to turbo+server.  
> **Phase 2 status — complete.** `run-server.sh`, server systemd unit, HTTP backend + CLI fallback verified.  
> **Phase 3 status — complete.** Live services restarted; `check.sh` green. Human Ctrl+Space E2E pending.

---

## 5. Test strategy

### Realism target

**Level 2** — production `whisper-cli` / `whisper-server` binaries on this machine with `samples/jfk.wav` and (where safe) live mic path via `test-mic.sh`. Level 1 full hotkey E2E remains human-only (needs focused field + speech).

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| small.en CLI transcript | whisper-cli + model + jfk.wav | local CUDA build | timed whisper-cli with `-sns --prompt` |
| turbo warm transcript | whisper-server + curl /inference | local | start server, curl multipart, measure |
| Daemon CLI path | dictation `_transcribe` | config backend=cli | invoke via unit test or thin script if added; else manual config + check |
| Daemon server path | dictation → HTTP → server | backend=server | same |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| Missing model file | model path absent | notify; no paste | L2 | rename/missing | check.sh / daemon notify |
| Server refused | nothing on port | fallback to CLI (if model present) | L2 | stop server | force request |
| Empty / short WAV | &lt; MIN_RECORD_SEC | existing too-short path | L2 | existing | unchanged |
| Second daemon | flock held | exit 0 | L2 | start twice | already covered |
| Blank / silence hallucination | quiet wav | `-sns` + model; may still empty → notify | L2 | optional | document residual |
| Server returns empty text | API ok, no speech | notify no speech | L2 | blank wav | |

### Human-only validation

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| Live Ctrl+Space dictation | Needs mic + focused editor | Speak a jargon sentence twice (cli small, then server turbo) | Text matches intent; single paste | Set model back to tiny/small; stop server unit |

---

## 6. Temporary scaffolding

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Proposed disposition |
|---|---|---|---|---|
| Phase 0 timing shell notes in plan | Evidence | Low | after P0 APPROVE | keep in plan only |
| Optional `/tmp` server for probes | Timing | None | after P0 | kill process |

---

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Amend plan / replace plan / supersede ADR |
|---|---|---|---|
| Turbo OOM / CUDA fail | whisper-cli crash / nvidia error | Stay on small.en CLI; document turbo unsupported | Amend — drop default turbo |
| Server API incompatible | curl fails / unparseable | Keep CLI-only Option 1; defer Option 2 | Amend or replace |
| Download fails | script error | Manual HF URL; or use non-quant `small.en` | Amend model name |
| Warm not faster | timings | Still ship server for RAM residency if load&gt;2s; else CLI-only | Amend P2 scope |

---

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Option 1: small.en + accuracy flags | P1 | model file + CLI flags in daemon |
| Option 2: turbo + warm path | P2 | server unit + backend=server works |
| Automated implement-plan through-completion | This plan Active | Phase outcomes recorded |
| No commit/PR | Execution | git commit not run unless asked |

---

## 9. Execution checklist and outcomes

- [x] Phase 0 evidence gathered
- [x] Phase 0 independent review approved
- [x] Pattern inventory reconciled after Phase 0
- [x] Each implementation phase completed
- [x] Happy-path integration passes
- [x] Edge-case matrix passes (server→CLI fallback L2)
- [x] Blast-radius invariants pass (singleton + single paste path unchanged)
- [x] Human-only gates completed or explicitly pending — **complete:** user confirmed live Ctrl+Space accuracy (2026-07-12)
- [x] Scaffolding disposition decided (Phase 0 server killed; no leftover scaffolds)
- [x] Validation outcomes recorded

### Validation outcomes (2026-07-12)

| Check | Result |
|---|---|
| `small.en` CLI jfk | OK (~0.9s) |
| `large-v3-turbo-q8_0` warm `/inference` | OK (~0.24s) |
| Daemon `_transcribe_server` / `_transcribe_cli` / fallback | ALL L2 ASSERTS PASSED |
| `bash scripts/dictation/check.sh` | All checks passed |
| systemd both units active | OK |

**Status → Complete** (human E2E remaining as explicit pending gate).

### Full review

| Round | Conformance | Functional | Verdict |
|---|---|---|---|
| 1 | FAIL | PASS | CHANGES_REQUIRED |
| 2 | FAIL (docs) | PASS | CHANGES_REQUIRED |
| 3 | PASS | PASS (R2) | **APPROVE** |
