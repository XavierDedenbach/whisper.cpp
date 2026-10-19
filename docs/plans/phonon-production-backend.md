# Implementation Plan: Configurable Phonon-2 dictation

**Status:** Complete — implementation; delivery validation pending
**Authority:** 2026-10-19 request to build Phonon-2 CPU on LG Gram, CUDA on Spark X,
configure the repository, ship-it-good to the fork, and activate on this laptop.
Spark owns its rollout; no remote Spark mutation. No chat archive requested.
**Spec:** S bounded contract below; one inference-adapter seam, existing capture,
queue, storage, and delivery architecture retained. No separate module spec.
**ADR:** Accepted ADR-001 audio ownership remains unchanged.
**Integration:** XavierDedenbach/whisper.cpp, declared default `master`.
**Assurance:** Assured delivery because runtime/recovery must remain intact.
**Primary tracker:** Not applicable to this personal fork/backend addition.
**Material cutover:** No fleet/data migration; explicit local activation after
merge, with configuration rollback and retained Whisper runtime. Spark deployment
and GPU hardware qualification remain its rollout owner's responsibility.
**Phase 0:** APPROVE — owner-run; named proof gates below precede dependent work.
**Lessons:** Company-memory lookup `not_configured`. Reuse ADR-001 and current
tests: one microphone owner, durable source chunks, ordered final paste, explicit
recovery, one daemon and one selected warm service. Preserve hardware-local config.

## Contract / observable Done

1. `DICTATION_ENGINE=whisper|phonon2` defaults to Whisper without changing any
   CPU/CUDA/SYCL profile. Phonon requires a warm local server and explicit
   `PHONON_DEVICE=cpu|cuda`; no silent cross-model or cross-device fallback.
2. Production Phonon requests use the existing shared/local dictionary as at
   most 25 canonical hotwords, keep phrase boundaries, validate typed responses,
   and use the existing corrections before durable completion. Invalid/failed
   requests remain retryable; no capture, ordering or paste implementation fork.
3. One managed server retains readiness warmup, restart/watchdog and login start.
   Checks verify Phonon identity and selected device, not just a listening port.
4. CPU installation is pinned and persistent outside disposable worktrees.
   CUDA supports a native ARM64 Spark build, host-loopback-only publication,
   GPU-required launch, and cleanup of only its own container.
5. Full affected tests, isolated desktop delivery/capture regressions, real CPU
   service/multiple-chunk inference proof, independent delivery review, CI,
   verified integration merge and exact feature cleanup. Activate the merged LG
   configuration when idle, verify services, health and retained transcript path.

## Owners / work units

- Adapter: `phonon_backend.py`, `vocab_prompt.py`, `Dictation._transcribe*`.
  Test first with real HTTP fixtures and malformed/empty/error/truncated output,
  canonical vocabulary, retry/restart and no Whisper fallback.
- Runtime: `run-server.sh`, `runtime-env.sh`, `install.sh`, `check.sh`, config and
  pinned CPU/CUDA installer/build inputs. Controlled executable fixtures prove
  launcher commands, default compatibility, readiness and lifecycle semantics;
  manifest checks establish NVIDIA base availability for both architectures.
- Shared pipeline: unchanged `temporal_audio.py`, `session_store.py`, X11 paste
  and clipboard ownership. Existing Xvfb regression suite and real inference
  exercise integration; never send test keys to the live desktop.
- Delivery: existing exploration worktree/branch; consolidate prototype-only
  scaffolding once in validate-for-pr. Keep personal data and model caches out
  of Git. Integration target is master, not a newly invented main/dev.

## Phase 0 evidence / remaining proof

- Local Phonon 0.2.11 CPU actual inference already measured against 51 retained
  chunks and independent published samples. Reuse that feasibility evidence;
  production-path tests still required before delivery.
- Upstream `phonon-cuda:1.0.8` manifest is AMD64-only. ARM64 cannot use it as-is.
  Add pinned upstream CUDA source build on NVIDIA PyTorch 25.09 multiarch base
  (manifest proves arm64 and amd64; NVIDIA Spark minimum is 25.09 / CUDA 13).
  No CUDA execution is claimed without Spark's real warmup/device acceptance.
- Runtime candidates preserve the existing `whisper-dictation-server.service`
  and common lifecycle owner. Model selection happens before any microphone
  capture, allowing the same restart/recovery contract.
- Default branch inspected/fetched: master; exploration is up to date with its
  base and has no merged/open PR. Primary checkout is unchanged.

## Stop / recovery / current frontier

Stop for changed accepted audio ownership, inaccessible required proof, failed
independent review launch, or a Spark CUDA build/runtime defect without a safe
supported path. Report hardware qualification separately; never label fixtures
as actual GPU inference. Feature checkpoints are recoverable, not deployment.

## Checkpoint 1 — integrated adapter and runtime

Adapter RED observed before implementation (missing engine/adapter/limit,
Whisper-only recovery and request shape). Adapter, fail-closed HTTP handling,
canonical dictionary limit, selected-device health, native CUDA launcher/build
path and installer are implemented. A failed middle-chunk fixture proves audio
remains recoverable while the later chunk completes.

An isolated production CPU runtime was installed successfully outside this
worktree and warmed on port 18180 using `run-server.sh`. Health proves Phonon-2,
CPU AVX2 C kernels and 12 threads; startup JFK inference succeeded. Live services
on 8178 remain unchanged. CPU/CUDA executable fixtures prove readiness and owned
shutdown, not actual GPU inference. NVIDIA base manifests prove ARM64 availability.

The first full regression run found an old test object without constructor state;
the backward-compatible Whisper class default repairs that boundary. The first
capture probe then exposed a real Whisper-asset prerequisite in both recorder
start paths. Focused RED tests reproduced it; Phonon now bypasses only that asset
check, preserving the shared recorder logic. The fresh 150-count capture used an
isolated Pulse sink and Xvfb, with no live paste; its completed result is below.

## Implementation proof / delivery handoff

All 158 affected tests and 54 subtests passed under isolated Xvfb. Bash syntax,
Ruff lint/format checks and Dockerfile `docker build --check` passed. Pytest now
uses an isolated HOME so a test daemon cannot recover the operator's history.

The fresh production-path 150-count run passed all ownership/delivery checks:
one recorder, 4,952,960 captured and durable PCM bytes with matching SHA256,
20 retained chunks (19 completed speech plus one silent tail), no failed chunks,
closed session, and one final delivery identical to the ordered store. Raw proof:
`/home/linu_x/.local/share/whisper-dictation/prototypes/phonon-production/dev-implemented-150-20261019T192820.695888Z/report.json`.
Its model identity is Phonon-2 / CPU from actual health. The probe's callback was
updated for the existing target-aware paste API; the earlier failed probe report
is retained and not misrepresented as a passing delivery test.

Strict recognition still first differs at 55, matching the already-accepted
prototype behavior; this is not a claim of exact numeric recognition. This dense
counting run had a 5.71 s median inference under current load. A bounded same-WAV,
same-25-hotword one-thread versus 12-thread check produced identical text hashes
and variable timing; 12 threads were not consistently worse. Keep the upstream
default rather than tune against this fixture. No uniform sub-two-second latency
claim is made. Temporary CPU test servers were stopped.

Consolidation removed the one-time prototype comparison scripts, preserving their
reports, private data and Git history. The maintained capture probe and production
vocabulary regressions remain. Round 1 found inherited-proxy exposure and an
initial selected-device validation gap; both were reproduced RED and repaired.
Phonon HTTP always bypasses proxies, and each chunk requires matching model/device
health before inference with at most one managed restart. All 18 adapter tests
and 11 subtests pass; real CPU warmup and three repeated daemon calls with 25
hotwords also pass with a deliberately unusable proxy and empty NO_PROXY.

Delivery validation continues; review rounds used: 2/3. Spark GPU execution and
final LG activation remain separate, not-yet-claimed evidence. No production
audio ownership code was forked.

Final consolidated regression suite: **157 passed, 54 subtests passed** (Xvfb).
The private clipboard fixture now uses `-noreset` after intermittent connection
loss between test methods; the preceding failed runs remain in the local review
ledger. All production capture and paste algorithms are unchanged. Ruff lint and
format, shell syntax and whitespace checks pass. Final independent review and
remote CI/merge are the remaining delivery gates before local activation.

Round 2 tightened the readiness contract: Phonon always completes real warmup
before READY, ignoring an inherited Whisper-only disable setting. Positive CPU
and CUDA launcher fixtures and a negative failed-inference fixture reproduced the
gap before repair. The production adapter/runtime suite passes: 19 tests and 11
subtests; lint, formatting, shell syntax and whitespace checks also pass.
unchanged ownership and desktop proof is reused. Architecture support claims now
distinguish manifest availability from launcher fixtures and actual hardware.
