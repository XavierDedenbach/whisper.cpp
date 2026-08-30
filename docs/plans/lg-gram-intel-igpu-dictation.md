# Implementation Plan: Accelerate LG Gram dictation on the Intel Iris Xe iGPU

**Status:** Complete — service ready and accepted in live use
**Approval authority:** Human — chat authorization on 2026-08-27 to run a bounded SYCL Level Zero proof with a warm server and the existing model
**Activation authority:** Human — chat authorization on 2026-08-27 to implement through completion, configure this LG Gram's user services for SYCL warm-server operation, and relaunch them
**ADR(s):** no-ADR — this is a reversible, machine-local build/runtime selection. It does not change whisper.cpp's public API or establish a durable cross-machine architecture.
**Epic / execution unit:** none
**Linear project:** none — local workstation operation
**Primary Linear issue:** none — local workstation operation
**Material cutover:** no — activation changes one user service configuration and has an immediate CPU-build rollback
**Cutover plan dependency:** none
**Routine deployment phase:** none — local service activation is included as a human-gated execution phase
**Supersedes:** none
**Superseded by:** none
**Target repo:** `/home/linu_x/Documents/git/whisper.cpp`
**Execution mode:** manual
**Phase 0 gate:** human
**Maximum Phase 0 rounds:** 3
**Authorized phases:** through-completion
**Context strategy:** current checkout with side-by-side build directories
**Scope:**

**In**

- Use the LG Gram's Intel Raptor Lake-P Iris Xe integrated GPU for whisper.cpp inference.
- Preserve the working CPU/OpenBLAS binary and service as the rollback path.
- Install the Intel compute runtime and minimal oneAPI C++/SYCL toolchain, then probe the upstream-supported SYCL backend first as explicitly requested.
- Keep Vulkan as a conditional later probe requiring human review if SYCL is unstable or insufficient.
- Add an explicit build-directory/backend selection to the dictation scripts so CPU and GPU builds can coexist.
- Benchmark correctness, latency, stability, GPU use, thermals, and service behavior before activation.

**Out**

- Replacing Ubuntu's working `i915` kernel driver, installing Intel DKMS kernel modules, changing GRUB GPU hangcheck settings, or switching to the experimental `xe` kernel driver.
- Changing the Whisper model or vocabulary while comparing backends.
- Removing the CPU build.
- OpenVINO model conversion unless both full-model GPU backends fail; OpenVINO is a replan option because it offloads only the encoder and adds a separate model artifact lifecycle.
- Commit, push, PR, or changes to this repository's tracked product behavior during Phase 0.

---

## 1. Observable outcome and invariants

### End-to-end outcome

The existing Ctrl+Space dictation flow runs the same `small.en` model and vocabulary through an Intel Iris Xe GPU-backed whisper.cpp binary. On the fixed 11-second JFK sample, the primary target is a warm-server median at or below 2.00 seconds. A result above 2.00 seconds can proceed only if it still reduces median end-to-end inference time by at least 35% from the recorded 5.17-second CPU CLI baseline (no more than 3.36 seconds), preserves the transcript, and completes 20 consecutive runs without a GPU reset, device loss, crash, or silent CPU fallback.

The winning GPU build runs from a side-by-side build directory. Setting `WHISPER_BUILD_DIR="build"` and restarting the user service restores the current CPU/OpenBLAS path without rebuilding or uninstalling anything.

### Baseline evidence (2026-08-27)

| Measurement | Result |
|---|---|
| CPU/OpenBLAS CLI, three fresh processes | 4.87 s, 5.17 s, 5.66 s; median 5.17 s |
| CPU/OpenBLAS warm server, three requests | 4.68 s, 4.82 s, 5.34 s; median 4.82 s |
| Warm-server model load | 0.30 s |
| Warm-server encoder | 3.55 s per run |
| Warm-server decoder | approximately 1.29 s per run |
| Conclusion | Model residency saves little; inference compute is the dominant delay |

### Blast-radius invariants

| Affected contract | Existing behavior | Characterization test | Allowed change |
|---|---|---|---|
| Hotkey and tray | Ctrl+Space toggles recording; silver/red tray LED | Active service plus manual start/stop | None |
| Text insertion | One clipboard paste into the focused application | Existing singleton and manual E2E | None |
| Model and vocabulary | `small.en`, shared vocabulary plus optional local overlay | Same model hash and normalized transcript | None during backend comparison |
| CPU fallback | `/build/bin/whisper-cli` and `whisper-server` work | Existing `check.sh` and JFK sample | Must remain intact |
| Autostart | One systemd user daemon | `systemctl --user is-enabled/is-active` and singleton check | Selected binary/runtime may change only through config |
| GPU selection | No GPU backend in current binary | Startup log and backend registry | Must explicitly identify Iris Xe; llvmpipe and silent CPU fallback are failures |
| System graphics | Ubuntu desktop uses stock `i915` | Current desktop session and kernel log | No kernel-driver or GRUB changes |
| Other fork machines | CPU remains the install default; CUDA/SYCL builds are side-by-side | Runtime selection tests for CPU, CUDA, and SYCL | No machine-local config overwrite or forced accelerator |

## Evidence archive: Phase 0 prototype

- **Form:** bounded early plan prototype
- **Status:** complete; SYCL was selected, implemented, and activated
- **Uncertainty resolved:** whether Vulkan or Intel SYCL provides a meaningful and stable latency improvement on this exact Raptor Lake-P Iris Xe device
- **Target environment/boundary:** native Ubuntu 24.04.4 X11 session, stock `i915`, current `small.en` model, production whisper.cpp binaries
- **Budget:** one execution session; at most 10 GB additional disk; no DKMS, kernel, GRUB, or display-stack changes
- **Stop/exit criteria:** a backend passes all promotion thresholds; both variants fail; any GPU reset/display instability occurs; or the package/build budget would be exceeded
- **Report:** append evidence to Phase 0 Round 1 below

| Variant | Why credible | Experiment | Required evidence / threshold |
|---|---|---|---|
| Intel SYCL first | whisper.cpp documents Intel iGPU support; Intel documents Iris Xe and Ubuntu 24.04 support | Install compute userspace + minimal oneAPI C++/SYCL components, build `build-sycl`, force Level Zero GPU, benchmark a warm server | `sycl-ls` plus the CLI backend log show Iris Xe Level Zero; primary median <=2.00 s; minimum promotion median <=3.36 s; 20/20 stable |
| Vulkan later, conditional | Mesa exposes `Intel Iris Xe Graphics (RPL-P)` as Vulkan GPU0 | Skipped because SYCL passed the primary latency and stability gates | No Vulkan build was created |

- **Artifact disposition:** retain the CPU build and selected SYCL build; no losing accelerator build was created
- **Findings update:** SYCL passed with the unchanged model, so precision, model, and Vulkan experiments were not needed

## 2. Phase 0 — risk-reduction portfolio

| Assumption | Consequence if false | Promising leads | Discriminating validation | Alternate probe | Pass/fail threshold |
|---|---|---|---|---|---|
| A1: Mesa Vulkan can execute whisper.cpp correctly on Raptor Lake-P | Lowest-impact route unavailable | Vulkan 1.4.318 device inventory; `GGML_VULKAN` backend | Build/run with `GGML_VK_VISIBLE_DEVICES=0`; inspect startup log and transcript | Vulkan validation build | Iris Xe selected, non-empty correct transcript, no llvmpipe/CPU fallback |
| A2: GPU execution materially improves user latency | System complexity produces no useful benefit | Vulkan full-model offload; SYCL Level Zero; OpenVINO encoder fallback | Three timed JFK runs after one warmup, compared with 5.17 s baseline | Five live dictation utterances | Median <=3.36 s and no transcript regression |
| A3: GPU backend is stable under repeated short jobs | Interactive dictation becomes less reliable | 20-run soak; kernel log; device monitor | Repeated CLI and server requests; inspect exit codes and GPU errors | Disable flash attention or add `--no-mmap` for a bounded retry | 20/20 passes, no GPU reset/hang/device loss |
| A4: SYCL userspace can be installed without replacing the display driver | Toolchain path risks the desktop stack | Ubuntu compute-runtime packages; Intel oneAPI C++ Essentials/minimal components | Simulate package transaction, exclude DKMS/kernel packages, then verify Level Zero | Full `intel-oneapi-toolkit` only after size/replacement review | No kernel/GRUB/display packages removed or replaced; `sycl-ls` shows Level Zero Iris Xe |
| A5: systemd can load the selected runtime consistently | Manual shell works but login service fails | Backend-aware launcher; source `/opt/intel/oneapi/setvars.sh` for SYCL | Start service from a clean user manager environment and inspect `ldd`/journal | Explicit generated environment file | No missing libraries; selected GPU appears in service log after reboot/login simulation |
| A6: CPU and GPU builds can coexist safely | Rollback requires rebuilding under pressure | Separate build directories plus `WHISPER_BUILD_DIR` | Switch config CPU -> GPU -> CPU and run health check each time | Absolute `WHISPER_BIN_DIR` | Both paths remain executable; rollback completes with one config edit + restart |

### Archived Phase 0 evidence and review

The following inventory records the pre-implementation state at the evidence gate; later sections record the completed integration and activation.

#### Round 1

##### Evidence inventory

| Assumption | Critical sub-claims | Evidence gathered | Outcome | Coverage & proxy risk | Validation confidence | Remaining work |
|---|---|---|---|---|---|---|
| A1 | Vulkan driver sees the real Intel iGPU; build dependencies are bounded | `vulkaninfo --summary`: GPU0 Intel Iris Xe RPL-P, Mesa 25.2.8, Vulkan 1.4.318; `glslc` and `spirv-headers` absent | Deferred by explicit SYCL-first authority | Vulkan was not built or run in this phase | Medium inventory / no execution | None unless SYCL is rejected |
| A2 | Current delay is inference compute rather than cold model load; SYCL materially improves it | CPU CLI median 5.17 s; CPU warm-server median 4.82 s. SYCL warm-server median 1.433 s across the first three measured requests and 1.501 s across a 20-run soak | Supported; primary <=2.00 s target passed | JFK is clean studio speech, not live mic | High synthetic / live acceptance later passed | Completed after integration |
| A3 | Selected backend survives repeated use | 20/20 warm-server requests returned the exact normalized transcript; range 1.435-1.556 s; no GPU reset, hang, device loss, or crash after runtime upgrade | Supported | One session rather than multi-day operation | High for Phase 0 | Production soak after integration |
| A4 | SYCL userspace can be installed without replacing the display driver | Transaction added 56 oneAPI/Ubuntu packages (1.359 GB download, 5.643 GB installed), then upgraded six compute packages and replaced old IGC ABI packages through Intel's current client-GPU PPA; no kernel, GRUB, Mesa display, or DKMS change. `sycl-ls` reports Level Zero Iris Xe | Supported | Intel PPA is an additional package source and should remain documented | High | Retain package inventory and rollback notes |
| A5 | A persistent warm server can load and use the runtime consistently | Temporary server loaded `SYCL0`, Level Zero Iris Xe, oneDNN, oneMKL, and served 24 correct requests. A fresh server's cached first request was 2.206 s and its second was 1.454 s | Supported in an explicit oneAPI shell; the systemd path had not yet met the contract | Clean user-manager environment and startup warmup were not yet integrated | High prototype / later production validation passed | Completed in P1 |
| A6 | Separate build selection can preserve rollback | `build-sycl/` coexists with healthy `build/`; production config still has no build override and the CPU CLI service remained active throughout | Supported manually; configurable selection unimplemented | Requires additive script/config work | High | Implement and prove CPU -> GPU -> CPU |

**Round summary:** The authorized FP32 SYCL Level Zero prototype passed its Phase 0 promotion thresholds with the unchanged `small.en` model. The backend log names `Intel Iris Xe Graphics` on `[level_zero:gpu:0]`, and steady warm-server latency is approximately 1.5 seconds. The key operational finding was that the server must perform one startup inference before being marked ready: the first-ever request spent 14.188 seconds compiling/warming kernels, a later process's cached first request took 2.206 seconds, and the following request took 1.454 seconds. Production activation occurred only after this gate passed.

##### SYCL build and benchmark evidence

| Evidence | Result |
|---|---|
| Compiler/toolchain | Intel oneAPI DPC++/C++ 2026.1.1; oneMKL 2026.1.0; oneDNN 2026.0.2 |
| GPU compute runtime | Intel Graphics Compute Runtime 26.27.39122.14; Level Zero loader 1.32.0 |
| Device discovery | `[level_zero:gpu:0] Intel Iris Xe Graphics`, 96 compute units, driver `1.15.39122+14` |
| Side-by-side build | `build-sycl/`, FP32, `GGML_SYCL=ON`, Level Zero ON, oneDNN ON, oneMKL, flash attention ON |
| Device proof | CLI backend registry selected `SYCL0`; checkout has no `ls-sycl-device` target, so the runtime registry and `sycl-ls` supplied equivalent proof |
| Fresh-process behavior | Initial run 17.06 s while kernels initialized; subsequent CLI wall time 4.04 s, internal total 3.874 s |
| Initial server warmup | 14.188 s, correct transcript |
| Warm server, three runs | 1.430 s, 1.441 s, 1.433 s; median 1.433 s |
| Warm server soak | 20/20 correct; median 1.501 s; range 1.435-1.556 s |
| Server restart | First cached request 2.206 s; second request 1.454 s |
| Accuracy | Exact normalized JFK transcript on all measured and soak requests |
| Kernel/runtime health | No GPU reset, hang, device loss, or post-upgrade crash. One `sycl-ls` userspace segfault occurred with Ubuntu's old 2023 runtime before replacement and did not recur |
| Pre-activation state | `whisper-dictation.service` was active/enabled on `WHISPER_BACKEND="cli"`, default CPU `build/`, `small.en`, 8 threads |

##### Review

**Gate:** human
**Verdict:** Phase 0 evidence passed; human approval subsequently authorized P1-P4 integration and activation

> **Phase 0 status — approved and promoted to the completed implementation below.**

## 3. Existing patterns and ownership

| Concern | Searches/files read | Existing anchor | Candidate decision | Owner/disposition |
|---|---|---|---|---|
| Intel backend | `README_sycl.md`, `ggml/src/ggml-sycl/` | `GGML_SYCL`, Level Zero, oneMKL/oneDNN | Selected by human after passing the SYCL-first Phase 0 gate | `build-sycl/`, runtime helper, config and docs |
| Vulkan backend | `ggml/src/ggml-vulkan/`, CMake options | `GGML_VULKAN`, `GGML_VK_VISIBLE_DEVICES` | Deferred; no longer needed because SYCL passed the primary target | No implementation |
| Binary resolution | `dictation.py`, `run-server.sh`, `check.sh`, `test-mic.sh` | Hard-coded `${ROOT}/build/bin` | Add `WHISPER_BUILD_DIR`, default `build` | Dictation scripts |
| Runtime environment | Generated launchers and user units | CPU needs no special environment | Source oneAPI only when backend is SYCL; export explicit device selector | Launchers/install script |
| Install flow | `scripts/dictation/install.sh` | CPU/OpenBLAS one-shot install | Keep default unchanged; add an explicit accelerator build path | New helper or opt-in flag |
| Service fallback | systemd user units; CLI fallback from server | CPU binaries in `build/` | Preserve CPU build and document one-command rollback | Config + health check |
| Validation | `check.sh`, `samples/jfk.wav`, CTest | Functional CPU sample | Add selected-build and actual-GPU assertions | Check script/tests |

## 4. Execution phases and units

| Unit | Deliverable | Authority ref | Files/areas | Depends on | First failing test | Green + regression verification | Effort |
|---|---|---|---|---|---|---|---|
| P0a | SYCL prerequisite transaction and side-by-side proof build | Approved Phase 0 | Intel compute userspace; oneAPI; `build-sycl/` | Plan approval | `clinfo` has zero platforms and `sycl-ls`/`icpx` are missing | Level Zero Iris Xe visible; SYCL targets build; same-model CLI and warm-server benchmark/soak | L |
| P0b | Conditional Vulkan prerequisite transaction and proof build | Deferred after P0a passed | System packages; `build-vulkan/` | P0a verdict | Not run | Skipped; SYCL satisfied all promotion gates | M |
| P1 | Configurable build directory and accelerator environment | Approved implementation after P0 backend choice | `dictation.py`, `run-server.sh`, `check.sh`, `test-mic.sh`, config/install launchers | Backend passes P0 | New resolver/launcher tests fail against hard-coded `build/bin` | Unit tests plus CPU -> GPU -> CPU integration | M |
| P2 | Reproducible Intel-GPU build path and documentation | Approved implementation | `build-sycl.sh`; dictation README | P1 | Backend build/check command is not represented by repo tooling | Helper configures the supported CLI/server targets; shell and fake-tool integration checks | M |
| P3 | Production-path validation | Approved implementation | Selected build, models, systemd user units | P1-P2 | Existing check cannot prove selected GPU | Health check proves binary, runtime libraries, Iris Xe selection, transcript, latency and soak | M |
| P4 | Local activation and operator handoff | Fresh explicit activation authority | User config and systemd user service | P3 pass | CPU config remains active | Set selected build/backend, restart, real-binary/systemd validation, one-edit rollback rehearsal; leave live utterances as post-handoff acceptance | S |

> **Execution outcome — P0a and P1-P4 complete. P0b was skipped because SYCL passed every promotion gate.**

### P1-P4 implementation evidence (2026-08-27)

| Unit | RED evidence before production edit | GREEN / integration evidence | Outcome |
|---|---|---|---|
| P1 build selection | `test_dictation_build_directory_override` resolved `build/bin` instead of configured `build-sycl/bin` | Seven focused tests pass; default `build` and `build-sycl` override both resolve correctly | Complete |
| P1 runtime environment | `test_runtime_env_selects_sycl_and_sources_oneapi` observed empty bin/device values and no oneAPI environment | `runtime-env.sh` sources oneAPI only for SYCL, exports `level_zero:gpu` and device 0, and leaves CPU independent | Complete |
| P1 warm readiness | `test_run_server_warms_before_notifying_ready` failed because the old wrapper looked only in `build/bin` and had no warmup/notify lifecycle | Fake composed test proves server start -> HTTP probe -> warmup -> readiness notification order | Complete |
| P2 build/installer/service | Build command was not represented by repo tooling; launcher test lacked runtime loading; restart-order test showed daemon before server; unit test lacked `Type=notify` | `build-sycl.sh` configures the supported CLI/server targets; generated launcher loads shared runtime; installer warms server before daemon; notify unit has 180 s startup timeout and mixed shutdown | Complete |
| P3 real wrapper | Existing health check could not select or identify an accelerator | Isolated real wrapper selected `[level_zero:gpu:0] Intel Iris Xe Graphics`, warmed, and returned the exact transcript in 1.499 s; final check fails closed for inactive, unreachable, or mismatched server binaries | Complete |
| P4 activation/rollback | CPU configuration was initially active | Installed user service selected `build-sycl`, five exact requests ranged 1.463-1.570 s, final probe 1.442 s; CPU -> GPU -> CPU -> GPU rehearsal passed | Complete; subjective live speech remains human-only |

Regression evidence:

- `python3 -m unittest scripts.dictation.test_runtime_integration`: 21/21 pass, including CPU/CUDA/SYCL selection, nonzero Intel Level Zero devices, optional normalized device-name enforcement, strict shared server-response validation, clean server shutdown, caller override, and server failure negatives.
- `bash -n scripts/dictation/*.sh` and Python `py_compile`: pass.
- `ctest --test-dir build -R '^test-whisper-cli-small\.en$' --output-on-failure`: 1/1 pass. Other model-specific CTests are not runnable because those model files are intentionally not installed.
- `bash scripts/dictation/check.sh`: all checks pass on both CPU rollback and final SYCL/server configurations.
- Final systemd state: one daemon, warm server active/enabled, loopback-only `127.0.0.1:8178`, status `Whisper model warm on sycl`.
- Kernel journal since activation: no GPU hang, reset, fault, or device-loss event.
- Two consecutive real service restarts prove the updated wrapper shuts down cleanly with no orphaned child or failed unit result on the second restart.
- Operator validation artifact: `/home/linu_x/Documents/git/whisper.cpp/tmp/whisper-dictation-sycl-validation.json` (gitignored).

### Evidence archive: candidate build commands

#### Intel SYCL first probe

1. Simulate and review installation of Ubuntu's client compute packages:

```bash
sudo apt install --simulate \
  intel-opencl-icd libze-intel-gpu1 libze1 libze-dev
```

2. Add Intel's signed oneAPI APT source using the current official instructions. Prefer the smallest C++/SYCL bundle or compiler + oneMKL + oneDNN package set that satisfies this checkout. Simulate it and record download/installed size before installation. Use the full `intel-oneapi-toolkit` only if the minimal package set does not satisfy CMake.

3. Do not install `intel-i915-dkms`, replace the kernel, or modify GRUB. The stock kernel already drives this client iGPU correctly.

4. Verify compute discovery before building:

```bash
source /opt/intel/oneapi/setvars.sh
clinfo -l
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
```

5. Build FP32 in a separate directory with the checked-in helper:

```bash
bash scripts/dictation/build-sycl.sh
```

6. Prove the Level Zero GPU path with device discovery and the actual selected inference binary:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
ONEAPI_DEVICE_SELECTOR=level_zero:gpu GGML_SYCL_DEVICE=0 \
  build-sycl/bin/whisper-cli \
  -m models/ggml-small.en.bin -f samples/jfk.wav \
  -nt -np -t 8 -l en -sns
```

Only if FP32 is correct and stable but misses the latency threshold, stop for review. FP16, a smaller/quantized model, and Vulkan are not part of the currently authorized proof.

#### Conditional Vulkan probe after review

System delta, after an APT simulation confirms no removals:

```bash
sudo apt install glslc spirv-headers
```

Side-by-side build:

```bash
cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_BLAS=ON \
  -DWHISPER_SDL2=OFF
cmake --build build-vulkan -j16 --target whisper-cli whisper-server
```

Forced-device proof:

```bash
GGML_VK_VISIBLE_DEVICES=0 build-vulkan/bin/whisper-cli \
  -m models/ggml-small.en.bin -f samples/jfk.wav \
  -nt -np -t 8 -l en -sns
```

## 5. Test strategy

### TDD and coverage contract

- **Coverage baseline command/result:** unavailable for `scripts/dictation/`; there is no configured Python/shell coverage threshold in this checkout.
- **Coverage completion gate:** no changed behavior without an automated requirement-level test; core CTest must remain green. No test, assertion, or exclusion may be weakened.

| Behavior/requirement | Test level and path | RED command and expected failure | GREEN/regression command | Coverage expectation |
|---|---|---|---|---|
| Build directory defaults to `build` and accepts side-by-side selection | Python unit test under `scripts/dictation/` | Override test initially resolved `build/bin` instead of the selected build | `python3 -m unittest discover -s scripts/dictation -p 'test_*.py'` | CPU, CUDA, SYCL, absolute, and relative build selection |
| Launcher prepares SYCL runtime only for SYCL | Shell/Python launcher test | Fake selected SYCL build has no oneAPI environment | Test launcher with a controlled fake `setvars.sh`; CPU/Vulkan path must not source it | CPU, Vulkan, SYCL and missing-oneAPI branches |
| Check script rejects silent CPU or llvmpipe fallback | L2 real-binary integration | Current `check.sh` passes without identifying an accelerator | Run selected build on JFK and assert Iris Xe backend/device in captured log | Selected GPU, CPU rollback, invalid device |
| CPU rollback remains functional | L2 real-binary integration | Characterization first; failure would indicate regression | CPU JFK sample + `check.sh` after GPU tests | Full rollback path |

### Realism target

**Level 2:** the production whisper.cpp CLI/server binaries, current `small.en` model, actual Iris Xe hardware, and the same systemd user services with synthetic JFK audio. Level 1 is reserved for the post-handoff five live Ctrl+Space utterances because it requires speech and a focused application; it is an operator acceptance check rather than a readiness gate.

### Happy-path integration

| Behavior | Systems composed | Environment | Command/evidence |
|---|---|---|---|
| GPU CLI transcript | GPU driver + selected runtime + ggml backend + model | Native X11 session | Three timed JFK runs; backend/device log; normalized transcript |
| GPU warm server | Same binary/runtime through HTTP server | Temporary loopback port | Three requests after warmup; compare with CLI and CPU baseline |
| Dictation daemon | Recorder -> selected CLI/server -> vocabulary -> paste | systemd user service | `check.sh`, journal, then human live utterances |
| Rollback | Config -> CPU binary -> warm server and daemon | Clean user-manager environment | With `WHISPER_ACCELERATOR=auto`, change only `WHISPER_BUILD_DIR=build`, restart both services, then run JFK + hotkey smoke test |

### Edge-case and failure matrix

| Scenario | Boundary/failure | Expected behavior | Test level | Environment | Command |
|---|---|---|---|---|---|
| llvmpipe selected | Vulkan physical device order/env wrong | Fail preflight; never benchmark/promote | L2 | Vulkan | Inspect device log with `GGML_VK_VISIBLE_DEVICES=0` |
| OpenCL works but Level Zero is absent | Incomplete Intel runtime | Do not build/promote SYCL until Level Zero GPU is visible | L2 | SYCL preflight | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls` |
| Missing oneAPI libraries in systemd | Interactive shell leaked environment | Service fails clearly and CPU rollback remains available | L2 | Clean user manager | `systemctl --user restart`; `journalctl`; `ldd` |
| Model mmap hangs | Known SYCL startup issue | One bounded retry with `--no-mmap`; record result | L2 | SYCL | Timed JFK command |
| Vulkan device loss or GPU reset | Driver/backend instability | Stop probe, restore CPU, retain logs; do not promote | L2 | 20-run soak | Exit codes + `journalctl -k` delta |
| FP16 changes transcript | Reduced precision | Keep FP32 or reject SYCL; no accuracy trade without review | L2 | Separate FP16 build | Normalized transcript diff |
| Thermal throttling | iGPU and CPU share package power | Record and reject misleading first-run-only gain | L2 | AC power, repeated runs | `intel_gpu_top`/temperature sample plus run timings |
| GPU service fails after login | Runtime env not persisted | CPU service rollback; fix launcher before retry | L2 | User manager | Stop/start and login-equivalent clean environment |

### Human-only post-handoff acceptance

The implementation-readiness gate is the automated real-binary/systemd validation above. Live microphone quality and subjective feel remain an operator acceptance check after handoff because they require the user's voice and focused desktop interaction; they do not block relaunch readiness.

The user confirmed on 2026-08-30 that live dictation is “much better,” completing this acceptance check.

| Gate | Why not automated | Exact procedure | Expected evidence | Rollback |
|---|---|---|---|---|
| Subjective latency and live accuracy | Requires speech, focus, and user preference | Dictate five representative utterances on the selected GPU, including vocabulary terms | GPU feels materially faster; text and single-paste behavior unchanged | Restore `WHISPER_BUILD_DIR=build`; restart both services |
| System package escalation | Adds Intel or shader toolchain packages | Review APT simulation, source, download size, installed size, removals/upgrades | No kernel/display replacement; change is bounded | Do not install; or remove only newly added compute/toolchain packages after inventory |

## 6. Scaffolding disposition

| Scaffold | Purpose | Maintained value | Cleanup checkpoint | Final disposition |
|---|---|---|---|---|
| `build-vulkan/` | Conditional GPU proof | None; never created | Backend decision | Not applicable; SYCL won |
| `build-sycl/` | Selected operational backend | High | In active use | Retain |
| Optional `build-sycl-f16/` | Precision experiment | None; never created | Backend decision | Not applicable; FP32 met target |
| Temporary benchmark server | Compare warm path | None | End of benchmark set | Deleted/stopped |
| Timing/kernel/GPU-monitor logs | Evidence | Summary only | Plan completion | Summarized here; raw temporary files discarded |

## 7. Fallbacks and replan triggers

| Blocker/signal | Evidence | Recovery or next investigation | Amend plan / replace plan / supersede ADR |
|---|---|---|---|
| Vulkan meets gates | Correct device, latency, soak | Select Vulkan; skip oneAPI entirely | Amend with accepted P0 evidence |
| Vulkan device loss or misses latency gate | Logs/timings | Proceed to SYCL without activating Vulkan | Continue planned conditional path |
| SYCL package simulation would replace kernel/display stack | APT transaction | Stop; use minimal packages/container research or remain on Vulkan/CPU | Amend or replace plan |
| SYCL sees only OpenCL CPU or no Level Zero GPU | `sycl-ls` | Inspect ICD/Level Zero packages and permissions; one bounded repair round | Amend Phase 0 evidence |
| Both full-model backends fail | Two failed evidence sets | Create replacement plan for OpenVINO GPU encoder offload, justified by encoder's 3.55 s share | Replace plan; no ADR needed unless interface scope expands |
| Gain is under 35% or live UX is not better | Benchmark/human gate | Keep CPU active; consider model quantization or smaller model as separate accuracy/latency trade | Close without activation or replace plan |
| GPU reset affects desktop | Kernel journal/display symptoms | Immediately restore CPU and stop GPU backend; no kernel tuning under this plan | Stop and require new reviewed approach |

## 8. Traceability

| Authority requirement | Artifact/unit | Verification |
|---|---|---|
| Scope the system changes | Baseline, Phase 0, candidate commands | Hardware, runtime, package and service inventory recorded |
| Produce a build plan | P0-P4 | Side-by-side configure/build/verify/activate sequence |
| Use Intel integrated graphics | P0a or P0b | Startup log names Iris Xe; monitor shows GPU activity |
| Preserve working dictation | Invariants, P1, P4 | CPU rollback test and unchanged hotkey/paste behavior |
| Improve latency | P0/P3 | Median <=3.36 s and human live comparison |

## 9. Primary Linear issue

- **Identity:** none — local workstation operation
- **Reconciliation state:** not applicable
- **Desired title:** Accelerate LG Gram Whisper dictation on Intel Iris Xe
- **High-level description:** Build and validate a reversible Intel-iGPU backend, select Vulkan or SYCL through measured gates, and activate it without losing the CPU fallback.

## 10. Execution checklist and outcomes

- [x] Current hardware, permissions, runtime and CPU baseline inventoried
- [x] Plan approved and P0a explicitly authorized
- [x] APT transactions simulated and reviewed against the authorized package/build budget before installation
- [x] SYCL Level Zero proof built and evaluated with `small.en` and a warm server
- [x] Conditional Vulkan proof explicitly deferred because SYCL passed the primary gate
- [x] Phase 0 SYCL choice and evidence approved
- [x] Exactly one GPU backend selected
- [x] Configurable build/runtime selection implemented test-first
- [x] Affected core CTest and dictation tests pass
- [x] Phase 0 correctness, latency and 20-run soak gates pass
- [x] Phase 0 kernel journal contains no new GPU reset/hang/device-loss event
- [x] Clean systemd environment starts and warms the selected backend
- [x] CPU rollback rehearsal passes
- [x] Post-handoff live acceptance passed; user confirmed dictation is much better on 2026-08-30
- [x] Final system package/build inventory and rollback commands recorded
- [x] Three independent full-review rounds completed; every finding was fixed and the complete gate suite rerun after the final correction

## Sources

- [whisper.cpp SYCL documentation](https://github.com/ggml-org/whisper.cpp/blob/master/README_sycl.md)
- [Intel oneAPI Toolkit installation guide for Linux 2026.1](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/overview.html)
- [Intel oneAPI APT installation](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-apt.html)
- [Intel DPC++/C++ compiler supported GPUs](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [Intel GPU driver and user-permission guidance](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-intel-gpu-drivers.html)
- [Intel client GPU packages for Ubuntu 24.04](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html)
