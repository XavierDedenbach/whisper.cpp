# Accepted risks

Risks this repository knowingly carries. Reviewers treat a matching finding as
accepted rather than reopening the implementation scope.

## Accepted risks

| ID | Risk (failure mode) | Why accepted | Accepted by | Date |
|---|---|---|---|---|
| R-1 | Startup recovery uses the existing FIFO, so a large old backlog can delay a newly finalized live chunk. | Automatic priority scheduling was explicitly classified as nice-to-have; exhaustive recovery keeps all audio durable. Revisit if startup recovery causes observed live latency. | Xavier Dedenbach | 2026-09-03 |
| R-2 | Crash-looped transcription attempts and `no_text` results do not implement a fully strict three-attempt state machine. | The requested outcome is preservation and continuation, which the retained WAV and transcript marker provide; richer retry-state hardening was explicitly deferred. Revisit if sessions repeatedly retry or require manual repair. | Xavier Dedenbach | 2026-09-03 |
| R-3 | SIGTERM handoff is bounded but lacks a process-faithful recorder/inference integration test. | The additional process simulation was explicitly classified as nice-to-have; focused lifecycle tests and systemd validation cover the maintained implementation. Revisit after an observed shutdown-loss incident. | Xavier Dedenbach | 2026-09-03 |
| R-4 | `small.en` can substitute or omit dictated content even when continuous capture, storage, ordering, and delivery are lossless. | The sealed 1–500 run retained every PCM byte and every chunk but recognized 483 values exactly, substituted 15, and omitted two. This model-level accuracy is acceptable for shipping the capture-reliability fix; retained WAVs permit later recovery or reprocessing. | Xavier Dedenbach | 2026-09-04 |

## Files under accepted risks

| Path | Risk IDs | Note |
|---|---|---|
| `scripts/dictation/dictation.py` | R-1, R-2, R-3, R-4 | Startup recovery FIFO, recovery-state handling, bounded shutdown, and model output limitations |
| `scripts/dictation/session_store.py` | R-1, R-2 | Recovery discovery and persisted chunk states |
| `scripts/dictation/test_runtime_integration.py` | R-3 | Shutdown coverage boundary |
| `docs/plans/dictation-temporal-segmentation.md` | R-4 | Human disposition of the sealed numeric-ASR result |
