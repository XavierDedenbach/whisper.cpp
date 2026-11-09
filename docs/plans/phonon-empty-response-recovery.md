# Phonon empty-response recovery

Status: Active (auto-approved). Authority: restore working LG Gram dictation after
the user's reset did not resolve repeated transcription failures. Size: S.
Integration target: master; feature checkpoint only, no PR/merge or Spark rollout.
Primary tracker: not applicable to this bounded local repair. Material cutover: no.
Assurance: Assured for recovery behavior. Phase 0: approved, owner-run.

## Contract and evidence

The Phonon adapter distinguishes valid JSON with `text: ""` (empty recognition)
from malformed, HTTP-error, truncated or missing-text replies (`None`). The
daemon currently conflates them. Replaying three retained failing chunks on the
installed CPU model returned HTTP 200, zero text characters and truncated=false.
Every such chunk caused a server restart; newer speech recordings waited behind
that backlog. Replaying the newer recordings directly produced speech promptly.
No personal audio or transcript is included in this branch.

Required outcome: retry empty recognition once, then return it as no text if the
selected model/device is healthy. Do not restart a healthy server for empty
recognition. Keep actual inference errors retryable, retain every WAV, and let the
next speech chunk complete normally. Punctuation-only and Whisper behavior stay
unchanged. No capture, ordering, vocabulary, auth or model change.

## Work and proof

1. Add RED regressions to `scripts/dictation/test_phonon_backend.py` for repeated
   empty recognition without restart, empty-then-speech, empty-then-error, wrong
   health, and durable no-text followed by ordered speech with retained WAVs.
2. In `Dictation._transcribe`, after its two Phonon attempts, accept the adapter's
   empty-string sentinel only with a matching health check. Leave `None` and
   punctuation failure recovery intact. Update the README's response contract.
3. Run focused tests, dictation suite, Ruff, and real CPU retained-audio replay
   through the daemon's delivery/storage path with paste/notifications disabled.
4. Register the new feature worktree using the existing installer and unmodified
   local CPU config; allow durable recovery to drain; verify no restart storm and
   newer session transcripts. Never inject recovered text into another app.

Direct hotfix proof: actual captured fixture, localized empty-vs-error mismatch,
one correction using the existing no-text path. Falsification checks cover a real
transport failure after an empty response and unhealthy model identity. There is
no second credible correction needed to classify the already typed response.

Lessons lookup: company memory not_configured. Reuse existing local restart and
capture-ownership evidence, but sample warmup alone is not live dictation proof.
Access: local service and retained WAVs available; no active recorder when paused.
Rollback: original PR8 worktree/registration retained; audio is never removed.
Stop if replay exposes invalid/truncated responses being accepted or audio loss.

## Resume

Implementation is complete. Focused regressions passed: 25 tests. Ruff lint and
format passed for all dictation Python files. The added empty-response and
empty-after-restart regressions were observed RED before their corrections.
Actual CPU delivery/storage replay of four retained WAVs returned
`no_text, complete, no_text, complete` in order, with matching retained audio
hashes, a 66-character ordered transcript and zero server restarts (3.515 seconds).
The replay disabled worker startup, notifications and paste, not real inference.
The new source is registered locally; machine CPU config is preserved and the
old PR8 source remains available for rollback. Whole-suite check passed: 167
tests in 66.54 seconds under Xvfb; Bash syntax also passed. Independent review
and final backlog observation are pending. Human microphone and
target-field paste remain an operator acceptance step; automatic replay must
not claim that boundary passed.
Effort recording starts prospectively; initial diagnosis time and provider token
usage are unavailable, not zero. Harness/model setting recorded unknown where
the runtime exposes no exact model identity.
