# Recorder reacquisition could erase interior dictation

- **Date:** 2026-09-05
- **Area:** dictation
- **Fix:** `codex/numeric-continuity-hardening`

## What broke

Long dictation could deliver its first chunk and final partial chunk while losing
one or more chunks from the middle. The missing chunk was often reported as an
empty WAV even though the microphone had worked immediately before and after it.

The legacy rolling path treated each time slice as a separate recorder lifetime.
At every boundary it stopped one `parecord` process, finalized that WAV, and
started another process against the same source. A failed or delayed source
reacquisition could therefore leave an empty or incomplete file and an interval
for which no process owned microphone bytes. Persisting and retrying that file
could not recover audio that had never reached the application.

## Why it was possible

The design made recorder process lifetime and transcription chunk lifetime the
same boundary. It had no invariant that one reader continuously owned every byte
from the beginning of a recording until the user stopped, and no byte-level
accounting from recorder stdout through durable chunk WAVs.

The old tests exercised individual rollover and recovery events, but did not prove
that a multi-minute recording used one recorder PID or that concatenating every
retained chunk reproduced the complete captured PCM stream exactly once. A healthy
first chunk therefore did not imply that later source handoffs were safe.

## What a later design should not repeat

Keep hardware-source ownership independent from storage and inference boundaries.
One recorder/reader must own the live PCM stream continuously; application code may
partition that stream only after receiving the bytes. Every received sample must
belong to exactly one durable segment, including odd reads, shutdown drain, and the
final partial segment.

Recovery should operate on durable segments and may retry inference or delivery,
but it must never be presented as a substitute for continuous capture. A realistic
acceptance test should attest one recorder PID, byte-exact source partitioning,
ordered terminal states, no failed chunks, and one final paste over a recording
long enough to cross many segment boundaries.
