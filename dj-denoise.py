#!/usr/bin/env python3
"""DPDFNet noise removal for dj: reads mono float32 (f32le) from stdin, writes to stdout.

Usage: dj-denoise.py MODEL [LIMIT_DB]

Audio must come at the model's native sample rate (dj takes care of that).
Processing is streamed, so memory doesn't grow with the recording length.

Segmented mode (set by `dj denoise -j N`):
  DJ_DENOISE_SKIP  extra output samples to skip on top of the model delay
                   (the overlap the caller fed in before the segment start)
  DJ_DENOISE_EMIT  exact number of output samples to emit (default: no limit)
"""

import os
import sys

import numpy as np
from dpdfnet import StreamEnhancer

CHUNK = 1 << 16  # samples per read


def main():
    model = sys.argv[1]
    limit = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    # Share of the original signal mixed into the cleaned one,
    # so noise is reduced by at most `limit` dB.
    alpha = 10 ** (-limit / 20) if limit is not None else 0.0

    extra_skip = int(os.environ.get("DJ_DENOISE_SKIP") or 0)
    emit_total = int(os.environ.get("DJ_DENOISE_EMIT") or -1)

    enhancer = StreamEnhancer(model=model)
    # The stream lags the input by two model windows (dpdfnet's offline mode
    # trims the same delay). Without compensation audio drifts from the video.
    delay = 2 * enhancer._win_len
    # Output samples to skip: model warm-up + the overlap the caller fed in.
    skip = delay + extra_skip

    src, dst = sys.stdin.buffer, sys.stdout.buffer
    noisy = np.zeros(0, np.float32)  # input not yet matched with output (for alpha)
    # The overlap that was skipped in the output corresponds to the first
    # `extra_skip` input samples, which must also be dropped from `noisy`.
    noisy_skip = extra_skip
    left = 0  # how many samples may still be emitted (unsegmented mode)
    emitted = 0  # total samples already written

    def emit(y):
        nonlocal noisy, skip, noisy_skip, left, emitted
        k = min(skip, len(y))
        y, skip = y[k:], skip - k
        if noisy_skip:
            d = min(noisy_skip, len(noisy))
            noisy, noisy_skip = noisy[d:], noisy_skip - d
        if emit_total >= 0:
            remaining = emit_total - emitted
            if remaining <= 0:
                return
            if len(y) > remaining:
                y = y[:remaining]
        else:
            y = y[:left]
        if not len(y):
            return
        if alpha:
            y = alpha * noisy[: len(y)] + (1 - alpha) * y
            noisy = noisy[len(y) :]
        left -= len(y)
        emitted += len(y)
        dst.write(y.astype("<f4").tobytes())

    tail = b""
    while block := src.read(CHUNK * 4):
        # Drain the rest of the input (the upstream ffmpeg was asked for a
        # fixed duration) so it doesn't die with SIGPIPE.
        if emit_total >= 0 and emitted >= emit_total:
            continue
        block = tail + block
        n = len(block) // 4 * 4
        tail = block[n:]
        x = np.frombuffer(block[:n], "<f4")
        left += len(x)
        if alpha:
            noisy = np.concatenate([noisy, x])
        emit(enhancer.process(x))

    # Feed silence to push out the delayed tail.
    if emit_total < 0 or emitted < emit_total:
        emit(enhancer.process(np.zeros(delay + enhancer._win_len, np.float32)))
        emit(enhancer.flush())
        if emit_total >= 0 and emitted < emit_total:
            # Underrun — pad with silence so the caller always gets the exact
            # number of samples it asked for (shouldn't normally happen).
            dst.write(np.zeros(emit_total - emitted, "<f4").tobytes())
    dst.flush()


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(1)
