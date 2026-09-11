#!/usr/bin/env python3
"""
trim_silence.py - energy-trim leading/trailing silence from a speaker's clips.

The recording guide asked for ~0.5-1 s of silence around each utterance, and
the prompt set is half isolated digits (~0.5 s of speech). The result: in the
grapheme-phone run, 86% of all training frames aligned to silence and the
acoustic models never learned the speech. Trimming to speech-tight clips
(+/- ~0.1 s) fixes the root cause.

Reads a speaker dir shaped as <speaker>/Rec1..RecN/ (e.g. recordings_norm/<name>),
writes a mirror under --out-root with every wav trimmed, sidecars copied, and
wav.scp rewritten to portable ./<utt>.wav.

Usage:
  python tools/trim_silence.py recordings_norm/speaker1 --out-root recordings_trim/speaker1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

PAD_S = 0.10        # keep this much audio either side of detected speech
FRAME_S = 0.025
HOP_S = 0.010
MIN_KEEP_S = 0.30   # never output anything shorter than this


def trim_one(src: Path, dst: Path) -> tuple[float, float]:
    with wave.open(str(src), "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    dur_in = len(x) / fr
    if len(x) == 0:
        shutil.copy2(src, dst)
        return dur_in, dur_in

    fl, hop = int(FRAME_S * fr), int(HOP_S * fr)
    rms = np.array([
        np.sqrt(np.mean(x[i:i + fl] ** 2) + 1e-9)
        for i in range(0, max(1, len(x) - fl), hop)
    ])
    noise = np.percentile(rms, 10)
    peak = np.percentile(rms, 95)
    thr = max(noise * 3.0, peak * 0.06)          # above noise floor, below the loud part
    voiced = np.where(rms > thr)[0]

    if len(voiced) == 0:                          # couldn't find speech - keep as-is
        y = x
    else:
        a = max(0, int(voiced[0] * hop - PAD_S * fr))
        b = min(len(x), int((voiced[-1] * hop + fl) + PAD_S * fr))
        y = x[a:b]
        if len(y) < MIN_KEEP_S * fr:              # too aggressive - fall back
            y = x

    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fr)
        w.writeframes(np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes())
    return dur_in, len(y) / fr


def portable_wav_scp(d: Path) -> None:
    ws = d / "wav.scp"
    if not ws.is_file():
        return
    utts = [ln.split(maxsplit=1)[0] for ln in ws.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ws.write_text("\n".join(f"{u} ./{u}.wav" for u in utts) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speaker_dir", type=Path, help="<speaker>/ dir containing Rec1..RecN")
    ap.add_argument("--out-root", required=True, type=Path, help="mirror dir to create")
    args = ap.parse_args(argv)

    passes = sorted(args.speaker_dir.glob("Rec*"), key=lambda p: int(p.name[3:] or 0))
    if not passes:
        print(f"no RecN dirs under {args.speaker_dir}", file=sys.stderr)
        return 2

    tot_in = tot_out = n = 0
    for src in passes:
        dst = args.out_root / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for wav in sorted(src.glob("*.wav")):
            di, do = trim_one(wav, dst / wav.name)
            tot_in += di
            tot_out += do
            n += 1
        for fn in ("text", "utt2spk", "recordings.tsv", "wav.scp"):
            if (src / fn).is_file():
                shutil.copy2(src / fn, dst / fn)
        portable_wav_scp(dst)
        print(f"  {src.name}: {len(list(src.glob('*.wav')))} clips trimmed")

    print(f"\n{args.speaker_dir.name}: {n} clips, {tot_in/60:.1f} min -> {tot_out/60:.1f} min "
          f"({100*(1-tot_out/tot_in):.0f}% removed) -> {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
