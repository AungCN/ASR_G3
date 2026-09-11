#!/usr/bin/env python3
"""
normalize_recordings.py - loudness-normalised COPY of recorder.py output.

The raw recordings are left untouched. For each input pass directory this writes
a mirror directory under --out-root with:
  * every .wav peak-normalised to --peak dBFS (default -6), gain capped at
    --max-gain dB (default 26) so near-silent clips are not blown up
  * text / utt2spk / recordings.tsv copied verbatim
  * wav.scp rewritten with portable "./<utt>.wav" paths

Intended for perceptual spot-checks and neural models. For Kaldi HMM-GMM the
raw audio is fine (CMVN cancels the level offset) - train from the originals.

Usage:
  python tools/normalize_recordings.py --out-root recordings_norm recordings/speaker1_Rec*
  python tools/normalize_recordings.py --peak -3 --max-gain 30 --out-root norm recordings/*_Rec1
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

COPY_VERBATIM = ("text", "utt2spk", "recordings.tsv")


def load_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path.name}: expected 16-bit, got {sw * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, fr, ch, sw


def write_wav(path: Path, data_i16: np.ndarray, fr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fr)
        w.writeframes(data_i16.tobytes())


def process_dir(src: Path, out_root: Path, peak_dbfs: float, max_gain_db: float) -> list[float]:
    dst = out_root / src.name
    dst.mkdir(parents=True, exist_ok=True)
    target_amp = 32767.0 * (10.0 ** (peak_dbfs / 20.0))
    max_gain = 10.0 ** (max_gain_db / 20.0)

    gains_db: list[float] = []
    for wav in sorted(src.glob("*.wav")):
        data, fr, _, _ = load_wav(wav)
        peak = float(np.max(np.abs(data))) or 1.0
        gain = min(target_amp / peak, max_gain)
        gains_db.append(20.0 * math.log10(gain) if gain > 0 else 0.0)
        out = np.clip(np.round(data * gain), -32768, 32767).astype("<i2")
        write_wav(dst / wav.name, out, fr)

    for fn in COPY_VERBATIM:
        if (src / fn).is_file():
            shutil.copy2(src / fn, dst / fn)

    if (src / "wav.scp").is_file():
        lines = []
        for line in (src / "wav.scp").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt = line.split(maxsplit=1)[0]
            lines.append(f"{utt} ./{utt}.wav")
        (dst / "wav.scp").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return gains_db


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="recorder pass directories to normalise")
    ap.add_argument("--out-root", required=True, type=Path, help="root dir for the normalised mirror")
    ap.add_argument("--peak", type=float, default=-6.0, help="target peak level in dBFS (default -6)")
    ap.add_argument("--max-gain", type=float, default=26.0, help="max gain to apply in dB (default 26)")
    args = ap.parse_args(argv)

    n_dirs = 0
    for d in args.dirs:
        if not d.is_dir():
            print(f"skip (not a dir): {d}", file=sys.stderr)
            continue
        gains = process_dir(d, args.out_root, args.peak, args.max_gain)
        n_dirs += 1
        if gains:
            g = sorted(gains)
            capped = sum(1 for x in gains if x >= args.max_gain - 1e-6)
            print(
                f"{d.name}: {len(gains)} clips -> {args.out_root / d.name}  "
                f"gain dB min/med/max = {g[0]:.1f}/{g[len(g) // 2]:.1f}/{g[-1]:.1f}"
                + (f"  ({capped} hit the {args.max_gain:g} dB cap)" if capped else "")
            )

    print(f"\nDone: {n_dirs} dir(s) normalised to {args.peak:g} dBFS peak under {args.out_root}/")
    print("Raw recordings unchanged. Validate the copy with tools/validate_recording.py.")
    return 0 if n_dirs else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
