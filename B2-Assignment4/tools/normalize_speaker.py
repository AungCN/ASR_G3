#!/usr/bin/env python3
"""
normalize_speaker.py - reshape one speaker's raw recorder.py output into the
project convention: <speaker_dir>/Rec1 .. Rec5 (chronological order).

People have sent recordings in every shape recorder.py can produce:
  - already correct:      <speaker>/Rec1 .. Rec5
  - flat, speaker-prefixed: <speaker>_Rec1 .. <speaker>_Rec5
  - auto-named:            rec_1439_28Aug2026, rec_1936_08Sep2026, ...
  - ad-hoc:                 Speaker, Speaker1, Speaker2, ...

This script doesn't care about folder names - it finds every subfolder that
directly contains .wav files, works out each one's actual recording time from
the utt-ids inside it (recorder.py names them <speaker>_YYYYMMDD_HHMMSS), sorts
them chronologically, and renames them to Rec1..RecN in that order. It also
rewrites each pass's wav.scp to portable ./<utt>.wav paths.

Usage:
  python tools/normalize_speaker.py recordings/Speaker5            # do it
  python tools/normalize_speaker.py recordings/Speaker5 --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TS_RE = re.compile(r"_(\d{8})_(\d{6})(?:\.wav)?$")


def pass_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.wav")))


def earliest_timestamp(d: Path) -> str:
    """Smallest YYYYMMDD_HHMMSS found among this folder's utt-ids; '' if none."""
    stamps = []
    for wav in d.glob("*.wav"):
        m = TS_RE.search(wav.name)
        if m:
            stamps.append(m.group(1) + m.group(2))
    if not stamps:
        # fallback: earliest file modification time
        try:
            mt = min(w.stat().st_mtime for w in d.glob("*.wav"))
            return f"mtime:{mt:020.6f}"
        except ValueError:
            return ""
    return min(stamps)


def portable_wav_scp(d: Path) -> int:
    ws = d / "wav.scp"
    if not ws.is_file():
        return 0
    utts = [ln.split(maxsplit=1)[0] for ln in ws.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ws.write_text("\n".join(f"{u} ./{u}.wav" for u in utts) + "\n", encoding="utf-8")
    return len(utts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speaker_dir", type=Path, help="recordings/<speaker>/ (any raw shape inside)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    args = ap.parse_args(argv)

    root = args.speaker_dir
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    dirs = pass_dirs(root)
    if not dirs:
        print(f"ERROR: no subfolder of {root} directly contains .wav files", file=sys.stderr)
        return 2

    ranked = sorted(dirs, key=earliest_timestamp)
    n = len(ranked)
    print(f"Found {n} pass folder(s) under {root}:")
    plan = []
    for i, d in enumerate(ranked, 1):
        target = root / f"Rec{i}"
        nwav = len(list(d.glob("*.wav")))
        same = d.resolve() == target.resolve()
        plan.append((d, target, same))
        arrow = "(already correct)" if same else f"-> {target.name}"
        print(f"  {i}. {d.name:20s} ({nwav:3d} wav)  {arrow}")

    if n != 5:
        print(f"\nWARNING: expected 5 passes, found {n}. Check nothing is missing/extra before proceeding.")

    if args.dry_run:
        print("\n--dry-run: nothing changed. Re-run without --dry-run to apply.")
        return 0

    # rename in two passes via temp names to avoid collisions (e.g. swapping
    # Rec2 <-> Rec3), then make wav.scp portable.
    tmp_names = []
    for d, target, same in plan:
        if same:
            continue
        tmp = root / f".tmp_{d.name}"
        d.rename(tmp)
        tmp_names.append((tmp, target))
    for tmp, target in tmp_names:
        tmp.rename(target)

    total = 0
    for i in range(1, n + 1):
        d = root / f"Rec{i}"
        c = portable_wav_scp(d)
        total += c
        print(f"  Rec{i}: wav.scp made portable ({c} utts)")

    print(f"\nDone: {root} now has Rec1..Rec{n} ({total} utts total).")
    print(f"Next: python tools/validate_recording.py {root}/Rec* --expect-prompts 150")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
