#!/usr/bin/env python3
"""
merge_data_dir.py - merge recorder.py output dirs into one Kaldi data/ dir.

Takes any number of pass/speaker directories (each holding wav.scp / text /
utt2spk, as produced by recorder.py) and writes a single, byte-sorted,
de-duplicated Kaldi data directory:

    <out>/wav.scp   <out>/text   <out>/utt2spk   <out>/spk2utt

This is a portable, pure-Python stand-in for Kaldi's
`utils/fix_data_dir.sh` + `utils/utt2spk_to_spk2utt.pl`, so the corpus can be
prepared on macOS/Windows without a Kaldi install. On the training box you can
still re-run `fix_data_dir.sh` afterwards - it will be a no-op.

wav.scp paths are rewritten to ABSOLUTE paths of the real .wav files by default
(the recorder writes machine-local absolute paths that break after transfer).
Use --wav-prefix to force a different location, or --wav-relative for paths
relative to <out>.

Usage:
  python tools/merge_data_dir.py -o data/train recordings/WinOo_Rec1 recordings/WinOo_Rec2 ...
  python tools/merge_data_dir.py -o data/all   recordings/*_Rec*
  python tools/merge_data_dir.py -o data/test  --wav-prefix /corpus/wav recordings/MgMg_Rec*

Exit code 0 on success, 1 on a fatal inconsistency, 2 on bad usage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

KALDI_FILES = ("wav.scp", "text", "utt2spk")


def read_map(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        out.append((parts[0], parts[1].strip() if len(parts) > 1 else ""))
    return out


def resolve_wav(entry: str, src_dir: Path) -> Path:
    p = Path(entry.strip().strip('"'))
    if not p.is_absolute():
        p = (src_dir / p).resolve()
    return p


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="recorder output dirs to merge")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output Kaldi data dir")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--wav-prefix", type=Path, help="rewrite wav.scp paths to <prefix>/<basename>")
    grp.add_argument("--wav-relative", action="store_true", help="write wav.scp paths relative to --out")
    ap.add_argument("--keep-missing", action="store_true", help="keep entries whose .wav is missing (default: drop + warn)")
    args = ap.parse_args(argv)

    text: dict[str, str] = {}
    wavscp: dict[str, str] = {}
    utt2spk: dict[str, str] = {}
    n_err = 0
    n_drop = 0

    for d in args.dirs:
        if not d.is_dir():
            print(f"ERROR: not a dir: {d}", file=sys.stderr)
            n_err += 1
            continue
        missing = [f for f in KALDI_FILES if not (d / f).is_file()]
        if missing:
            print(f"ERROR: {d} missing {missing}", file=sys.stderr)
            n_err += 1
            continue

        t = dict(read_map(d / "text"))
        w = dict(read_map(d / "wav.scp"))
        u = dict(read_map(d / "utt2spk"))

        for utt in t:
            if utt not in w or utt not in u:
                print(f"ERROR: {d}: {utt} not in all three files", file=sys.stderr)
                n_err += 1
                continue

            wav_path = resolve_wav(w[utt], d)
            if not wav_path.is_file() and not args.keep_missing:
                print(f"WARN: dropping {utt} - wav not found: {wav_path}", file=sys.stderr)
                n_drop += 1
                continue

            if args.wav_prefix:
                new_wav = str((args.wav_prefix / wav_path.name))
            elif args.wav_relative:
                try:
                    new_wav = str(Path(wav_path).resolve().relative_to(args.out.resolve()))
                except ValueError:
                    import os

                    new_wav = os.path.relpath(wav_path, args.out.resolve())
            else:
                new_wav = str(wav_path)

            # conflict detection on repeated utt-ids
            if utt in text:
                if text[utt] != t[utt].strip() or utt2spk[utt] != u[utt].strip():
                    print(f"ERROR: conflicting duplicate utt-id '{utt}' across dirs", file=sys.stderr)
                    n_err += 1
                continue

            text[utt] = t[utt].strip()
            wavscp[utt] = new_wav
            utt2spk[utt] = u[utt].strip()

    if n_err:
        print(f"\nAborting: {n_err} error(s), nothing written.", file=sys.stderr)
        return 1
    if not text:
        print("Aborting: no utterances collected.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    order = sorted(text)  # plain byte sort == LC_ALL=C

    (args.out / "text").write_text("".join(f"{u} {text[u]}\n" for u in order), encoding="utf-8")
    (args.out / "wav.scp").write_text("".join(f"{u} {wavscp[u]}\n" for u in order), encoding="utf-8")
    (args.out / "utt2spk").write_text("".join(f"{u} {utt2spk[u]}\n" for u in order), encoding="utf-8")

    spk2utt: dict[str, list[str]] = {}
    for u in order:
        spk2utt.setdefault(utt2spk[u], []).append(u)
    (args.out / "spk2utt").write_text(
        "".join(f"{spk} {' '.join(spk2utt[spk])}\n" for spk in sorted(spk2utt)), encoding="utf-8"
    )

    print(f"Wrote {args.out}/ : {len(order)} utts, {len(spk2utt)} speaker(s)"
          + (f", {n_drop} dropped (missing wav)" if n_drop else ""))
    for spk in sorted(spk2utt):
        print(f"  {spk:20s} {len(spk2utt[spk])} utts")
    print("\nNext: on the Kaldi box, `utils/fix_data_dir.sh " + str(args.out) + "` (should be a no-op).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
