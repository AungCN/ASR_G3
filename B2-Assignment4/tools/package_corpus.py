#!/usr/bin/env python3
"""
package_corpus.py - assemble the group's recordings into a submission package.

For each speaker directory (recordings/<speaker>/ with Rec1..RecN inside) this
builds, under --out-root (default: raw-corpus/):

    <speaker>/Rec1..RecN/     copied pass dirs, wav.scp made portable (./<utt>.wav)
    <speaker>/kaldi_data/     per-speaker merged Kaldi dir (sorted, deduped, +spk2utt)
    <speaker>/MANIFEST.txt    speaker metadata; gender/group left as <FILL IN>

    kaldi_data_all/           all speakers merged into one Kaldi dir
    GROUP_MANIFEST.txt        one-page summary of every speaker

Run tools/validate_recording.py first - this script refuses a speaker whose
pass dirs have validation ERRORS unless --force is given.

Usage:
  python tools/package_corpus.py recordings/speaker1 recordings/speaker2 recordings/speaker3
  python tools/package_corpus.py --zip aief_b2_assignment4.zip recordings/*/
  python tools/package_corpus.py --out-root raw-corpus --group "Group-2" recordings/*/
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import wave
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PY = sys.executable
SIDECARS = ("wav.scp", "text", "utt2spk", "recordings.tsv")


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *[str(a) for a in args]], capture_output=True, text=True)


def portable_wav_scp(d: Path) -> None:
    ws = d / "wav.scp"
    if not ws.is_file():
        return
    utts = [ln.split(maxsplit=1)[0] for ln in ws.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ws.write_text("\n".join(f"{u} ./{u}.wav" for u in utts) + "\n", encoding="utf-8")


def pass_stats(d: Path) -> tuple[int, float, float, float]:
    durs = []
    for w in sorted(d.glob("*.wav")):
        with wave.open(str(w), "rb") as f:
            durs.append(f.getnframes() / f.getframerate())
    if not durs:
        return 0, 0.0, 0.0, 0.0
    return len(durs), min(durs), sum(durs) / len(durs), max(durs)


def speaker_manifest(spk: str, src: Path, passes: list[Path], group: str, gender: str) -> str:
    L = [
        f"SPEAKER SUBMISSION MANIFEST - Assignment 4 (Mini ASR)",
        "=" * 55,
        "",
        f"speaker_id      : {spk}",
        f"group           : {group or '<FILL IN>'}",
        f"gender          : {gender or '<FILL IN: male / female>'}   # drives the by-speaker test split",
        f"source_dir      : {src}",
        f"audio_format    : 16 kHz, mono, 16-bit PCM WAV",
        f"prompt_script   : mini-asr-v1.txt (150 prompts)",
        f"passes          : {len(passes)}",
        "",
        "PER-PASS SUMMARY",
        "-" * 55,
    ]
    total = 0
    for i, p in enumerate(passes, 1):
        n, mn, mean, mx = pass_stats(p)
        total += n
        note = "" if n == 150 else f"  <-- {n} clips (expected 150)"
        L.append(f"  Rec{i}: {n:3d} clips | clip {mn:.2f}-{mx:.2f}s (mean {mean:.2f}){note}")
    L += ["", f"total utterances : {total}", ""]

    notes = src / "NOTES.txt"
    if notes.is_file() and notes.read_text(encoding="utf-8").strip():
        L += ["NOTES (from recordings/{}/NOTES.txt)".format(spk), "-" * 55]
        L += ["  " + ln if ln.strip() else "" for ln in notes.read_text(encoding="utf-8").splitlines()]
        L += [""]

    L += [
        "CONTENTS",
        "-" * 55,
        "  Rec1..RecN/   raw pass dirs (150 .wav + wav.scp + text + utt2spk + recordings.tsv)",
        "  kaldi_data/   merged Kaldi dir: byte-sorted, de-duplicated, + spk2utt",
        "                wav.scp paths relative to kaldi_data/",
        "  MANIFEST.txt  this file",
        "",
        f"generated: {datetime.date.today().isoformat()}",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speakers", nargs="+", type=Path, help="recordings/<speaker>/ dirs (each with Rec1..RecN)")
    ap.add_argument("--out-root", type=Path, default=Path("raw-corpus"), help="package root (default raw-corpus)")
    ap.add_argument("--group", default="", help="group label written into every manifest")
    ap.add_argument("--all-gender", default="", choices=["", "male", "female"],
                    help="set gender for every speaker in this run")
    ap.add_argument("--gender", action="append", default=[], metavar="SPK=male|female",
                    help="set gender for one speaker (repeatable); overrides --all-gender")
    ap.add_argument("--zip", type=Path, help="also write a single zip of the whole package")
    ap.add_argument("--force", action="store_true", help="package even if validate_recording.py reports errors")
    args = ap.parse_args(argv)

    gender_map = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.gender if "=" in kv}

    speakers = [s for s in args.speakers if s.is_dir()]
    if not speakers:
        print("no speaker dirs given", file=sys.stderr)
        return 2

    all_pass_dirs: list[Path] = []
    internal_id: dict[str, str] = {}   # folder name -> actual speaker id used in utt2spk
    for src in speakers:
        spk = src.name
        passes = sorted(src.glob("Rec*"), key=lambda p: int(p.name[3:] or 0))
        if not passes:
            print(f"skip {src}: no RecN subdirs", file=sys.stderr)
            continue

        v = sh(TOOLS / "validate_recording.py", *passes)
        if v.returncode != 0 and not args.force:
            print(f"ERROR: {spk} has validation errors - fix them or use --force\n{v.stdout}", file=sys.stderr)
            return 1

        dst_spk = args.out_root / spk
        if dst_spk.exists():
            shutil.rmtree(dst_spk)
        dst_spk.mkdir(parents=True)

        for i, p in enumerate(passes, 1):
            dpass = dst_spk / f"Rec{i}"
            shutil.copytree(p, dpass, ignore=shutil.ignore_patterns(".DS_Store"))
            portable_wav_scp(dpass)
            all_pass_dirs.append(dpass)
            if spk not in internal_id:
                u2s = (dpass / "utt2spk").read_text(encoding="utf-8").split(None, 2)
                if len(u2s) >= 2:
                    internal_id[spk] = u2s[1]

        m = sh(TOOLS / "merge_data_dir.py", "-o", dst_spk / "kaldi_data", "--wav-relative",
               *[dst_spk / f"Rec{i}" for i in range(1, len(passes) + 1)])
        if m.returncode != 0:
            print(f"ERROR merging {spk}:\n{m.stderr}", file=sys.stderr)
            return 1
        gender = gender_map.get(spk, args.all_gender)
        (dst_spk / "MANIFEST.txt").write_text(
            speaker_manifest(spk, src, passes, args.group, gender), encoding="utf-8")
        note = ""
        if internal_id.get(spk) and internal_id[spk] != spk:
            note = f"  (internal speaker id in utt2spk is '{internal_id[spk]}', differs in case/spelling)"
        print(f"packaged {spk}: {len(passes)} passes -> {dst_spk}{note}")

    # combined merge across every packaged pass dir
    ka = args.out_root / "kaldi_data_all"
    m = sh(TOOLS / "merge_data_dir.py", "-o", ka, "--wav-relative", *all_pass_dirs)
    print(m.stdout.strip())
    if m.returncode != 0:
        print(m.stderr, file=sys.stderr)
        return 1

    spk2utt = (ka / "spk2utt").read_text(encoding="utf-8").splitlines()
    gm = [
        "GROUP SUBMISSION MANIFEST - Assignment 4 (Mini ASR)",
        "=" * 55,
        f"group     : {args.group or '<FILL IN>'}",
        f"speakers  : {len(spk2utt)}",
        f"generated : {datetime.date.today().isoformat()}",
        "",
        "SPEAKERS (fill gender in each <speaker>/MANIFEST.txt)",
        "-" * 55,
    ]
    by_internal_id = {v: k for k, v in internal_id.items()}  # utt2spk id -> folder name
    for line in spk2utt:
        sid, *utts = line.split()
        folder = by_internal_id.get(sid, sid)
        g = gender_map.get(sid) or gender_map.get(folder) or args.all_gender or "<FILL IN>"
        gm.append(f"  {sid:20s} {len(utts):4d} utts   gender: {g}")
    gm += [
        "",
        "LAYOUT",
        "-" * 55,
        "  <speaker>/Rec1..5/   raw pass dirs",
        "  <speaker>/kaldi_data/  per-speaker merged Kaldi dir",
        "  kaldi_data_all/        all speakers merged (train/dev/test split by speaker)",
        "",
        "Test set = all utterances of one male + one female speaker.",
        "",
    ]
    (args.out_root / "GROUP_MANIFEST.txt").write_text("\n".join(gm), encoding="utf-8")
    print(f"wrote {args.out_root}/GROUP_MANIFEST.txt")

    if args.zip:
        base = args.zip.with_suffix("")
        shutil.make_archive(str(base), "zip", root_dir=args.out_root.parent, base_dir=args.out_root.name)
        z = base.with_suffix(".zip")
        print(f"wrote {z}  ({z.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
