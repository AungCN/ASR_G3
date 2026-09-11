#!/usr/bin/env python3
"""
validate_recording.py - sanity-check one recorder.py output directory.

Checks a single pass directory (e.g. recordings/WinOo_Rec1/):
  * wav.scp / text / utt2spk exist and are mutually consistent
  * utt-id sets match the actual .wav files (no orphans either way)
  * every wav is mono / 16-bit / 16 kHz  (override with --sr / --bits / --channels)
  * transcripts are non-empty
  * Kaldi files are byte-sorted by utt-id
  * duration stats; flags very short (< --min-dur) and heavily clipped clips
  * recordings.tsv (if present) matches the utt-id set

Exit code 0 = clean (warnings allowed), 1 = at least one ERROR, 2 = bad usage.

Usage:
  python tools/validate_recording.py recordings/WinOo_Rec1
  python tools/validate_recording.py recordings/*_Rec*        # many dirs
  python tools/validate_recording.py --expect-prompts 150 recordings/WinOo_Rec1
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

KALDI_FILES = ("wav.scp", "text", "utt2spk")


class Report:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.speakers: set[str] = set()

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def dump(self) -> None:
        print(f"\n=== {self.name} ===")
        for w in self.warnings:
            print(f"  WARN  {w}")
        for e in self.errors:
            print(f"  ERROR {e}")
        if not self.errors and not self.warnings:
            print("  OK")
        elif not self.errors:
            print(f"  -> {len(self.warnings)} warning(s), no errors")
        else:
            print(f"  -> {len(self.errors)} error(s), {len(self.warnings)} warning(s)")


def read_kaldi_map(path: Path) -> list[tuple[str, str]]:
    """Return list of (utt_id, rest) preserving file order; skips blank lines."""
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def wav_info(path: Path) -> tuple[int, int, int, float, float]:
    """(channels, sampwidth_bytes, framerate, duration_s, peak_ratio)."""
    with wave.open(str(path), "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        dur = n / fr if fr else 0.0
        peak_ratio = 0.0
        if sw == 2 and n:
            import array

            frames = w.readframes(min(n, fr * 30))  # cap at 30 s for the peak scan
            a = array.array("h")
            a.frombytes(frames)
            peak = max((abs(x) for x in a), default=0)
            peak_ratio = peak / 32768.0
    return ch, sw, fr, dur, peak_ratio


def validate_dir(d: Path, args: argparse.Namespace) -> Report:
    rep = Report(str(d))
    if not d.is_dir():
        rep.err("not a directory")
        return rep

    wavs = sorted(p for p in d.glob("*.wav"))
    wav_ids = {p.stem for p in wavs}
    if not wavs:
        rep.err("no .wav files found")

    # --- Kaldi files present + parse ---
    maps: dict[str, list[tuple[str, str]]] = {}
    for fn in KALDI_FILES:
        fp = d / fn
        if not fp.is_file():
            rep.err(f"missing {fn}")
            continue
        maps[fn] = read_kaldi_map(fp)

    # --- per-file internal checks ---
    id_sets: dict[str, set[str]] = {}
    for fn, rows in maps.items():
        ids = [r[0] for r in rows]
        id_sets[fn] = set(ids)
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            rep.err(f"{fn}: duplicate utt-ids: {', '.join(dupes[:10])}")
        if ids != sorted(ids):
            rep.warn(f"{fn}: not byte-sorted by utt-id (run merge_data_dir.py to fix)")

    # --- cross-file consistency ---
    if len(id_sets) == len(KALDI_FILES):
        base = id_sets["text"]
        for fn in ("wav.scp", "utt2spk"):
            if id_sets[fn] != base:
                only_a = sorted(base - id_sets[fn])[:5]
                only_b = sorted(id_sets[fn] - base)[:5]
                rep.err(f"{fn} utt-ids differ from text (text-only={only_a} {fn}-only={only_b})")
        if wavs and base != wav_ids:
            missing_wav = sorted(base - wav_ids)[:5]
            missing_ent = sorted(wav_ids - base)[:5]
            if missing_wav:
                rep.err(f"utt-ids with no .wav file: {missing_wav}")
            if missing_ent:
                rep.err(f".wav files with no Kaldi entry: {missing_ent}")

    # --- text content ---
    for utt, txt in maps.get("text", []):
        if not txt.strip():
            rep.err(f"empty transcript for {utt}")

    # --- utt2spk prefix rule ---
    for utt, spk in maps.get("utt2spk", []):
        if not utt.startswith(spk):
            rep.warn(f"utt-id '{utt}' does not start with speaker '{spk}' (Kaldi sorting risk)")
    speakers = {spk for _, spk in maps.get("utt2spk", [])}
    rep.speakers = speakers
    if len(speakers) > 1:
        rep.warn(f"multiple speakers in one pass dir: {sorted(speakers)}")

    # --- wav.scp paths resolve ---
    for utt, rest in maps.get("wav.scp", []):
        cand = rest.strip().strip('"')
        p = Path(cand)
        if not p.is_absolute():
            p = (d / p).resolve()
        if not p.is_file():
            rep.err(f"wav.scp path for {utt} does not exist: {cand}")

    # --- audio format + duration stats ---
    durs: list[float] = []
    for wpath in wavs:
        try:
            ch, sw, fr, dur, peak = wav_info(wpath)
        except Exception as e:  # noqa: BLE001
            rep.err(f"cannot read {wpath.name}: {e}")
            continue
        durs.append(dur)
        if ch != args.channels:
            rep.err(f"{wpath.name}: {ch} channel(s), expected {args.channels}")
        if sw * 8 != args.bits:
            rep.err(f"{wpath.name}: {sw * 8}-bit, expected {args.bits}-bit")
        if fr != args.sr:
            rep.err(f"{wpath.name}: {fr} Hz, expected {args.sr} Hz")
        if dur < args.min_dur:
            rep.warn(f"{wpath.name}: very short ({dur:.2f}s) - check it is not silence/cut")
        if peak >= 0.999:
            rep.warn(f"{wpath.name}: near full-scale peak ({peak:.3f}) - possible clipping")

    if durs:
        durs.sort()
        total = sum(durs)
        print(
            f"  {len(durs)} clips | dur min/mean/max = "
            f"{durs[0]:.2f}/{total / len(durs):.2f}/{durs[-1]:.2f}s | total {total / 60:.1f} min"
        )
        if args.expect_prompts and len(durs) != args.expect_prompts:
            rep.warn(
                f"{len(durs)} clips but --expect-prompts={args.expect_prompts} "
                f"(missing {args.expect_prompts - len(durs)})"
            )

    # --- recordings.tsv cross-check (metadata, warn only) ---
    tsv = d / "recordings.tsv"
    if tsv.is_file():
        rows = [ln.split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rows and rows[0][:2] == ["speaker", "utt_id"]:
            tsv_ids = {r[1] for r in rows[1:] if len(r) > 1}
            if "text" in id_sets and tsv_ids != id_sets["text"]:
                rep.warn(
                    "recordings.tsv utt-id set differs from Kaldi text "
                    f"(tsv={len(tsv_ids)}, text={len(id_sets['text'])})"
                )
        else:
            rep.warn("recordings.tsv header not recognised")
    else:
        rep.warn("no recordings.tsv (metadata file) in this dir")

    return rep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="recorder output directories to check")
    ap.add_argument("--sr", type=int, default=16000, help="expected sample rate (default 16000)")
    ap.add_argument("--bits", type=int, default=16, help="expected bit depth (default 16)")
    ap.add_argument("--channels", type=int, default=1, help="expected channel count (default 1)")
    ap.add_argument("--min-dur", type=float, default=0.4, help="warn below this duration in s (default 0.4)")
    ap.add_argument("--expect-prompts", type=int, default=0, help="warn if clip count != this (0 = off)")
    args = ap.parse_args(argv)

    reports = [validate_dir(d, args) for d in args.dirs]
    for r in reports:
        r.dump()

    # cross-dir: passes sharing a parent folder (recordings/<speaker>/RecN) should
    # all carry the SAME utt2spk speaker id - catches e.g. someone typing a
    # different "name" into the recorder for each pass.
    by_parent: dict[Path, list[Report]] = {}
    for d, r in zip(args.dirs, reports):
        by_parent.setdefault(Path(d).resolve().parent, []).append(r)
    for parent, group in by_parent.items():
        if len(group) < 2:
            continue
        all_spk = set().union(*(r.speakers for r in group))
        if len(all_spk) > 1:
            print(f"\n=== {parent} (cross-pass) ===")
            print(f"  ERROR passes under this folder don't share one speaker id: {sorted(all_spk)}")
            group[0].errors.append("cross-pass speaker id mismatch")  # counted in totals below

    n_err = sum(len(r.errors) for r in reports)
    n_warn = sum(len(r.warnings) for r in reports)
    print(f"\nSUMMARY: {len(reports)} dir(s), {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
