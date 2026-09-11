"""
asr_backend.py - decode ONE arbitrary WAV file with an already-trained model
from mini_asr_kaldi.ipynb, for live demos / ad-hoc testing (not the fixed
train/dev/test pipeline).

All five trained models are supported. mono/tri1/tri2 decode directly.
tri3 (SAT) and tri3_mmi_b0.1 need an fMLLR transform, which is normally
estimated per *speaker* from many utterances - here it's estimated from just
this one clip, so treat their output as more of a curiosity than tri3/MMI's
real capability (mono remains the best-performing, most reliable model from
this project's evaluation runs - see step3_asr/README.md).

Usage (CLI):
    python asr_backend.py path/to/clip.wav [--model mono|tri1|tri2|tri3|tri3_mmi_b0.1]

As a library:
    from asr_backend import transcribe
    text, meta = transcribe("clip.wav", model="mono")
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
PROJ = NB_DIR.parent
S5 = NB_DIR / "s5"
IMAGE = "tklwin/kaldi-apple-silicon:latest"
WS = "/workspace"
LIVE_WAV_DIR = S5 / "data" / "live_wavs"          # kept inside PROJ -> visible to the container
SUPPORTED_MODELS = ("mono", "tri1", "tri2", "tri3", "tri3_mmi_b0.1")
NEEDS_FMLLR = {"tri3", "tri3_mmi_b0.1"}           # estimate a (very noisy, 1-utt) fMLLR transform first
GRAPH_DIR = {                                      # where each model's HCLG.fst lives
    "mono": "exp/mono/graph", "tri1": "exp/tri1/graph", "tri2": "exp/tri2/graph",
    "tri3": "exp/tri3/graph", "tri3_mmi_b0.1": "exp/tri3/graph",  # MMI reuses tri3's graph/tree
}
LMWT, WIP = 17, 1.0   # matches what score_kaldi_wer.sh picked throughout the notebook runs


class ASRError(RuntimeError):
    pass


def _docker(script: str) -> subprocess.CompletedProcess:
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJ}:{WS}",
        "-w", f"{WS}/step3_asr/s5",
        "-e", "LC_ALL=C",
        IMAGE, "bash", "-lc",
        f"set -euo pipefail; source ./path.sh; {script}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _run(script: str) -> str:
    p = _docker(script)
    if p.returncode != 0:
        raise ASRError(p.stderr.strip() or p.stdout.strip() or f"step failed: {script[:80]}")
    return p.stdout


def check_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as w:
        ch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
    if (ch, sw, fr) != (1, 2, 16000):
        raise ASRError(
            f"{path.name}: got {ch}ch/{sw*8}-bit/{fr}Hz, need mono/16-bit/16000Hz "
            f"(this is what recorder.py always produces)."
        )


def _model_ready(model: str) -> bool:
    graph_ok = (S5 / GRAPH_DIR[model] / "HCLG.fst").is_file()
    mdl_ok = (S5 / "exp" / model / "final.mdl").is_file()
    return graph_ok and mdl_ok


def model_graph(model: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise ASRError(f"model must be one of {SUPPORTED_MODELS}")
    if not _model_ready(model):
        raise ASRError(f"exp/{model} isn't trained yet - run mini_asr_kaldi.ipynb through that "
                        f"model's section first.")
    return model


def available_models() -> list[str]:
    return [m for m in SUPPORTED_MODELS if _model_ready(m)]


def transcribe(wav_path: str | Path, model: str = "mono") -> tuple[str, dict]:
    """Returns (syllable-tokenised transcript, timing/debug info)."""
    t0 = time.time()
    wav_path = Path(wav_path).resolve()
    if not wav_path.is_file():
        raise ASRError(f"no such file: {wav_path}")
    check_wav(wav_path)
    model = model_graph(model)

    LIVE_WAV_DIR.mkdir(parents=True, exist_ok=True)
    utt_id = f"live_{int(time.time() * 1000)}"
    dst = LIVE_WAV_DIR / f"{utt_id}.wav"
    shutil.copy2(wav_path, dst)
    cwav = f"{WS}/step3_asr/s5/data/live_wavs/{utt_id}.wav"

    d = S5 / "data" / "live"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "wav.scp").write_text(f"{utt_id} {cwav}\n", encoding="utf-8")
    (d / "utt2spk").write_text(f"{utt_id} live_spk\n", encoding="utf-8")
    (d / "spk2utt").write_text(f"live_spk {utt_id}\n", encoding="utf-8")

    # must match the notebook's features (MFCC + 3 pitch = 16-dim), or gmm-latgen aborts
    _run(
        f"rm -rf data/live/feats.scp data/live/cmvn.scp mfcc/live* && "
        f"steps/make_mfcc_pitch.sh --nj 1 --mfcc-config conf/mfcc.conf "
        f"  --pitch-config conf/pitch.conf --cmd run.pl data/live exp/make_mfcc/live mfcc >/dev/null && "
        f"steps/compute_cmvn_stats.sh data/live exp/make_mfcc/live mfcc >/dev/null"
    )

    graph = GRAPH_DIR[model]
    if model in NEEDS_FMLLR:
        # fMLLR is normally estimated per speaker from many utterances; here it's
        # estimated from this one clip alone, so treat it as noisy at best.
        fmllr_dec = "exp/tri3/decode_live"
        _run(f"rm -rf {fmllr_dec} && steps/decode_fmllr.sh --nj 1 --cmd run.pl "
             f"  --skip-scoring true {graph} data/live {fmllr_dec} >/dev/null")
        if model == "tri3":
            dec = fmllr_dec
        else:  # tri3_mmi_b0.1: redecode with the tri3-estimated transform
            dec = f"exp/{model}/decode_live"
            _run(f"rm -rf {dec} && steps/decode.sh --nj 1 --cmd run.pl --iter 4 "
                 f"  --transform-dir {fmllr_dec} --skip-scoring true "
                 f"  {graph} data/live {dec} >/dev/null")
    else:
        dec = f"exp/{model}/decode_live"
        _run(f"rm -rf {dec} && steps/decode.sh --nj 1 --cmd run.pl --skip-scoring true "
             f"  {graph} data/live {dec} >/dev/null")
    out = _run(
        f"gunzip -c {dec}/lat.1.gz | "
        f"lattice-scale --inv-acoustic-scale={LMWT} ark:- ark:- 2>/dev/null | "
        f"lattice-add-penalty --word-ins-penalty={WIP} ark:- ark:- 2>/dev/null | "
        f"lattice-best-path ark:- ark,t:- 2>/dev/null | "
        f"utils/int2sym.pl -f 2- data/lang_test/words.txt"
    )
    line = out.strip()
    words = line.split()[1:] if line else []
    meta = {"model": model, "utt_id": utt_id, "seconds": round(time.time() - t0, 2)}
    return " ".join(words), meta


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--model", default="mono", choices=SUPPORTED_MODELS)
    args = ap.parse_args(argv)
    try:
        text, meta = transcribe(args.wav, args.model)
    except ASRError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[{meta['model']}, {meta['seconds']}s]  {text or '(nothing recognised)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
