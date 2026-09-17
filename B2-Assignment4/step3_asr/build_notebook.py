#!/usr/bin/env python3
"""
build_notebook.py - regenerates mini_asr_kaldi.ipynb from the cells below.
Run: python build_notebook.py
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ======================================================================
md(r"""
# Assignment 4 - Mini ASR with Kaldi (Myanmar)

End-to-end HMM-GMM (monophone -> triphone) plus one advanced model, scored
with WER and SER (syllable error rate).

* **Corpus:** 7 speakers (5 train, 2 held-out test: one male, one female).
  Folder names live in `speaker_config.py` (not in git). 5 passes x 150
  prompts each. Audio from `../recordings_trim/` (normalised + trimmed).
* **Split:** `train`/`dev` = the 5 training voices, pooled, ~88/12 random
  (seed `SPLIT_SEED`) - dev is matched to train. `test` = the 2 held-out
  speakers, all passes - speaker-independent.
* **Units:** each Myanmar syllable is one phone (~65 units, one 3-state HMM
  each) - see [`mm_syllable.py`](mm_syllable.py).
* **Features:** MFCC + 3 Kaldi pitch features (Burmese is tonal) + CMVN.
* **LM:** closed-vocabulary bigram from the training transcripts.
* **Kaldi:** runs in Docker (`tklwin/kaldi-apple-silicon`) via the
  `kaldi(...)` helper, project mounted at `/workspace`.

> Prereqs: Docker Desktop running, `tklwin/kaldi-apple-silicon:latest`
> pulled. Kernel = the repo venv (`../../.venv`).
""")

# ----------------------------------------------------------------------
md("## 1 - Configuration")

code(r"""
import os, re, shutil, subprocess, sys, json, unicodedata
from pathlib import Path
import pandas as pd

# --- paths (host) ---
NB_DIR   = Path.cwd()                     # .../B2-Assignment4/step3_asr
PROJ     = NB_DIR.parent                  # mounted as /workspace
S5       = NB_DIR / "s5"                  # Kaldi working dir
REC_SET  = "recordings_trim"              # normalised + silence-trimmed audio
RECS     = PROJ / REC_SET
S5.mkdir(exist_ok=True)

# --- container ---
IMAGE       = "tklwin/kaldi-apple-silicon:latest"
WS          = "/workspace"                # PROJ inside the container
S5_WS       = "/workspace/step3_asr/s5"   # S5 inside the container
KALDI_ROOT  = "/opt/kaldi"

# --- run params ---
# nj must be <= #speakers present in that set (Kaldi splits data by speaker).
NJ_TRAIN  = 4
NJ_DECODE = {"train": 4, "dev": 2, "test": 2}
MFCC_CONF = "--use-energy=false --sample-frequency=16000"
LM_ORDER  = 2

# --- split ---
# dev = matched random hold-out from train speakers. test = the 2 held-out
# speakers (one male, one female), all utterances - speaker-independent.
# Folder names live in speaker_config.py (gitignored, real names) - copy
# speaker_config.example.py to make your own.
sys.path.insert(0, str(NB_DIR))
try:
    from speaker_config import TRAIN_SPEAKER_DIRS, TEST_SPEAKER_DIRS, DROP_UTTS
except ImportError as e:
    raise SystemExit(
        "Missing step3_asr/speaker_config.py - copy speaker_config.example.py "
        "to speaker_config.py and fill in your recording folder names."
    ) from e

ALL_PASSES     = [1, 2, 3, 4, 5]
TRAIN_SPEAKERS = [(d, ALL_PASSES) for d in TRAIN_SPEAKER_DIRS]
TEST_SPEAKERS  = [(TEST_SPEAKER_DIRS["male"], ALL_PASSES), (TEST_SPEAKER_DIRS["female"], ALL_PASSES)]
DEV_FRACTION   = 0.12
SPLIT_SEED     = 1234

# GMM tree/gauss sizes, scaled down from Kaldi wsj defaults to avoid
# overfitting this small corpus. Not yet re-tuned for the current 5 speakers.
GMM_SIZES = {"tri1": (1200, 9000), "tri2": (1800, 12000), "tri3": (1800, 12000)}

BOOST_SIL = 1.25  # helps the small post-trim silence model train

print("PROJ :", PROJ)
print("S5   :", S5)
assert RECS.is_dir(), RECS
""")

# ----------------------------------------------------------------------
md(r"""
## 2 - The `kaldi()` helper

Runs a bash snippet in the container, `PROJ` mounted at `/workspace`, cwd
`s5/`. `path.sh` puts Kaldi binaries and `steps/`/`utils/` on `PATH`.
""")

code(r"""
def _docker_base(interactive=False):
    return [
        "docker", "run", "--rm",
        *(["-it"] if interactive else []),
        "-v", f"{PROJ}:{WS}",
        "-w", S5_WS,
        "-e", "LC_ALL=C",
        IMAGE,
    ]

def kaldi(script: str, check=True, quiet=False):
    # Run `script` in bash inside the container (after sourcing ./path.sh).
    full = f"set -euo pipefail; source ./path.sh 2>/dev/null || true; {script}"
    cmd = _docker_base() + ["bash", "-lc", full]
    if not quiet:
        print("$", script if len(script) < 200 else script[:200] + " ...")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout and not quiet:
        print(p.stdout.rstrip())
    if p.returncode != 0:
        print(p.stderr.rstrip(), file=sys.stderr)
        if check:
            raise RuntimeError(f"kaldi step failed ({p.returncode}): {script[:120]}")
    return p

def kaldi_out(script: str) -> str:
    return kaldi(script, quiet=True).stdout

# smoke test - image present & Kaldi callable
_p = subprocess.run(_docker_base() + ["bash","-lc",
        "echo KALDI_ROOT=$KALDI_ROOT; ls $KALDI_ROOT/egs/wsj/s5 | tr '\\n' ' '; "
        "which compute-mfcc-feats gmm-init-mono arpa2fst nnet3-init 2>&1"],
     capture_output=True, text=True)
print(_p.stdout or _p.stderr)
""")

# ----------------------------------------------------------------------
md("## 3 - Working directory: `path.sh`, `cmd.sh`, `steps/`, `utils/`, `conf/`")

code(r"""
# path.sh - sourced by kaldi() and by Kaldi's own scripts
(S5 / "path.sh").write_text(
    'export KALDI_ROOT=/opt/kaldi\n'
    '[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh\n'
    'export PATH=$PWD/utils:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH\n'
    'if [ -f $KALDI_ROOT/tools/config/common_path.sh ]; then\n'
    '  . $KALDI_ROOT/tools/config/common_path.sh\n'
    'else\n'
    '  for d in $KALDI_ROOT/src/*bin; do export PATH=$d:$PATH; done\n'
    'fi\n'
    'export LC_ALL=C\n', encoding="utf-8")

# cmd.sh - all local, no scheduler
(S5 / "cmd.sh").write_text(
    'export train_cmd="run.pl"\n'
    'export decode_cmd="run.pl"\n'
    'export cuda_cmd="run.pl"\n', encoding="utf-8")

# steps/ and utils/ come from the image's wsj recipe. Copy mm_syllable.py in
# so container-side python can import it.
shutil.copy2(NB_DIR / "mm_syllable.py", S5 / "mm_syllable.py")
(S5 / "syl_tok.py").write_text(
    "import sys\n"
    "from mm_syllable import syllable_break\n"
    "for line in sys.stdin:\n"
    "    p = line.rstrip('\\n').split(None, 1)\n"
    "    print(p[0], *(syllable_break(p[1]) if len(p) > 1 else []))\n",
    encoding="utf-8")
(S5 / "local").mkdir(exist_ok=True)

kaldi(
  'ln -sfn $KALDI_ROOT/egs/wsj/s5/steps steps; '
  'ln -sfn $KALDI_ROOT/egs/wsj/s5/utils utils; '
  'mkdir -p conf exp data; '
  'printf -- "%s\\n" "--use-energy=false" "--sample-frequency=16000" > conf/mfcc.conf; '
  'ls -l steps utils && echo "--- scoring script ---" && ls steps/scoring/score_kaldi_wer.sh')
""")

# ----------------------------------------------------------------------
md(r"""
## 4 - Data preparation

Build Kaldi `data/{train,dev,test}` from the recordings. Transcripts are
normalised (NFC, punctuation stripped, spaces collapsed).

* **train/dev** - training speakers, pooled, split `DEV_FRACTION` at random.
  dev is matched to train (same speakers) - for comparing models.
* **test** - the 2 held-out speakers - speaker-independent, the number that
  matters.
""")

code(r"""
import random
sys.path.insert(0, str(NB_DIR))
from mm_syllable import syllable_break

PUNCT = re.compile(r"[,\.‘’\"'()\[\]/:;!?]+")

def norm_text(s: str) -> str:
    # NFC, drop punctuation, syllable-segment so tokens match the syllable
    # lexicon (word-level text was mostly OOV against a syllable lexicon).
    s = unicodedata.normalize("NFC", s.strip())
    s = PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(syllable_break(s))

def collect(spec):
    # -> list of (utt, spk, container_wav_path, normalised_text, raw_text)
    # spk comes from each pass's own utt2spk, not the folder name - Kaldi
    # needs utt-ids to sort into per-speaker blocks, which only holds if spk
    # matches the id actually embedded in the utt-id text.
    rows = []
    for folder, passes in spec:
        for pz in passes:
            pdir = RECS / folder / f"Rec{pz}"
            t = {}
            for ln in (pdir / "text").read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    k, _, v = ln.partition(" "); t[k] = v
            u2s = {}
            for ln in (pdir / "utt2spk").read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    k, _, v = ln.partition(" "); u2s[k] = v
            for utt in sorted(t):
                if utt in DROP_UTTS:
                    continue
                spk = u2s[utt]
                raw = unicodedata.normalize("NFC", t[utt].strip())
                raw = re.sub(r"\s+", " ", PUNCT.sub(" ", raw)).strip()
                rows.append((utt, spk, f"{WS}/{REC_SET}/{folder}/Rec{pz}/{utt}.wav",
                             norm_text(t[utt]),   # syllable-segmented (for Kaldi + SER)
                             raw))                # original whitespace tokens (for WER)
    return rows

def write_data_dir(name, rows):
    d = S5 / "data" / name
    # Wipe first: a stale feats.scp/cmvn.scp from a previous run makes
    # fix_data_dir.sh silently drop any speaker not in the old cache.
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    cols = {"wav.scp": lambda r: f"{r[0]} {r[2]}",
            "text":    lambda r: f"{r[0]} {r[3]}",
            "utt2spk": lambda r: f"{r[0]} {r[1]}"}
    for fn, f in cols.items():
        (d / fn).write_text("\n".join(sorted(f(r) for r in rows)) + "\n", encoding="utf-8")
    # word-level reference for WER, kept OUTSIDE the data dir (fix_data_dir won't touch it)
    (S5 / "data/local").mkdir(parents=True, exist_ok=True)
    (S5 / "data/local" / f"{name}.wordref").write_text(
        "\n".join(sorted(f"{r[0]} {r[4]}" for r in rows)) + "\n", encoding="utf-8")
    return d

pool = collect(TRAIN_SPEAKERS)
random.Random(SPLIT_SEED).shuffle(pool)
n_dev = int(round(len(pool) * DEV_FRACTION))
dev_rows, train_rows = pool[:n_dev], pool[n_dev:]
test_rows = collect(TEST_SPEAKERS)

for nm, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
    write_data_dir(nm, rows)
    kaldi(f"utils/utt2spk_to_spk2utt.pl data/{nm}/utt2spk > data/{nm}/spk2utt; "
          f"utils/fix_data_dir.sh data/{nm}; "
          f"wc -l < data/{nm}/text | xargs echo 'data/{nm} utterances:'")
print(f"train {len(train_rows)} | dev {len(dev_rows)} (random {DEV_FRACTION:.0%} of "
      f"{len(pool)}, seed {SPLIT_SEED}) | test {len(test_rows)} (unseen speaker)")
""")

code(r"""
# quick look at the units in the training text
train_text = [l.split(" ",1)[1] for l in (S5/"data/train/text").read_text(encoding="utf-8").splitlines()]
syls = sorted({s for line in train_text for s in syllable_break(line)})
print(f"train: {len(train_text)} utts | {len(syls)} unique syllables (each = one phone/unit)")
print(" ".join(syls))
""")

# ----------------------------------------------------------------------
md(r"""
## 5 - Lexicon & `lang/`

Unit = the whole syllable, one 3-state HMM each (~65 units). The *word* in
`words.txt`/`text` stays the Unicode syllable; the *phone* is an ASCII id
(`S001`, ...) since Kaldi's `prepare_lang.sh` mangles multi-codepoint phone
names. Map recorded in `data/local/syl2phone.txt`.

`utils/prepare_lang.sh` builds `data/lang/`.
""")

code(r"""
dd = S5 / "data/local/dict"; dd.mkdir(parents=True, exist_ok=True)

syl2ph = {s: f"S{i:03d}" for i, s in enumerate(syls, 1)}   # syllable -> ASCII phone id
(dd.parent / "syl2phone.txt").write_text(
    "\n".join(f"{s}\t{p}" for s, p in syl2ph.items()) + "\n", encoding="utf-8")

lex = ["!SIL sil", "<UNK> spn"] + [f"{s} {syl2ph[s]}" for s in syls]
phones = list(syl2ph.values())
(dd / "lexicon.txt").write_text("\n".join(lex) + "\n", encoding="utf-8")
(dd / "nonsilence_phones.txt").write_text("\n".join(phones) + "\n", encoding="utf-8")
(dd / "silence_phones.txt").write_text("sil\nspn\n", encoding="utf-8")
(dd / "optional_silence.txt").write_text("sil\n", encoding="utf-8")
(dd / "extra_questions.txt").write_text("", encoding="utf-8")

kaldi('utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang '
      '&& echo "--- phones ---" && head -n 5 data/lang/phones.txt && echo ... '
      '&& echo "words:" $(wc -l < data/lang/words.txt)')
""")

# ----------------------------------------------------------------------
md(r"""
## 6 - Language model  (bigram -> `G.fst`)

Estimate a simple **bigram** (add-one smoothed) over syllable tokens from the
training transcripts, write it as ARPA, and compile to `data/lang_test/G.fst`
with `arpa2fst`.
""")

code(r"""
from collections import Counter
import math

BOS, EOS = "<s>", "</s>"
uni, bi = Counter(), Counter()
for line in train_text:
    toks = [BOS] + syllable_break(line) + [EOS]
    uni.update(toks)
    bi.update(zip(toks[:-1], toks[1:]))

V = sorted(set(uni) - {BOS, EOS}) + [EOS]          # predictable tokens
vocab = set(V) | {BOS}
def logp(w, h):                                     # add-one bigram, log10
    return math.log10((bi[(h, w)] + 1) / (uni[h] + len(vocab)))

arpa = S5 / "data/local/lm.arpa"
lines = ["", "\\data\\", f"ngram 1={len(V)+1}", f"ngram 2={sum(1 for _ in bi)}", "", "\\1-grams:"]
for w in [BOS] + V:
    p = math.log10(max(uni[w], 1) / sum(uni.values()))
    lines.append(f"{p:.5f}\t{w}" + ("" if w == EOS else "\t-1.0"))
lines += ["", "\\2-grams:"]
for (h, w), c in bi.items():
    lines.append(f"{logp(w, h):.5f}\t{h} {w}")
lines += ["", "\\end\\", ""]
arpa.write_text("\n".join(lines), encoding="utf-8")
print(arpa.read_text(encoding='utf-8')[:400], "...")

kaldi(
  'mkdir -p data/lang_test && cp -r data/lang/* data/lang_test/ && '
  'arpa2fst --disambig-symbol="#0" --read-symbol-table=data/lang_test/words.txt '
  '  data/local/lm.arpa data/lang_test/G.fst && '
  'fstisstochastic data/lang_test/G.fst || true && '
  'utils/validate_lang.pl --skip-determinization-check data/lang_test && echo LANG_TEST_OK')
""")

# ----------------------------------------------------------------------
md(r"""
## 7 - Features: MFCC **+ pitch** + CMVN

Burmese is tonal - many syllables differ only in tone, which plain MFCCs
throw away. `steps/make_mfcc_pitch.sh` appends Kaldi's 3 pitch features
(POV, normalised log-pitch, delta-pitch) -> 16-dim before deltas.
""")

code(r"""
# empty conf files -> Kaldi defaults (16 kHz already set via --sample-frequency below)
kaldi('printf -- "%s\\n" "--sample-frequency=16000" "--use-energy=false" > conf/mfcc.conf; '
      'printf -- "%s\\n" "--sample-frequency=16000" > conf/pitch.conf; cat conf/mfcc.conf conf/pitch.conf')

for nm in ("train", "dev", "test"):
    kaldi(f"steps/make_mfcc_pitch.sh --nj {NJ_TRAIN} --mfcc-config conf/mfcc.conf "
          f"  --pitch-config conf/pitch.conf --cmd run.pl data/{nm} exp/make_mfcc/{nm} mfcc")
    kaldi(f"steps/compute_cmvn_stats.sh data/{nm} exp/make_mfcc/{nm} mfcc")
    kaldi(f"utils/fix_data_dir.sh data/{nm}")
kaldi("for n in train dev test; do echo -n \"$n feat-dim: \"; feat-to-dim "
      "scp:data/$n/feats.scp -; done")
""")

# ----------------------------------------------------------------------
md(r"""
## 8 - Decode + score helper (SER + WER)

`text` is syllable-segmented, so Kaldi's scorer gives SER directly. For WER,
the hypothesis is greedily re-segmented into whitespace "words" (closed
vocab, longest-match) and scored against `data/local/<set>.wordref`. Results
accumulate in `RESULTS`.
""")

code(r"""
RESULTS = []

# closed word vocabulary = every whitespace token across the prompt set
WORD_VOCAB = sorted(
    {w for f in ("train", "dev", "test")
       for l in (S5 / "data/local" / f"{f}.wordref").read_text(encoding="utf-8").splitlines()
       for w in l.split()[1:]},
    key=len, reverse=True)                      # longest first for greedy matching

def _resegment(syls: list[str]) -> list[str]:
    s = "".join(syls)
    out, i = [], 0
    while i < len(s):
        for w in WORD_VOCAB:
            if s.startswith(w, i):
                out.append(w); i += len(w); break
        else:
            out.append(s[i]); i += 1           # fallback: single char
    return out

def _best_params(decode_dir: str):
    txt = kaldi_out(f"cat {decode_dir}/scoring_kaldi/best_wer").strip()
    ser = float(re.search(r"%WER (\S+)", txt).group(1))     # 'text' is syllables -> this is SER
    lmwt, pen = re.search(r"wer_(\d+)_([0-9.]+)", txt).groups()
    return ser, lmwt, pen

def _word_wer(dec: str, data: str, lmwt: str, pen: str) -> float:
    hyp_lines = kaldi_out(f"cat {dec}/scoring_kaldi/penalty_{pen}/{lmwt}.txt").splitlines()
    ref = {l.split()[0]: l.split()[1:] for l in
           (S5 / "data/local" / f"{data}.wordref").read_text(encoding="utf-8").splitlines() if l.strip()}
    rp, hp = S5 / f"{dec}/wer_ref.txt", S5 / f"{dec}/wer_hyp.txt"
    (S5 / dec).mkdir(parents=True, exist_ok=True)
    with rp.open("w", encoding="utf-8") as rf, hp.open("w", encoding="utf-8") as hf:
        for l in hyp_lines:
            p = l.split()
            u = p[0]
            if u not in ref:
                continue
            rf.write(f"{u} {' '.join(ref[u])}\n")
            hf.write(f"{u} {' '.join(_resegment(p[1:]))}\n")
    out = kaldi_out(f"compute-wer --text --mode=present "
                    f"ark:{dec}/wer_ref.txt ark:{dec}/wer_hyp.txt 2>/dev/null | head -1")
    m = re.search(r"%WER (\S+)", out)
    return float(m.group(1)) if m else float("nan")

def decode_and_score(exp_dir, graph="graph", fmllr=False, decoder=None):
    name = exp_dir.split("/")[-1]
    for data in ("dev", "test"):
        dec = f"{exp_dir}/decode_{data}"
        nj = NJ_DECODE[data]
        if decoder:
            decoder(data, nj)
        elif fmllr:
            kaldi(f"steps/decode_fmllr.sh --nj {nj} --cmd run.pl --skip-scoring true "
                  f"  {exp_dir}/{graph} data/{data} {dec}")
        else:
            kaldi(f"steps/decode.sh --nj {nj} --cmd run.pl --skip-scoring true "
                  f"  {exp_dir}/{graph} data/{data} {dec}")
        kaldi(f"steps/scoring/score_kaldi_wer.sh --cmd run.pl data/{data} {exp_dir}/{graph} {dec}")
        ser, lmwt, pen = _best_params(dec)
        wer = _word_wer(dec, data, lmwt, pen)
        RESULTS.append({"model": name, "data": data, "WER": wer, "SER": ser,
                        "lmwt": int(lmwt), "pen": pen})
        print(f"  {name}/{data}:  WER={wer:.2f}  SER={ser:.2f}  (lmwt={lmwt}, pen={pen})")
    return pd.DataFrame([r for r in RESULTS if r["model"] == name])
""")

# ----------------------------------------------------------------------
md("## 9 - Monophone")

code(r"""
kaldi(f"steps/train_mono.sh --nj {NJ_TRAIN} --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  data/train data/lang exp/mono")
kaldi("utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph")
decode_and_score("exp/mono")
""")

# ----------------------------------------------------------------------
md("## 10 - Triphone tri1 (Δ + ΔΔ)")

code(r"""
L, G = GMM_SIZES["tri1"]
kaldi(f"steps/align_si.sh --nj {NJ_TRAIN} --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  data/train data/lang exp/mono exp/mono_ali")
kaldi(f"steps/train_deltas.sh --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  {L} {G} data/train data/lang exp/mono_ali exp/tri1")
kaldi("utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph")
decode_and_score("exp/tri1")
""")

# ----------------------------------------------------------------------
md("## 11 - Triphone tri2 (LDA + MLLT)")

code(r"""
L, G = GMM_SIZES["tri2"]
kaldi(f"steps/align_si.sh --nj {NJ_TRAIN} --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  data/train data/lang exp/tri1 exp/tri1_ali")
kaldi(f"steps/train_lda_mllt.sh --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  {L} {G} data/train data/lang exp/tri1_ali exp/tri2")
kaldi("utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph")
decode_and_score("exp/tri2")
""")

# ----------------------------------------------------------------------
md("## 12 - Triphone tri3 (LDA + MLLT + SAT)")

code(r"""
L, G = GMM_SIZES["tri3"]
kaldi(f"steps/align_si.sh --nj {NJ_TRAIN} --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali")
kaldi(f"steps/train_sat.sh --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  {L} {G} data/train data/lang exp/tri2_ali exp/tri3")
kaldi("utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph")
decode_and_score("exp/tri3", fmllr=True)
""")

# ----------------------------------------------------------------------
md(r"""
## 13 - Advanced model

This image has no SGMM2 binaries, so the advanced model is boosted-MMI
discriminative training on top of tri3.

**13b** (optional, off by default): a chain TDNN sketch for more time/RAM,
templated on `run_tdnn_1c.sh` in the image.
""")

code(r"""
# 13a - boosted MMI on tri3 (advanced model)
kaldi(f"steps/align_fmllr.sh --nj {NJ_TRAIN} --cmd run.pl --boost-silence {BOOST_SIL} "
      f"  data/train data/lang exp/tri3 exp/tri3_ali")
kaldi(f"steps/make_denlats.sh --nj {NJ_TRAIN} --cmd run.pl --sub-split {NJ_TRAIN} "
      f"  --transform-dir exp/tri3_ali data/train data/lang exp/tri3 exp/tri3_denlats")
kaldi(f"steps/train_mmi.sh --cmd run.pl --boost 0.1 "
      f"  data/train data/lang exp/tri3_ali exp/tri3_denlats exp/tri3_mmi_b0.1")

MMI_ITER = 4   # 3.mdl / 4.mdl are the usual sweet spot

def _mmi_decode(data, nj):
    kaldi(f"steps/decode.sh --nj {nj} --cmd run.pl --skip-scoring true --iter {MMI_ITER} "
          f"  --transform-dir exp/tri3/decode_{data} "
          f"  exp/tri3/graph data/{data} exp/tri3_mmi_b0.1/decode_{data}")

decode_and_score("exp/tri3_mmi_b0.1", graph="../tri3/graph", decoder=_mmi_decode)
""")

code(r"""
pd.DataFrame([r for r in RESULTS if r["model"] == "tri3_mmi_b0.1"])
""")

code(r"""
# 13b - chain TDNN (optional, heaviest step)
RUN_CHAIN = False
if RUN_CHAIN:
    kaldi(f"steps/align_fmllr_lats.sh --nj {NJ_TRAIN} --cmd run.pl "
          f"  data/train data/lang exp/tri3 exp/tri3_lats && rm -f exp/tri3_lats/fsts.*.gz")
    kaldi("cp -rT data/lang data/lang_chain && "
          "steps/nnet3/chain/gen_topo.py "
          "  $(cat data/lang_chain/phones/silence.csl) "
          "  $(cat data/lang_chain/phones/nonsilence.csl) > data/lang_chain/topo")
    kaldi("steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 "
          "  --cmd run.pl 1200 data/train data/lang_chain exp/tri3_ali exp/chain/tree")
    xcfg = (
      'input dim=13 name=input\\n'
      'relu-batchnorm-layer name=tdnn1 dim=256 input=Append(-1,0,1)\\n'
      'relu-batchnorm-layer name=tdnn2 dim=256 input=Append(-1,0,1,2)\\n'
      'relu-batchnorm-layer name=tdnn3 dim=256 input=Append(-3,0,3)\\n'
      'relu-batchnorm-layer name=tdnn4 dim=256 input=Append(-6,-3,0)\\n'
      'output-layer name=output dim=$num_targets include-log-softmax=false\\n'
      'output-layer name=output-xent dim=$num_targets learning-rate-factor=5.0\\n')
    kaldi('mkdir -p exp/chain/tdnn/configs && '
          f'printf "{xcfg}" > exp/chain/tdnn/configs/network.xconfig && '
          'steps/nnet3/xconfig_to_configs.py '
          '  --xconfig-file exp/chain/tdnn/configs/network.xconfig '
          '  --config-dir exp/chain/tdnn/configs/')
    kaldi("steps/nnet3/chain/train.py --stage -10 --cmd run.pl "
          "  --feat.cmvn-opts '--norm-means=false --norm-vars=false' "
          "  --chain.xent-regularize 0.1 --chain.leaky-hmm-coefficient 0.1 "
          "  --trainer.num-chunk-per-minibatch 64 --trainer.frames-per-iter 200000 "
          "  --trainer.num-epochs 6 --trainer.optimization.num-jobs-initial 1 "
          "  --trainer.optimization.num-jobs-final 1 --egs.chunk-width 140,100,160 "
          "  --cleanup.remove-egs true --feat-dir data/train --tree-dir exp/chain/tree "
          "  --lat-dir exp/tri3_lats --dir exp/chain/tdnn")
    kaldi("utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test exp/chain/tdnn exp/chain/tdnn/graph")
    def _chain_decode(data, nj):
        kaldi(f"steps/nnet3/decode.sh --nj {nj} --cmd run.pl --acwt 1.0 --post-decode-acwt 10.0 "
              f"  --skip-scoring true exp/chain/tdnn/graph data/{data} exp/chain/tdnn/decode_{data}")
    decode_and_score("exp/chain/tdnn", graph="graph", decoder=_chain_decode)
else:
    print("chain TDNN disabled (set RUN_CHAIN=True to enable)")
""")

# ----------------------------------------------------------------------
md("## 14 - Results")

code(r"""
df = pd.DataFrame(RESULTS).drop_duplicates(["model","data"], keep="last")
order = ["mono","tri1","tri2","tri3","tri3_mmi_b0.1","tdnn"]
df["model"] = pd.Categorical(df["model"], [m for m in order if m in set(df.model)]
                             + sorted(set(df.model) - set(order)), ordered=True)
table = df.pivot_table(index="model", columns="data", values=["WER","SER"], observed=True)
table = table.reindex(columns=pd.MultiIndex.from_product([["WER","SER"],["dev","test"]]))
display(table.round(2))
table.to_csv(S5/"RESULTS.csv")

ax = table["WER"].plot(kind="bar", figsize=(8,4), rot=0,
                       title="WER by model (test = 1 male + 1 female speaker)")
ax.set_ylabel("WER %"); ax.figure.tight_layout()
ax.figure.savefig(NB_DIR/"results_wer.png", dpi=120)
""")

# ----------------------------------------------------------------------
md(r"""
## 15 - Notes & limitations

* Only 5 training speakers - models are still data-starved. `dev` (matched)
  shows model progression; `test` (2 unseen speakers) is the honest
  speaker-independent number. Real fix: more training speakers.
* Closed vocabulary + bigram -> WER/SER here are optimistic vs. open-vocab.
* One held-out speaker had truncated clips dropped via `DROP_UTTS` - see
  that speaker's `NOTES.txt` under `../recordings/`.
* No SGMM2 in this image - advanced model is boosted-MMI on tri3. Chain-TDNN
  is off by default (`RUN_CHAIN=False`).
* `nj` is capped by speaker count (Kaldi splits data by speaker).
* Edit `build_notebook.py`, not the notebook, then rerun it to regenerate.
""")

# ======================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (AIEF-B2 venv)", "language": "python", "name": "aief-b2"},
    "language_info": {"name": "python"},
}
out = Path(__file__).with_name("mini_asr_kaldi.ipynb")
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
