# step3_asr/ — Kaldi ASR pipeline (Assignment 4, Step 3)

Everything for building the HMM-GMM + advanced models and scoring WER / SER,
driven from one Jupyter notebook. Kaldi itself runs in Docker

## Files

| File | What |
|---|---|
| `mini_asr_kaldi.ipynb` | the pipeline notebook — run this |
| `build_notebook.py` | regenerates the `.ipynb` from source cells (edit here, not the notebook) |
| `mm_syllable.py` | Myanmar syllable segmentation |
| `tools/trim_silence.py`, `tools/normalize_recordings.py` | build `recordings_trim/` (in `../tools/`) |
| `asr_backend.py` | decode ONE arbitrary WAV with an already-trained model (used by `asr_tester.py`, also usable standalone/CLI) |
| `asr_tester.py` | PyQt6 GUI: record/load audio, transcribe live, score against a typed reference |
| `s5/` | Kaldi working dir, created by the notebook (data/, exp/, mfcc/, …) — large, disposable |
| `results_wer.png`, `s5/RESULTS.csv` | written by the results cell |

## One-time setup

```bash
# 1. Docker Desktop running, then pull the Kaldi image (~large, one time)
docker pull tklwin/kaldi-apple-silicon:latest

# 2. Jupyter + deps are already in the repo venv:
#    /Users/acnmacm4/ACNM4_Workstation/AIEF_ACN/AIEF_B2_References_Assignment/.venv
#    (jupyterlab, nbformat, pandas, matplotlib)

# 3. launch
cd B2-Assignment4/step3_asr
../../.venv/bin/jupyter lab mini_asr_kaldi.ipynb
```

Register the venv as a kernel once if it isn't listed:

```bash
../../.venv/bin/python -m ipykernel install --user --name aief-b2 --display-name "Python 3 (AIEF-B2 venv)"
```

## What the notebook does

1. **Config** — paths, the split, Docker params.
2. **`kaldi()` helper** — runs a bash snippet inside the image with the project
   mounted at `/workspace`, working dir `step3_asr/s5`.
3. **Work dir** — `path.sh`, `cmd.sh` (`run.pl`, all local), `steps/`+`utils/`
   symlinked from the image's `wsj/s5`, `conf/mfcc.conf`.
4. **Data prep** — `data/{train,dev,test}` with container `wav.scp` paths,
   NFC + punctuation-stripped transcripts, `fix_data_dir.sh`.
5. **Lexicon / `lang/`** — each syllable is one unit; ASCII phone ids
   (`S001`..), `prepare_lang.sh`. Transcripts are syllable-segmented.
6. **LM** — add-one **bigram** over training syllables → `arpa2fst` → `G.fst`.
7. **Features** — MFCC **+ 3 pitch features** (`make_mfcc_pitch.sh`) + CMVN.
8. **Decode/score helper** — SER via `score_kaldi_wer.sh` (text is syllables);
   WER by greedy re-segmentation of the hyp into prompt words.
9-12. **HMM-GMM** — monophone → tri1 (Δ+ΔΔ) → tri2 (LDA+MLLT) → tri3 (SAT).
13. **Advanced** — boosted-MMI on tri3 (this image has no SGMM2 binaries; a
    chain-TDNN cell is included but disabled by default, `RUN_CHAIN=False`).
14. **Results** — WER/SER table (`RESULTS.csv`) + bar chart.
15. **Notes / limitations.**

## Split (matches the assignment spec)

**6 speakers, 4 for training and 2 held out for testing** (one male voice,
one female voice) - the actual recording-folder names live in
`speaker_config.py`, which is not committed to git (see the main
[README](../README.md) for why). Everywhere below, speakers are referred to
by role, not name.

| set | speakers | passes | utts |
|---|---|---|---|
| train | the 4 training voices | pooled, random 88% (seed 1234) | 2645 |
| dev | same 4 | pooled, random 12% (seed 1234) - **matched** to train | 361 |
| test | held-out male voice + held-out female voice | all 5 | 1497 (a couple of known-bad clips dropped) - **speaker-independent** |

`NJ_TRAIN=4`, `GMM_SIZES` roughly doubled (tri1 1200/9000, tri2/tri3
1800/12000) to match the bigger training set. More speakers land -> add them
to `speaker_config.py`, `python build_notebook.py`, rerun from section 4.

## Results (2026-09-11, after the OOV-bug fix / `RESULTS.csv`)

**SER (syllable error rate) is the primary metric** - `text` is
syllable-segmented so Kaldi's scorer reports SER directly. WER is approximate:
the syllable hypothesis is greedily re-segmented into prompt "words" (a lossy
step - a single syllable error can break a whole word), so treat WER as a
pessimistic upper bound.

| model | dev SER | test SER | dev WER~ | test WER~ |
|---|---:|---:|---:|---:|
| mono | 18.5 | 47.6 | 75 | 105 |
| tri1 (Δ+ΔΔ) | **9.2** | 63.6 | 40 | 115 |
| tri2 (LDA+MLLT) | 9.6 | 71.4 | 45 | 106 |
| tri3 (SAT) | 8.8 | **45.5** | 35 | 107 |
| tri3 + bMMI | 8.9 | 47.6 | 35 | 104 |

**Test SER split by speaker** (tri3 / mono): held-out male voice
38.6 / 31.8 vs held-out female voice 52.4 / 63.6.

**The fix that got here** - every run before 2026-09-11 had ~85-140% WER
because of one bug: the Kaldi `text` files carried prompt-level tokens
(`နံပါတ်` = 2 syllables, `၁၅` = 2 digits) while the lexicon carried
*syllables*, so **74% of transcript tokens were out-of-vocabulary**, got
mapped to `<UNK>`/`spn`, and the acoustic models aligned ~90% of frames to
silence/noise and never learned the speech. `norm_text()` now
syllable-segments the transcript; `mono_ali` silence dropped from ~86% to
34.5%, dev SER from ~85% to ~9%.

Stacked on top in the same run: audio from **`recordings_trim/`**
(normalised + energy-trimmed, ~22% of duration removed), **syllable-as-phone
units** with ASCII phone ids, **MFCC + 3 pitch features** (Burmese is tonal),
`--boost-silence 1.25`.

## Details

* **dev ~9% SER** (matched, closed-vocab) - a working recogniser.
* **test ~45% SER** (speaker-independent, one male + one female) - usable, not
  great. Still a gender gap (~38% male vs ~52% female for tri3) because
  training is all male.
* One of the training voices has severe mic clipping (100% of its clips, see
  that speaker's `NOTES.txt` under `../recordings/`) - a rerun with it
  excluded would isolate how much it hurts.
* WER via re-segmentation is crude; a better syllable->word aligner (or just
  reporting SER as the headline) would clean up that column.

Next levers: more speakers of both genders (assignment wants ~10), a proper
Burmese phone set instead of syllable units, and cleaner recording levels.

## Enhancement

After reading up on neural networks, data augmentation, and other improvement
ideas (see `../reference_asr/`), the next thing tried was a common trick
called error correction: look at where `tri3` tends to go wrong on data it
already knows well, write down those patterns, and use them to patch up its
guesses on new recordings before scoring them - no retraining needed.


Tested properly (patterns learned from `train` only, tuned and picked using
`dev`, checked once against `test`), the honest result is that **it doesn't
help at all** - the best setting found simply changes nothing, and any looser
setting makes the guesses worse, not better. The reason: the words in this
project are mostly numbers (phone numbers, quantities, dates), and the same
digit shows up correctly in some recordings and wrongly in others depending
on what's actually being said - there's no fixed "this is usually wrong, fix
it to that" pattern to learn, so a rule that helps one recording actively
breaks another. This isn't in `build_notebook.py` since it has no benefit;
it's recorded here so the attempt (and why it didn't pan out) isn't lost.

## tester (`asr_tester.py`) - "open test" with a UI

A small PyQt6 app (same stack as `recording_tool/recorder.py`) for demoing the
trained model interactively - record or load any WAV, transcribe it with one
of the trained models, optionally type what was actually said and see the SER.

```bash
../../.venv/bin/python asr_tester.py
```

- **Record (Space)** or **Load WAV...**, then **Transcribe (T)**.
- Model dropdown lists all trained models found under `s5/exp/*` (auto-detected):
  `mono`, `tri1`, `tri2`, `tri3`, `tri3_mmi_b0.1`. `mono` is the default
  (simplest, no fMLLR); `tri3` has the best test SER (see Results above). `tri3`/`tri3_mmi_b0.1` decode too, but their fMLLR
  transform is normally estimated per speaker from many utterances - here
  it's estimated from just the one clip you recorded, so their output is more
  of a curiosity than their real capability.
- Decoding takes well under a second for mono/tri1/tri2 (~0.6-0.9s); tri3 and
  tri3+MMI add an fMLLR estimation pass first, ~1.1-1.7s total - still fine
  live.
- Backend is `asr_backend.py` (`transcribe(wav_path, model="mono")`) - reusable
  from a script or another notebook cell, independent of the GUI.
- **Set expectations before demoing live**: training is 4 male speakers. On
  a clip from a *training* speaker it's near-perfect on the number/phone
  prompts; on an unseen male voice expect ~38% syllable error, on a female
  voice ~52% (all-male training). Prompts from `mini-asr-v1.txt` read the
  same way as the recordings are the best bet. The app's banner says this up
  front.

## Runtime

HMM-GMM stages on this corpus are minutes each on the 10-core VM; SGMM2 a bit
more. First `mkgraph` builds `G.fst`/`HCLG` — fast here (tiny vocab). If the
container can't see `data/` paths, check the `-v {PROJ}:/workspace` mount and
that `wav.scp` holds `/workspace/recordings/...` paths.
