# step3_asr/ — Kaldi ASR pipeline (Assignment 4, Step 3)

Builds the HMM-GMM + advanced models and scores WER/SER, driven from one
Jupyter notebook. Kaldi runs in Docker.

## Files

| File | What |
|---|---|
| `mini_asr_kaldi.ipynb` | the pipeline notebook — run this |
| `build_notebook.py` | regenerates the `.ipynb` (edit here, not the notebook) |
| `mm_syllable.py` | Myanmar syllable segmentation |
| `../tools/trim_silence.py`, `../tools/normalize_recordings.py` | build `recordings_trim/` |
| `asr_backend.py` | decode one WAV with a trained model (used by `asr_tester.py`) |
| `asr_tester.py` | PyQt6 GUI: record/load audio, transcribe, score |
| `s5/` | Kaldi working dir, created by the notebook — large, disposable |
| `results_wer.png`, `s5/RESULTS.csv` | written by the results cell |

## One-time setup

```bash
docker pull tklwin/kaldi-apple-silicon:latest   # Docker Desktop must be running
cd B2-Assignment4/step3_asr
../../.venv/bin/jupyter lab mini_asr_kaldi.ipynb
```

Register the venv as a kernel once if it isn't listed:

```bash
../../.venv/bin/python -m ipykernel install --user --name aief-b2 --display-name "Python 3 (AIEF-B2 venv)"
```

## What the notebook does

1. **Config** — paths, split, Docker params.
2. **`kaldi()` helper** — runs a bash snippet in the image, project mounted at `/workspace`.
3. **Work dir** — `path.sh`, `cmd.sh`, `steps/`+`utils/` from the image's `wsj/s5`.
4. **Data prep** — `data/{train,dev,test}`, NFC + punctuation-stripped transcripts.
5. **Lexicon / `lang/`** — syllable = unit, ASCII phone ids (`S001`..).
6. **LM** — add-one bigram over training syllables.
7. **Features** — MFCC + 3 pitch features + CMVN.
8. **Decode/score helper** — SER direct from Kaldi; WER via re-segmentation into words.
9-12. **HMM-GMM** — monophone → tri1 (Δ+ΔΔ) → tri2 (LDA+MLLT) → tri3 (SAT).
13. **Advanced** — boosted-MMI on tri3 (no SGMM2 in this image; chain-TDNN cell disabled by default).
14. **Results** — WER/SER table + bar chart.
15. **Notes / limitations.**

## Split (matches the assignment spec)

**7 speakers, 5 for training, 2 held out for testing** (one male, one
female). Actual folder names live in `speaker_config.py` (gitignored, see
main [README](../README.md)). Speakers are referred to by role, not name.

| set | speakers | passes | utts |
|---|---|---|---|
| train | 5 training voices (4 male, 1 female) | random 88% (seed 1234) | 3306 |
| dev | same 5 | random 12% (seed 1234) - matched to train | 451 |
| test | held-out male + held-out female | all 5 | 1497 - speaker-independent |

`GMM_SIZES` was sized for 4 speakers and not re-tuned for the current 5. Add
a speaker: update `speaker_config.py`, `python build_notebook.py`, rerun.

**Two bugs found adding the 5th speaker** (worth knowing before adding a 6th):
1. `s5/` is reused between runs; data-prep only overwrote `wav.scp`/`text`/
   `utt2spk`, leaving old cached features behind. Kaldi silently used the
   stale cache and dropped the new speaker with no error. Fixed: wipe each
   `data/<set>/` dir before rewriting it.
2. Her utt-ids used different capitalization than her folder name, which
   broke Kaldi's required per-speaker sort order once a 5th speaker was
   added. Fixed: read each speaker's id from their own recordings, not the
   folder name.

## Results (2026-09-14, 5 training speakers)

SER (syllable error rate) is the primary metric. WER is approximate
(syllable hyp re-segmented into words) — treat it as an upper bound.

| model | dev SER | test SER | dev WER~ | test WER~ |
|---|---:|---:|---:|---:|
| mono | 20.6 | 50.8 | 87 | 104 |
| tri1 (Δ+ΔΔ) | 11.8 | 61.2 | 54 | 125 |
| tri2 (LDA+MLLT) | 10.7 | 67.4 | 56 | 116 |
| tri3 (SAT) | **9.5** | **48.7** | 54 | 110 |
| tri3 + bMMI | 9.7 | 50.8 | 56 | 111 |

Test SER by speaker (tri3 / mono): held-out male 32.8 / 30.5, held-out
female 64.7 / 71.3.

**Honest finding:** adding one female training voice did not close the
gender gap — it got slightly worse (female test SER 52.4% → 64.7% for tri3),
while the male side improved (38.6% → 32.8%). Likely causes: one voice isn't
"more female voices" in general, `GMM_SIZES` wasn't retuned for the bigger
set, and her audio needed unusually heavy trimming (45% vs ~22% typical),
suggesting a different recording setup. Not a reason to abandon the plan —
one added speaker is too small a sample to judge it — but not "fixed" either.

**Root fix that got the pipeline working at all:** Kaldi's `text` held
prompt-level tokens while the lexicon held syllables, so 74% of tokens were
out-of-vocabulary and models learned almost nothing (~85-140% WER). Fixed by
syllable-segmenting the transcript. Combined with silence-trimmed audio,
syllable-as-phone units, and MFCC+pitch features, dev SER went from ~85% to
~9%.

## Details

* **dev ~9.5% SER** (matched) — a working recognizer.
* **test ~49% SER** (speaker-independent) — usable, not great. Gender gap
  persists (~33% male vs ~65% female for tri3).
* One training voice has severe mic clipping (see its `NOTES.txt` under
  `../recordings/`).
* WER via re-segmentation is crude — SER is the more reliable number.

Next: more speakers of both genders (assignment wants ~10), re-tune
`GMM_SIZES`, a proper Burmese phone set, cleaner recording levels.

## Tried: automatic error correction (didn't help)

Looked at where `tri3` tends to go wrong and tried patching its guesses
after decoding, using patterns learned only from `train`. Tuned on `dev`,
checked once on `test`: **no improvement** — the best setting changes
nothing, looser settings make it worse. Most content here is numbers, and
the same digit is right in one recording and wrong in another, so there's no
stable "this token is usually wrong" pattern to learn. Not added to
`build_notebook.py` since it has no benefit.

## Live tester (`asr_tester.py`)

```bash
../../.venv/bin/python asr_tester.py
```

- **Record (Space)** or **Load WAV...**, then **Transcribe (T)**.
- Model dropdown auto-detects trained models under `s5/exp/*`: `mono`,
  `tri1`, `tri2`, `tri3`, `tri3_mmi_b0.1`. `tri3` has the best test SER.
  `tri3`/`tri3_mmi_b0.1` need fMLLR, normally estimated from many
  utterances — here it's just the one clip, so treat their output as a
  curiosity.
- Decode time: <1s for mono/tri1/tri2; ~1-1.7s for tri3/tri3+MMI (fMLLR pass).
- Backend: `asr_backend.py` (`transcribe(wav_path, model="mono")`), reusable
  outside the GUI.
- Expectations: training is 5 speakers (4 male, 1 female). Near-perfect on a
  training speaker's clips; ~33% syllable error on an unseen male voice,
  ~65% on an unseen female voice. Update the app's banner if these change.

## Runtime

HMM-GMM stages are minutes each. If the container can't see `data/` paths,
check the `-v {PROJ}:/workspace` mount and that `wav.scp` holds
`/workspace/recordings/...` paths.
