# Adding a new speaker — do-it-yourself checklist

Whatever shape their recorder.py output arrives in, do this every time. All
commands run from `B2-Assignment4/`, using the shared venv at `../.venv`.
`<name>` below is whatever folder name you give that person's recordings
under `recordings/` (their own choice — see the main [README](README.md) for
why recording folders and personal names generally stay out of git).

## 1. Restructure to Rec1..Rec5

```bash
../.venv/bin/python tools/normalize_speaker.py recordings/<name> --dry-run   # preview
../.venv/bin/python tools/normalize_speaker.py recordings/<name>             # apply
```

Works no matter what their pass folders are called (`rec_1439_28Aug2026`,
`<name>_Rec1`, `<name>`/`<name>1`/`<name>2`/...) — it sorts them by the
recording timestamps inside, not the folder names.

## 2. Validate

```bash
../.venv/bin/python tools/validate_recording.py recordings/<name>/Rec* --expect-prompts 150
```

Fix any **ERROR**, including a "passes under this folder don't share one
speaker id" error — that means the recorder was given a different name for
different passes; fix by rewriting the 2nd column of each `RecN/utt2spk` to
one consistent value. Warnings (short clip, missing prompt) are fine —
optionally jot them in `recordings/<name>/NOTES.txt` (free text; it gets
pulled into that speaker's package notes automatically). A wall of "possible
clipping" warnings is worth a closer look — it can mean the microphone level
was too hot and the recording is genuinely distorted, not a false alarm.

## 3. Rebuild the submission package + zip

```bash
../.venv/bin/python tools/package_corpus.py --group "Group-N" \
    --gender <name>=male_or_female \
    --gender <other-name>=male_or_female \
    --zip aief_b2_assignment4.zip recordings/*/
```

List every speaker's gender each time — the script rebuilds `raw-corpus/`
from scratch, so copy the growing `--gender` list forward as people are added.

## 4. Decide: does this speaker join training or testing?

The split lives in `step3_asr/speaker_config.py` (not committed to git — see
`speaker_config.example.py` for the template if you don't have one yet):

```python
TRAIN_SPEAKER_DIRS = ["<name1>", "<name2>", "<name3>"]
TEST_SPEAKER_DIRS  = {"male": "<name4>", "female": "<name5>"}
```

**Default recommendation: add new speakers to `TRAIN_SPEAKER_DIRS`.** More
training voices is consistently the highest-value thing you can do for
accuracy. Only change `TEST_SPEAKER_DIRS` if your team wants a different
held-out male/female pair — the assignment only needs one of each.

After editing `speaker_config.py`, regenerate the notebook:
```bash
cd step3_asr
../../.venv/bin/python build_notebook.py
```

## 5. Run the notebook

```bash
../../.venv/bin/jupyter lab mini_asr_kaldi.ipynb
```
Menu: **Kernel → Restart Kernel and Run All Cells**. Docker Desktop must be
open first. Takes longer as the corpus grows.

## 6. Check the results

Scroll to the bottom two cells (results table + bar chart), or after it
finishes:
```bash
cat s5/RESULTS.csv
```
Compare to the numbers in `step3_asr/README.md` — accuracy should trend
better as training voices increase.
