# Assignment 4 — Mini ASR: setup & recording guide

Everything needed to record the corpus (Steps 1–2). Step 3 (Kaldi HMM-GMM /
DNN, WER/SER) is not covered here.

All commands are run **from the `B2-Assignment4/` folder** unless stated
otherwise. `../.venv` is the shared virtual-env at the repo root
(`AIEF_B2_References_Assignment/.venv`, Python 3.13).

---

## 1. Install (once per machine)

### macOS

```bash
# PortAudio system library (the sounddevice wheel links against it)
brew install portaudio

# Python packages into the shared repo venv
../.venv/bin/python -m pip install -r requirements.txt
```

### Windows

```powershell
# from the repo root, activate the venv (or create one: py -3 -m venv .venv)
..\.venv\Scripts\Activate.ps1
python -m pip install -r B2-Assignment4\requirements.txt
```

PortAudio ships inside the `sounddevice` wheel on Windows — no separate install.

### Linux

```bash
sudo apt install libportaudio2 portaudio19-dev     # Debian/Ubuntu
../.venv/bin/python -m pip install -r requirements.txt
```

### Verify

```bash
../.venv/bin/python - <<'PY'
import sounddevice as sd, PyQt6.QtCore as c
print("PortAudio:", sd.get_portaudio_version()[1])
print("Qt:", c.QT_VERSION_STR)
print("default input:", sd.query_devices(kind='input')['name'])
PY
```

You should see your microphone as the default input. If it errors, see
**Troubleshooting**.

---

## 2. Microphone permission (macOS)

macOS blocks mic access until you grant it to the app that launches Python:

- **System Settings → Privacy & Security → Microphone**
- Enable the terminal you launch from (**Terminal**, **iTerm**) **or Visual
  Studio Code** if you run `record.sh` from the VS Code integrated terminal.
- Quit and reopen that app after granting.

First run also pops a one-time "… wants to use the microphone" dialog — click
**Allow**. If you never saw it and recordings are silent, permission is the
cause.

---

## 3. Record — 5 passes per speaker

Each speaker records the 150 prompts **5 times**. One output folder per pass.

### Easiest: the launcher

```bash
./record.sh WinOo 1        # -> recordings/WinOo/Rec1/
./record.sh WinOo 2        # -> recordings/WinOo/Rec2/
./record.sh WinOo 3        # -> recordings/WinOo/Rec3/
./record.sh WinOo 4        # -> recordings/WinOo/Rec4/
./record.sh WinOo 5        # -> recordings/WinOo/Rec5/
```

`WinOo` is a short tag for the output folder name. In the **Speaker
Information** dialog that pops up, type your **full name with no spaces**
(e.g. `WinOo`) and use the **exact same string every pass** — it becomes the
Kaldi speaker id and the `utt_id` prefix.

macOS: you can also double-click **`record.command`** in Finder; it asks for
the name and pass number.

### Direct (all platforms, incl. Windows)

```bash
# macOS / Linux
../.venv/bin/python recording_tool/recorder.py -p mini-asr-v1.txt -d recordings/WinOo/Rec1 -m ordered
```
```powershell
# Windows
python .\recording_tool\recorder.py -p .\mini-asr-v1.txt -d .\recordings\WinOo\Rec1 -m ordered
```

### Controls

| Key | Action |
|-----|--------|
| `Space` | start / stop recording |
| `P` | play back the take you just recorded |
| `S` | save (writes the `.wav` + appends to `wav.scp` / `text` / `utt2spk` / `recordings.tsv`) |
| `N` | next prompt |
| `B` | previous prompt |
| `Ctrl+D` | delete the selected item in the list |

**Per take:** short silence (~0.5–1 s) → read the prompt → short silence →
`Space` to stop → `P` to check → `S` to save → `N`. Wait ~1 second between
saves (the `utt_id` timestamp is second-resolution).

**Pass style:**

| Pass | Delivery |
|------|----------|
| 1–3 | normal reading pace |
| 4 | a little faster / brisker |
| 5 | natural, conversational — not a "reading" tone |

**Audio hygiene:** quiet room, no fan / wind / traffic / music; keep a
constant mouth-to-mic distance; head still. Read numbers carefully
("ထောင်" vs "သောင်း"); some prompts are digit strings meant to be read
digit-by-digit. Do **not** post-edit or re-encode the WAVs — the recorder
already writes 16 kHz / mono / 16-bit.

---

## 4. Validate before submitting

```bash
../.venv/bin/python tools/validate_recording.py recordings/WinOo/Rec1 --expect-prompts 150
../.venv/bin/python tools/validate_recording.py recordings/WinOo/Rec*      # all passes at once
```

Checks: audio format (16 kHz/mono/16-bit), `wav.scp`/`text`/`utt2spk` mutually
consistent and sorted, no orphan utt-ids or WAVs, non-empty transcripts,
duration stats, clipping / very-short flags, `recordings.tsv` cross-check.
Fix every **ERROR** before you submit (warnings are advisory).

---

## 5. Merge into a Kaldi data dir

`wav.scp` contains absolute paths from the recording machine, and entries are
appended unsorted. To build a clean, portable Kaldi `data/` dir:

```bash
# one speaker, 5 passes
../.venv/bin/python tools/merge_data_dir.py -o data/WinOo recordings/WinOo/Rec*

# whole group
../.venv/bin/python tools/merge_data_dir.py -o data/all recordings/*/Rec*

# point wav.scp at where the audio will live on the Kaldi box
../.venv/bin/python tools/merge_data_dir.py -o data/all --wav-prefix /corpus/wav recordings/*/Rec*
```

Output is byte-sorted, de-duplicated, and includes `spk2utt`. On the training
box, `utils/fix_data_dir.sh data/all` should then be a no-op.

**Loudness-normalised copy** (for perceptual spot-checks / neural models; raw
audio stays the training source):

```bash
../.venv/bin/python tools/normalize_recordings.py --out-root recordings_norm/<speaker> recordings/<speaker>/Rec*
```

---

## 6. Submission (Step 2)

Each member drops their `recordings/<speaker>/` folder into the shared repo.
The team lead then builds the package with one command:

```bash
../.venv/bin/python tools/package_corpus.py --group "Group-N" \
    --zip aief_b2_assignment4.zip recordings/*/
```

This creates, under `raw-corpus/`:

```
raw-corpus/<speaker>/Rec1..5/     raw pass dirs (portable wav.scp)
raw-corpus/<speaker>/kaldi_data/  per-speaker merged Kaldi dir
raw-corpus/<speaker>/MANIFEST.txt <-- fill in gender + group
raw-corpus/kaldi_data_all/        all speakers merged
raw-corpus/GROUP_MANIFEST.txt
```

and `aief_b2_assignment4.zip` to upload / share via Google Drive.
`package_corpus.py` runs `validate_recording.py` on every speaker first and
refuses to package one with ERRORs (use `--force` to override).

**Fill `gender:` in every `raw-corpus/<speaker>/MANIFEST.txt` before you zip** —
the Step 3 train/dev/test split is **by speaker**, with the test set = all
utterances of one male + one female speaker.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sounddevice'` | `../.venv/bin/python -m pip install -r requirements.txt` |
| `OSError: PortAudio library not found` (macOS) | `brew install portaudio`, then reinstall `sounddevice` |
| Recordings are silent / all zeros | Grant mic permission (§2); check the right input device is default |
| Wrong mic used | Set the input in **System Settings → Sound → Input**, or macOS menu-bar; the recorder uses the OS default |
| GUI window doesn't appear | Must run on a machine with a display and a logged-in desktop session; not over plain SSH |
| Prompt text shows boxes (□□□) | Install a Myanmar font: `Pyidaungsu` / `Noto Sans Myanmar` / `Padauk` |
| Duplicate `utt_id` reported by the validator | Two saves in the same second — delete the extra WAV + its lines and re-record |
| `record.sh: Permission denied` | `chmod +x record.sh` |
