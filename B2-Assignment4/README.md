# Myanmar Mini Speech Recognition Project

A small class project that builds a speech-recognition system for spoken
Myanmar (Burmese) from scratch: recording voices reading a short script,
turning those recordings into a trained model, and a simple app to try the
model out live. This is coursework for the assignment described in
[ASSIGNMENT.md](ASSIGNMENT.md).

## What's here

| Stage | What it does | Where |
|---|---|---|
| 1. Record | A desktop app walks each person through reading 150 prompts (numbers, dates, phone numbers, short sentences) out loud and saves the audio | [`recording_tool/`](recording_tool/) |
| 2. Check & package | Scripts check the recordings for problems and bundle everyone's audio together for submission | [`tools/`](tools/) |
| 3. Train & evaluate | A notebook turns the recordings into a working speech-recognition model and measures how accurate it is | [`step3_asr/`](step3_asr/) |
| 4. Try it live | A small app records or loads a clip and shows what the trained model heard | [`step3_asr/asr_tester.py`](step3_asr/asr_tester.py) |

## Where things stand

As of the most recent run, the system:

- gets roughly **9 out of 10 syllables right** on speech from the people it
  was trained on (read back to it, but not the exact same recordings) — that's
  a genuinely working recognizer for this small script of prompts,
- gets about **half the syllables right** on a voice it has never heard
  before,
- and is noticeably better at recognizing an unfamiliar **male** voice than
  an unfamiliar **female** one. One woman's voice has since been added to
  training, but that alone didn't close the gap — if anything it got
  slightly worse for the unfamiliar female test voice, which is an honest,
  slightly surprising result. See "the honest finding" in
  [`step3_asr/README.md`](step3_asr/README.md) for the numbers and why.

In short: it works, but it still needs more voices — especially more women's
voices, plural — to become reliably accurate for anyone. One extra voice
wasn't enough to move the needle either way for sure; that's the clearest
next step. Full details, numbers, and what was tried along the way are in
[`step3_asr/README.md`](step3_asr/README.md).

## Trying it yourself

Everything below assumes the one-time setup in [SETUP.md](SETUP.md) is done
(installs a few Python packages and, on macOS, one system library for
microphone access).

**Record your own voice reading the prompts:**
```bash
./record.sh YourName 1
```
(repeat for passes 2 through 5 — see [ASSIGNMENT.md](ASSIGNMENT.md) for why 5
passes, and [SETUP.md](SETUP.md) for the full walkthrough)

**Hear the trained model guess what you said:**
```bash
cd step3_asr
../../.venv/bin/python asr_tester.py
```
Record a short prompt, press Transcribe, and see the result. There's a
built-in note in the app explaining what to expect (see "Where things stand"
above) so a wrong answer doesn't look like something is broken.

**See the whole training process, end to end:**
```bash
cd step3_asr
../../.venv/bin/jupyter lab mini_asr_kaldi.ipynb
```
This is the notebook that takes raw recordings all the way through to a
trained model and a results table. It's meant to be read top to bottom like a
report as well as run.

**Adding someone new to the project:** see [ADD_SPEAKER.md](ADD_SPEAKER.md)
for the step-by-step checklist.

## About the data

The voice recordings themselves — and anything built directly from them
(cleaned-up copies, the packaged submission, the trained model files) — are
**deliberately not stored in this repository.** Two reasons:

1. **They're personal.** They're recordings of real people's voices reading
   personal-ish prompts (phone-number-shaped numbers, made-up dates, and so
   on), and they should stay with the people who made them, not end up in a
   public place by accident.
2. **They're large.** The full set of recordings is well over a gigabyte,
   which doesn't belong in a source-code repository.

Because of this, the code in this project never hard-codes anyone's name —
it refers to recordings by role instead (for example, "training voice 1" or
"the held-out female voice"). The one place actual folder names are needed is
a small local file, `step3_asr/speaker_config.py`, which is excluded from git
(see [`.gitignore`](.gitignore)). To reproduce this project with your own
recordings, copy `step3_asr/speaker_config.example.py` to
`step3_asr/speaker_config.py` and fill in your own folder names — see
[ADD_SPEAKER.md](ADD_SPEAKER.md) for the full process.

## Project layout

```
B2-Assignment4/
  ASSIGNMENT.md        the original assignment brief
  SETUP.md              one-time setup (install steps, microphone permission)
  ADD_SPEAKER.md        checklist for adding a new voice to the project
  mini-asr-v1.txt       the 150 prompts everyone reads
  recording_tool/       the recording app
  tools/                scripts to check, clean up, and package recordings
  step3_asr/            the training notebook, results, and the live tester
  recordings/           (not in git) - raw recordings, one folder per person
  raw-corpus/           (not in git) - packaged recordings ready to submit
```

## Credits & references

- Assignment and recording tool originally provided by **Ye Kyaw Thu, LU Lab.,
  Myanmar** — [ye-kyaw-thu/AIE-F-B2](https://github.com/ye-kyaw-thu/AIE-F-B2).
- Speech-recognition training uses the open-source
  [Kaldi Speech Recognition Toolkit](https://kaldi-asr.net/), run through the
  [`tklwin/kaldi-apple-silicon`](https://hub.docker.com/r/tklwin/kaldi-apple-silicon)
  Docker image (a build of Kaldi that runs natively on Apple Silicon Macs).
- Recording participants: Aint Kyi Phyu Sin, Aung Ko Ko Oo, Htun Aung Kyaw, Phyo Myat Oo, Thein Kyaw Lwin, Aung Chan Nyein
