# summary

A simple walk-through of the work done on this project, without technical
terms. For numbers and details, see [README.md](README.md) and
[step3_asr/README.md](step3_asr/README.md).

Goal: build a program that listens to spoken Myanmar (mostly numbers, dates,
short sentences) and guesses what was said, by recording real voices and
using them to teach it.

## 1. Recording setup

Built a recording app that reads out 150 prompts and saves the audio. Found
and fixed a bug where deleted recordings weren't fully cleaned up.

## 2. Recording

Seven people recorded the full prompt list, five times each. Five people
teach the program; two (one man, one woman) are kept aside purely for
testing on voices it's never heard.

## 3. Checking recordings for problems

Built tools to catch recording mistakes — mismatched names, a person
accidentally split across different sessions, mislabeled files. Found and
fixed several real issues this way. One person's recordings were too loud
and distorted; noted since it can't be fixed after the fact.

## 4. Bundling everything together

A tool packages everyone's recordings into one organised bundle, ready for
training.

## 5. First training attempt

Disappointing: on a new voice, the program barely guessed 2-3 words right.

## 6. Root cause

The program was taught on whole words, but internally set up to recognise
much smaller sound pieces (syllables). That mismatch meant it barely learned
anything. Fixing it was the single biggest improvement in the project — the
program went from "barely gets a couple of words" to correctly hearing
roughly 9 out of 10 sound pieces on speech from people it trained on.

## 7. Cleaning up the audio

Trimmed silence, evened out volume, and added pitch information (useful for
Myanmar, since tone can change meaning). Further improved accuracy.

## 8. Retraining and honest results

- On trained voices (different recordings), gets **9 out of 10** sound
  pieces right — a genuinely working result.
- On a totally new voice, gets a bit **under half** right.
- Noticeably better on new male voices than new female ones. A woman's
  voice was added to training to test if that helps — it didn't; results
  got slightly worse for new female voices. Honest result, not hidden: one
  extra voice isn't enough to prove anything either way.

## 9. A small app to try it live

Record or load a clip, pick a trained version of the program, and see what
it guessed, with a note on what accuracy to expect.

## 10. Write-ups and safe sharing

Wrote plain-language docs on what the project does, plus step-by-step
guides for recording and adding a new person. No real names appear in code
or shared files — people are referred to by role, and real names live in
one local file that's never shared. We then published the tools, docs, and
results on GitHub, but not the voice recordings themselves (personal, and
too large).

## 11. Trying advanced ideas — honestly

Reviewed research papers on Myanmar speech recognition for other techniques
worth trying. One idea: automatically patch the program's common mistakes
after it guesses. Built and tested it properly — **it didn't help**. Most
content here is numbers, and the same digit is right in one sentence and
wrong in another, so there's no stable "this is usually wrong" pattern to
learn. Written up here rather than left out, since knowing what doesn't work
is useful too.

## Where things stand now

- Works well on voices similar to the ones it trained on.
- Still noticeably weaker on brand-new voices, especially women's, due to
  an imbalance in who recorded so far.
- Clearest next step: record more people, especially more women, rather
  than more processing tricks.
