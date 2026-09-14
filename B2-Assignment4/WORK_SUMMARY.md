# summary

This is a simple walk-through of the work done on this project so far, from
start to now, without technical terms. For numbers and details, see
[README.md](README.md) and [step3_asr/README.md](step3_asr/README.md).

Build a small computer program that can listen to someone speaking Myanmar
(Burmese) — mostly numbers, dates, and short everyday sentences — and guess
what they said, by first collecting voice recordings from real people and
using them to teach the program.

## 1. Building the recording setup

We started with a recording app that shows a person a list of things to say
(150 short prompts) and saves what they say as sound files. We tested it,
found a small bug where deleted recordings weren't fully cleaned up, and
fixed it before anyone recorded for real.

## 2. Recording

Six people recorded themselves reading the full list of prompts, each
person doing it five times (reading it plainly, then a bit faster, then more
naturally, to give some variety). That gave us a solid pile of recordings to
work with — four people set aside to teach the program, and two people (one
man, one woman) set aside purely to test it on voices it had never heard.

## 3. Checking the recordings for problems

Before using any of it, we built tools to check the recordings for mistakes —
things like a recording that didn't match its name, a person accidentally
saved under slightly different names across different sessions, or files that
were only half-cleaned-up after a deletion. We found and fixed a few real
problems this way, including one where a single person's five sessions had
accidentally been labelled as five different people, which would have
confused the training later. We also noticed one person's recordings were
often too loud and distorted, which we noted down since it can't be fixed
after the fact.

## 4. Bundling everything together

We wrote a tool that gathers all six people's recordings into one organised
package, checks that nothing is missing, and prepares it to be handed off to
the training step.

## 5. first Training

We built the actual training process next. The first full run was a
disappointment: when tested on a new voice, the program could barely guess
more than two or three words correctly. Something was clearly badly wrong.

## 6. Issues

After digging in, we found the core issue: the program was being told to
learn on "words," but internally it was set up to recognise much smaller
sound pieces (roughly, syllables). Because those two things didn't line up,
most of what the program was shown didn't make sense to it, so it barely
learned anything — it was mostly hearing "noise" instead of real speech.

Fixing this one mismatch was the single biggest improvement in the whole
project. Once it was fixed, the program went from "barely gets a couple of
words" to "correctly hears roughly 9 out of 10 sound pieces" on speech from
people it had trained on.

## 7. Cleaning up the audio itself

Alongside that fix, we also cleaned up the recordings: trimming the silence
at the start and end of each clip, evening out volume differences between
people, and giving the program a bit more information about the pitch of the
voice (useful for Myanmar, where tone can change meaning). All of this
further improved accuracy.

## 8. Retraining and measuring results honestly

With the fixes in place, we retrained everything and measured results
properly:

- On voices the program was trained on (but different recordings than it
  studied), it gets roughly **9 out of 10 sound pieces right** — a genuinely
  working result.
- On a completely new voice it has never heard, it gets a bit **under half**
  right.
- It does noticeably better on a new **male** voice than a new **female**
  voice. Almost everyone used for training so far has been male, so a woman's
  voice was added to training to see if that would help. It didn't - the
  program actually got a little worse on new female voices, not better. This
  is an honest result, not something we tried to hide — one extra voice
  isn't enough to prove or disprove anything, and more training voices of
  both kinds, especially women's voices, is still the clear next step.

## 9. A small app to try it live

We built a simple desktop app where you can record your own voice, or load a
saved clip, press one button, and see what the program guessed — with a
choice of which trained version of the program to use, and a short built-in
note explaining what kind of accuracy to expect so a wrong guess doesn't look
like something is broken.

## 10. Writing everything up and sharing it safely

We put together plain-language write-ups explaining what the project is,
what it does, and where things stand, plus step-by-step instructions for
recording, checking recordings, and adding a new person to the project later.

Because the recordings are personal — real people's voices — we made sure
that no one's actual name appears anywhere in the shared project files or
code. People are referred to by role instead (for example, "a training
voice" or "the held-out female voice"), and the real names are kept in a
single small file on our own computer that is never shared. Names are only
meant to be added, if the team chooses to, in one clearly-marked "credits"
spot in the write-up.

We then shared this project (the tools, the write-ups, and the trained
program's results) on GitHub — but not the actual voice recordings
themselves, both because they're personal and because they're too large to
put there sensibly.

## 11. Looking into more advanced ideas — and being honest when one didn't work

We read through a few research papers on Myanmar speech recognition to see
if there were smarter techniques worth trying — things like neural-network
based approaches, or automatically expanding the training data.

One promising-sounding idea was to add a second step after the program makes
its guess: look at the mistakes it commonly makes, and automatically patch
them up afterwards. We built and carefully tested this. The honest result:
**it didn't help.** Tested fairly — learning only from data it hadn't been
tested on — the best version of this patch-up step ended up changing
nothing, and any more aggressive version made results worse, not better.

The reason turned out to be simple: most of what people said in this project
were numbers (phone numbers, quantities, and so on), and the same digit is
correct in one sentence and wrong in another depending on context — so there
was no reliable general "this is usually a mistake" pattern to learn from.
We wrote this attempt up honestly rather than leaving it out, since knowing
what doesn't work is useful too.

## Where things stand now

- The program genuinely works on voices similar to the ones it trained on.
- It's still noticeably weaker on brand-new voices, especially women's
  voices, because of an imbalance in who recorded so far.
- The clearest next step is simply recording more people — more voices
  overall, and specifically more women's voices — rather than more clever
  processing tricks, since we've shown that extra processing has limited
  room left to help without more training data.
