"""
mm_syllable.py - Myanmar (Burmese) syllable segmentation + helpers for ASR.

`syllable_break(text)` splits a Unicode Myanmar string into syllables. The rule
is the widely used one (Ye Kyaw Thu's `sylbreak`): a break is inserted *before*
a consonant / independent-vowel / digit that is **not** preceded by the stacking
sign VIRAMA (U+1039) or the kinzi sequence, and is not itself a medial /
dependent vowel / tone mark. ASCII runs, spaces and punctuation are passed
through as their own tokens.

Used for:
  * the pronunciation lexicon  (syllable  ->  sequence of character "phones")
  * Syllable Error Rate (SER)  (tokenise ref & hyp into syllables, score)

Pure standard library. `python mm_syllable.py "..."` prints the segmentation.
"""
from __future__ import annotations

import re
import sys

# Myanmar Unicode block pieces
MY_CONSONANT = r"က-အ"          # က .. အ
MY_INDEP_VOWEL = r"ဣ-ဧဩဪ"
MY_DIGIT = r"၀-၉"
MY_VIRASIGN = "္"                    # ASAT is 103A; 1039 is the stacking virama
MY_ASAT = "်"
MY_DEP = r"ါ-း်-ဿၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏႚ-ႝ"

_BREAK_BEFORE = re.compile(
    r"(?<!" + MY_VIRASIGN + r")"                       # not right after a stacker
    r"(?=[" + MY_CONSONANT + MY_INDEP_VOWEL + MY_DIGIT + r"])"
)


def syllable_break(text: str) -> list[str]:
    """Return the list of syllable / non-Myanmar tokens in `text`."""
    out: list[str] = []
    for chunk in text.split():
        if not chunk:
            continue
        # split Myanmar vs non-Myanmar spans, keep order
        for span in re.findall(r"[က-႟]+|[^က-႟]+", chunk):
            if re.match(r"[က-႟]", span):
                pieces = [p for p in _BREAK_BEFORE.split(span) if p]
                # merge a leading lone dependent sign onto the previous piece
                merged: list[str] = []
                for p in pieces:
                    if merged and re.match(r"[" + MY_DEP + MY_VIRASIGN + r"]", p):
                        merged[-1] += p
                    else:
                        merged.append(p)
                out.extend(merged)
            else:
                out.append(span)
    return out


def syllables_str(text: str, sep: str = " ") -> str:
    return sep.join(syllable_break(text))


# characters that are their own "phone" but never a syllable nucleus on their own
_COMBINING = set("္")


def char_phones(syllable: str) -> list[str]:
    """Pronunciation of a syllable = its Unicode code points, with the stacking
    virama glued to the consonant it stacks (so 'k+VIRAMA+s' -> ['k', '_s'])."""
    phones: list[str] = []
    i = 0
    s = syllable
    while i < len(s):
        ch = s[i]
        if ch == "္" and i + 1 < len(s):
            phones.append("_" + s[i + 1])   # stacked consonant
            i += 2
            continue
        phones.append(ch)
        i += 1
    return phones


if __name__ == "__main__":
    arg = " ".join(sys.argv[1:]) or "ဖုန်းနံပါတ် ၀၉၁၂၃၄၅၆၇၈ ပါ"
    syls = syllable_break(arg)
    print("input   :", arg)
    print("syllables:", " | ".join(syls), f"  ({len(syls)} tokens)")
    for s in syls:
        print(f"  {s!r:12s} -> {char_phones(s)}")
