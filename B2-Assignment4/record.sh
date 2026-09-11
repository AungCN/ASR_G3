#!/usr/bin/env bash
# Assignment 4 - Mini ASR :: recording launcher
# -----------------------------------------------------------------------------
# Runs recording_tool/recorder.py (v0.91) with the correct prompt file and a
# per-speaker, per-pass output directory, using the shared repo venv.
#
# Usage:
#   ./record.sh <SpeakerShortName> <PassNumber 1-5>
#
# Example (speaker "Win Oo", full name typed into the GUI dialog as "WinOo"):
#   ./record.sh WinOo 1      # -> recordings/WinOo/Rec1/
#   ./record.sh WinOo 2      # -> recordings/WinOo/Rec2/
#   ...
#   ./record.sh WinOo 5      # -> recordings/WinOo/Rec5/
#
# In the GUI "Speaker Information" dialog, type your FULL name with NO spaces
# (e.g. WinOo). Use the SAME string for every pass so utt-ids stay consistent.
#
# Passes 1-3: read normally. Pass 4: read a little faster. Pass 5: natural,
# conversational delivery (not "reading" tone). See ASSIGNMENT.md "Recording Guide".
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# Pick a Python: repo venv first, then python3 on PATH.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
    echo "WARN: repo venv not found at $REPO_ROOT/.venv - using $PY" >&2
else
    echo "ERROR: no Python found. See SETUP.md." >&2
    exit 1
fi

SPEAKER="${1:-}"
PASS="${2:-}"

if [[ -z "$SPEAKER" || -z "$PASS" ]]; then
    echo "Usage: ./record.sh <SpeakerShortName> <PassNumber 1-5>" >&2
    echo "Example: ./record.sh WinOo 1" >&2
    exit 2
fi

if ! [[ "$PASS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: pass number must be a positive integer (got '$PASS')." >&2
    exit 2
fi

OUT_DIR="$HERE/recordings/${SPEAKER}/Rec${PASS}"

# Dependency check with a friendly message.
if ! "$PY" -c "import PyQt6, sounddevice, numpy" 2>/dev/null; then
    echo "ERROR: missing Python deps (PyQt6 / sounddevice / numpy)." >&2
    echo "Install with:  $PY -m pip install -r \"$HERE/requirements.txt\"" >&2
    echo "macOS also needs:  brew install portaudio" >&2
    exit 3
fi

echo "Speaker    : $SPEAKER"
echo "Pass       : $PASS"
echo "Prompts    : $HERE/mini-asr-v1.txt  (150 prompts, ordered)"
echo "Output dir : $OUT_DIR"
echo "Python     : $PY"
echo
echo "In the dialog, type your full name WITH NO SPACES (e.g. ${SPEAKER})."
echo "Controls: Space=rec/stop  P=play  S=save  N=next  B=prev  Ctrl+D=delete"
echo

exec "$PY" "$HERE/recording_tool/recorder.py" \
    -p "$HERE/mini-asr-v1.txt" \
    -d "$OUT_DIR" \
    -m ordered
