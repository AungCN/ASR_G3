#!/usr/bin/env bash
# Double-clickable wrapper for record.sh (macOS Finder).
# It asks for speaker name + pass number, then launches the recorder.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -p "Speaker short name (no spaces, e.g. WinOo): " SPEAKER
read -r -p "Pass number (1-5): " PASS

exec "$HERE/record.sh" "$SPEAKER" "$PASS"
