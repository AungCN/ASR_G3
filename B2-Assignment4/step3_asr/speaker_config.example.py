"""
speaker_config.example.py - template. Copy this file to speaker_config.py
(which git ignores) and fill in the real folder names under ../recordings/
for your own set of recordings.
"""

# Folders used for training (a small slice is automatically held back for tuning)
TRAIN_SPEAKER_DIRS = ["speaker1", "speaker2", "speaker3", "speaker4"]

# Held-out folders for the final evaluation: one male, one female speaker
TEST_SPEAKER_DIRS = {"male": "test_speaker_m", "female": "test_speaker_f"}

# Specific recordings to exclude (e.g. known bad takes), as "<utt_id>" strings
# found in that speaker's recordings/<name>/<pass>/text file
DROP_UTTS: set[str] = set()
