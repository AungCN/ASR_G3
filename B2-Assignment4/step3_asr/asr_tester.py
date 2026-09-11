## Live ASR Tester - Assignment 4 Mini ASR (LU Lab., Myanmar)
## Record or load a WAV, run it through a trained Kaldi model (asr_backend.py),
## see the recognised text, and optionally score it against what was actually said.
##
## Same recording stack as recording_tool/recorder.py (PyQt6 + sounddevice),
## kept deliberately simple: this is a demo/QA tool, not a data-collection one.
##
## Run:  ../../.venv/bin/python asr_tester.py

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_backend import ASRError, available_models, transcribe  # noqa: E402
from mm_syllable import syllable_break  # noqa: E402

SAMPLE_RATE = 16000


def edit_distance(a: list[str], b: list[str]) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(a)][len(b)]


class TranscribeWorker(QThread):
    done = pyqtSignal(str, dict)
    failed = pyqtSignal(str)

    def __init__(self, wav_path: str, model: str):
        super().__init__()
        self.wav_path, self.model = wav_path, model

    def run(self):
        try:
            text, meta = transcribe(self.wav_path, self.model)
            self.done.emit(text, meta)
        except ASRError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"unexpected error: {e}")


class ASRTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini ASR - Live Tester (AIEF-B2, LU Lab.)")
        self.setGeometry(150, 150, 720, 620)
        self.audio_buffer: list[np.ndarray] = []
        self.is_recording = False
        self.current_wav: str | None = None
        self.worker: TranscribeWorker | None = None

        self._init_font()
        self._init_ui()

    def _init_font(self):
        for name in ("Pyidaungsu", "Myanmar3", "Padauk", "Myanmar Text", "Noto Sans Myanmar"):
            if name in QFontDatabase.families():
                self.app_font = QFont(name, 14)
                break
        else:
            self.app_font = QFont(); self.app_font.setPointSize(14)
        QApplication.setFont(self.app_font)

    def _init_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        v = QVBoxLayout(root)

        note = QLabel(
            "Trained on 4 male speakers. Syllable error rate: ~9% on held-out data from\n"
            "the training speakers, ~38% for an unseen male voice, ~52% for a female voice\n"
            "(training is all-male). Best on prompts from mini-asr-v1.txt read the same way\n"
            "as the recordings. tri3 / tri3+MMI estimate their fMLLR transform from just\n"
            "your one clip (normally many utterances), so treat those as a curiosity."
        )
        note.setStyleSheet("color: #a33; font-size: 12px;")
        note.setWordWrap(True)
        v.addWidget(note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_box = QComboBox()
        models = available_models()
        self.model_box.addItems(models or ["mono"])
        if "mono" in models:
            self.model_box.setCurrentText("mono")
        row.addWidget(self.model_box)
        row.addStretch()
        v.addLayout(row)

        self.record_btn = QPushButton("Start Recording (Space)")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self.toggle_recording)
        v.addWidget(self.record_btn)

        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play (P)")
        self.play_btn.clicked.connect(self.play_current)
        self.load_btn = QPushButton("Load WAV...")
        self.load_btn.clicked.connect(self.load_wav)
        self.transcribe_btn = QPushButton("Transcribe (T)")
        self.transcribe_btn.clicked.connect(self.do_transcribe)
        for b in (self.play_btn, self.load_btn, self.transcribe_btn):
            b.setMinimumHeight(36); btn_row.addWidget(b)
        v.addLayout(btn_row)

        self.status = QLabel("Ready. Record or load a WAV, then Transcribe.")
        self.status.setStyleSheet("font-style: italic; color: #555;")
        v.addWidget(self.status)

        self.result_label = QLabel("(no result yet)")
        self.result_label.setStyleSheet("font-size: 22px; padding: 12px; background: #458B74;")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.result_label)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("What was actually said (optional):"))
        self.ref_edit = QLineEdit()
        self.ref_edit.returnPressed.connect(self.score_current)
        ref_row.addWidget(self.ref_edit)
        self.score_btn = QPushButton("Score")
        self.score_btn.clicked.connect(self.score_current)
        ref_row.addWidget(self.score_btn)
        v.addLayout(ref_row)

        self.score_label = QLabel("")
        v.addWidget(self.score_label)

        v.addWidget(QLabel("History:"))
        self.history = QListWidget()
        v.addWidget(self.history)

    def keyPressEvent(self, event):
        # Space/P/T shortcuts (mirrors recording_tool/recorder.py's key layout)
        if event.key() == Qt.Key.Key_Space:
            self.toggle_recording()
        elif event.key() == Qt.Key.Key_P:
            self.play_current()
        elif event.key() == Qt.Key.Key_T:
            self.do_transcribe()
        else:
            super().keyPressEvent(event)

    # ---- recording ----
    def toggle_recording(self):
        if not self.is_recording:
            self.audio_buffer = []
            self.is_recording = True
            self.record_btn.setText("Stop Recording (Space)")
            self.status.setText("Recording... press Space to stop.")
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16,
                                         callback=self._audio_cb)
            self.stream.start()
        else:
            self.is_recording = False
            self.record_btn.setText("Start Recording (Space)")
            self.stream.stop(); self.stream.close()
            if not self.audio_buffer:
                self.status.setText("No audio captured.")
                return
            out = Path(__file__).with_name("live_wavs_gui")
            out.mkdir(exist_ok=True)
            path = out / "last_recording.wav"
            data = np.concatenate(self.audio_buffer)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes(data.tobytes())
            self.current_wav = str(path)
            self.status.setText(f"Recorded {len(data) / SAMPLE_RATE:.2f}s - press T / Transcribe.")

    def _audio_cb(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_buffer.append(indata.copy())

    def play_current(self):
        if not self.current_wav:
            self.status.setText("Nothing to play.")
            return
        with wave.open(self.current_wav, "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sd.play(data, SAMPLE_RATE)

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load WAV", "", "WAV files (*.wav)")
        if path:
            self.current_wav = path
            self.status.setText(f"Loaded {Path(path).name} - press T / Transcribe.")

    # ---- transcribe ----
    def do_transcribe(self):
        if not self.current_wav:
            QMessageBox.warning(self, "No audio", "Record or load a WAV first.")
            return
        self.transcribe_btn.setEnabled(False)
        self.status.setText("Decoding...")
        self.worker = TranscribeWorker(self.current_wav, self.model_box.currentText())
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, text: str, meta: dict):
        self.transcribe_btn.setEnabled(True)
        self._last_text = text
        shown = text if text else "(nothing recognised)"
        self.result_label.setText(shown)
        self.status.setText(f"model={meta['model']}  decode={meta['seconds']}s")
        self.history.insertItem(0, f"[{meta['model']}] {Path(self.current_wav).name}: {shown}")
        self.score_label.setText("")

    def _on_failed(self, msg: str):
        self.transcribe_btn.setEnabled(True)
        self.status.setText("Error - see message box.")
        QMessageBox.critical(self, "Transcribe failed", msg)

    def score_current(self):
        ref = self.ref_edit.text().strip()
        if not ref or not hasattr(self, "_last_text"):
            return
        r = syllable_break(ref)
        h = syllable_break(self._last_text)
        er = edit_distance(r, h)
        ser = 100 * er / max(len(r), 1)
        self.score_label.setText(
            f"ref: {' '.join(r)}   |   hyp: {' '.join(h)}   |   SER = {ser:.0f}%  ({er}/{len(r)})"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ASRTester()
    win.show()
    sys.exit(app.exec())
