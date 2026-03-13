# gui.py  (PyQt6 + Modern QSS) - WATERMARK DETECTOR + PREVIEW
import os
import sys
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QTextCursor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QGroupBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSpinBox,
    QFormLayout, QLabel, QLineEdit
)

import config

try:
    import ocr_detector  # type: ignore[import]
except Exception as _e:
    ocr_detector = None  # type: ignore[assignment]
    print(f"[OCR] Moduł ocr_detector niedostępny: {_e}")

SUPPORTED_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

# ============================ QSS Themes ============================

_DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px;
}
QPushButton {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px;
    padding: 5px 14px; min-height: 28px;
}
QPushButton:hover  { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #2a2a3d; color: #6c7086; border-color: #313244; }
QPushButton#btn_start { background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#btn_start:hover { background-color: #89d18a; }
QPushButton#btn_start:disabled { background-color: #2a3b2a; color: #4a5a4a; }
QPushButton#btn_stop { background-color: #f38ba8; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#btn_stop:hover { background-color: #e07090; }
QPushButton#btn_stop:disabled { background-color: #3b2a2a; color: #5a4a4a; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 8px; padding: 4px; color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QTableWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; gridline-color: #313244;
}
QTableWidget::item:selected { background-color: #313244; color: #89b4fa; }
QHeaderView::section {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; padding: 4px; font-weight: bold;
}
QDoubleSpinBox, QSpinBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 2px;
}
QLineEdit {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 4px;
}
QTextEdit {
    background-color: #11111b; color: #a6e3a1;
    border: 1px solid #45475a; border-radius: 6px;
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace; font-size: 12px;
}
QProgressBar {
    background-color: #313244; border: 1px solid #45475a;
    border-radius: 5px; text-align: center; color: #cdd6f4; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #89b4fa,stop:1 #a6e3a1); border-radius: 4px;
}
QSplitter::handle { background-color: #45475a; }
QStatusBar { background-color: #181825; color: #6c7086; border-top: 1px solid #45475a; }
QLabel#preview_label, QLabel#zoom_label {
    background-color: #11111b; border: 1px solid #45475a; border-radius: 6px;
}
"""

_LIGHT_QSS = """
QMainWindow, QWidget {
    background-color: #eff1f5; color: #4c4f69;
    font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px;
}
QPushButton {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px;
    padding: 5px 14px; min-height: 28px;
}
QPushButton:hover { background-color: #dce0e8; border-color: #1e66f5; }
QPushButton:pressed { background-color: #ccd0da; }
QPushButton:disabled { background-color: #e6e9ef; color: #9ca0b0; border-color: #ccd0da; }
QPushButton#btn_start { background-color: #40a02b; color: #eff1f5; font-weight: bold; border: none; }
QPushButton#btn_start:hover { background-color: #379128; }
QPushButton#btn_start:disabled { background-color: #c8e6c0; color: #9abf93; }
QPushButton#btn_stop { background-color: #d20f39; color: #eff1f5; font-weight: bold; border: none; }
QPushButton#btn_stop:hover { background-color: #b50e33; }
QPushButton#btn_stop:disabled { background-color: #f5b8c5; color: #c08090; }
QGroupBox {
    border: 1px solid #bcc0cc; border-radius: 6px;
    margin-top: 8px; padding: 4px; color: #1e66f5; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
QTableWidget {
    background-color: #dce0e8; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px; gridline-color: #bcc0cc;
}
QTableWidget::item:selected { background-color: #c8d4f5; color: #1e66f5; }
QHeaderView::section {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; padding: 4px; font-weight: bold;
}
QDoubleSpinBox, QSpinBox {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 2px;
}
QLineEdit {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 4px;
}
QTextEdit {
    background-color: #e6e9ef; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 6px;
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace; font-size: 12px;
}
QProgressBar {
    background-color: #dce0e8; border: 1px solid #bcc0cc;
    border-radius: 5px; text-align: center; color: #4c4f69; min-height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e66f5,stop:1 #40a02b); border-radius: 4px;
}
QSplitter::handle { background-color: #bcc0cc; }
QStatusBar { background-color: #dce0e8; color: #9ca0b0; border-top: 1px solid #bcc0cc; }
QLabel#preview_label, QLabel#zoom_label {
    background-color: #e6e9ef; border: 1px solid #bcc0cc; border-radius: 6px;
}
"""

# ============================ Toggle Switch ============================

class ToggleSwitch(QtWidgets.QAbstractButton):
    """Prosty toggle switch jako zamiennik QCheckBox."""

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self._label = label
        self.setCheckable(True)
        self.setChecked(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self._label) + 8 if self._label else 0
        return QtCore.QSize(52 + text_w, 26)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        track_color = QtGui.QColor("#a6e3a1") if self.isChecked() else QtGui.QColor("#45475a")
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 3, 46, 20, 10, 10)

        thumb_x = 26 if self.isChecked() else 2
        p.setBrush(QtGui.QColor("#1e1e2e") if self.isChecked() else QtGui.QColor("#cdd6f4"))
        p.drawEllipse(thumb_x, 5, 18, 16)

        if self._label:
            p.setPen(QtGui.QColor("#cdd6f4"))
            p.setFont(self.font())
            p.drawText(52, 0, self.width() - 52, 26, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label)

        p.end()

    def isChecked(self) -> bool:
        return super().isChecked()


# ============================ helpers ============================

def is_supported_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


# ============================ Worker ============================

class WatermarkWorker(QtCore.QThread):
    progress = pyqtSignal(int, int)
    file_started = pyqtSignal(int, str, int)  # idx, name, total_frames
    file_finished = pyqtSignal(int, dict)
    log_line = pyqtSignal(str)
    frame_detected = pyqtSignal(np.ndarray, list)
    all_done = pyqtSignal()

    def __init__(self, files, confidence, sample_rate, output_dir, detailed_scan, parent=None):
        super().__init__(parent)
        self._files = files
        self._confidence = confidence
        self._sample_rate = sample_rate
        self._output_dir = output_dir
        self._detailed_scan = detailed_scan
        self._stop = False
        self._current_total_frames = 0

    def stop(self):
        self._stop = True

    def run(self):
        if ocr_detector is None:
            self.log_line.emit("[BŁĄD] Moduł ocr_detector nie został poprawnie załadowany.")
            self.all_done.emit()
            return

        original_base = getattr(config, "REPORTS_BASE_DIR", "reports")
        if self._output_dir:
            setattr(config, "REPORTS_BASE_DIR", self._output_dir)

        try:
            for idx, path in enumerate(self._files):
                if self._stop:
                    break

                fname = os.path.basename(path)

                # Sprawdź liczbę klatek (dla wideo)
                total_frames = 0
                cap = cv2.VideoCapture(os.path.abspath(path))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                self._current_total_frames = total_frames

                self.file_started.emit(idx, fname, total_frames)
                self.log_line.emit(f"[{idx+1}/{len(self._files)}] Rozpoczynam analizę: {fname} (Conf: {self._confidence}, Sample: {self._sample_rate}, Detailed: {self._detailed_scan})")

                def cb(curr, tot):
                    self.progress.emit(int(curr), int(tot))

                def preview_cb(*args):
                    if len(args) == 1:
                        self.frame_detected.emit(args[0], [])
                    elif len(args) >= 2:
                        self.frame_detected.emit(args[0], args[1])

                try:
                    res = ocr_detector.scan_for_watermarks(
                        path,
                        check_stop=lambda: self._stop,
                        progress_callback=cb,
                        confidence=self._confidence,
                        sample_rate=self._sample_rate,
                        detailed_scan=self._detailed_scan,
                        preview_callback=preview_cb
                    )
                    details = res if isinstance(res, dict) else {}
                    details["full_path"] = os.path.abspath(path)
                    details["total_frames"] = total_frames
                except Exception as e:
                    self.log_line.emit(f"[BŁĄD] {fname}: {e}")
                    details = {"status": "ERROR", "full_path": os.path.abspath(path), "error": str(e), "total_frames": total_frames}

                self.file_finished.emit(idx, details)
        finally:
            setattr(config, "REPORTS_BASE_DIR", original_base)

        self.all_done.emit()


# ============================ Drop Area ============================

class DropTableWidget(QTableWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if os.path.isdir(local):
                    for root, _, files in os.walk(local):
                        for f in files:
                            paths.append(os.path.join(root, f))
                else:
                    paths.append(local)
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ============================ GUI ============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watermark Detector (PyQt6) z Podglądem")
        self.resize(1400, 800)

        self.worker: Optional[WatermarkWorker] = None
        self.files: list[str] = []
        self.files_set: set[str] = set()
        self.report_paths: dict[int, str] = {}
        self.current_run_dir: Optional[str] = None
        # Przechowujemy dane per-plik: total_frames, detected_frames
        self._file_frame_data: dict[int, dict] = {}

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ---- LEFT PANEL ----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.setSpacing(6)
        left_layout.addLayout(top)

        self.btn_pick_files = QPushButton("📂 Dodaj pliki…")
        self.btn_pick_files.setToolTip("Dodaj pojedyncze pliki wideo lub obrazów do kolejki analizy")
        self.btn_pick_files.clicked.connect(self.pick_files)
        top.addWidget(self.btn_pick_files)

        self.btn_pick_folder = QPushButton("📁 Dodaj folder…")
        self.btn_pick_folder.setToolTip("Dodaj cały folder — wszystkie obsługiwane pliki zostaną dodane rekurencyjnie")
        self.btn_pick_folder.clicked.connect(self.pick_folder)
        top.addWidget(self.btn_pick_folder)

        self.grp_opts = QGroupBox("Parametry OCR (Watermark)")
        opts_lay = QVBoxLayout(self.grp_opts)
        opts_lay.setContentsMargins(8, 4, 8, 4)

        param_lay = QFormLayout()

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.60)
        self.spin_conf.setToolTip(
            "OCR Confidence — minimalna pewność rozpoznania tekstu (0.1–1.0).\n"
            "Niższa wartość = więcej detekcji (ale też fałszywe alarmy).\n"
            "Wyższa wartość = tylko bardzo pewne wyniki. Zalecane: 0.55–0.70."
        )

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 300)
        self.spin_sample.setValue(30)
        self.spin_sample.setToolTip(
            "Sample Rate — co ile klatek wideo jest analizowana jedna klatka.\n"
            "Wartość 30 = analizuj co 30. klatkę (1 raz na sekundę przy 30fps).\n"
            "Mniejsza wartość = dokładniej, ale wolniej. Zalecane: 15–30."
        )

        lbl_conf = QLabel("OCR Confidence:")
        lbl_conf.setToolTip(self.spin_conf.toolTip())
        lbl_sample = QLabel("Sample rate:")
        lbl_sample.setToolTip(self.spin_sample.toolTip())

        param_lay.addRow(lbl_conf, self.spin_conf)
        param_lay.addRow(lbl_sample, self.spin_sample)

        # Toggle switch zamiast checkboxa
        self.toggle_detailed = ToggleSwitch("Szczegółowa analiza (Dwufazowa)")
        self.toggle_detailed.setToolTip(
            "Tryb dwufazowy — po szybkim skanie uruchamia zaawansowane filtry obrazu\n"
            "(CLAHE, Top-Hat, gamma, odwrócone kolory) na klatkach bez detekcji.\n"
            "Wolniejszy, ale wykrywa trudne watermarki (np. białe na jasnym tle)."
        )
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.toggle_detailed)
        toggle_row.addStretch()
        param_lay.addRow("", toggle_row)

        dir_lay = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Domyślny folder z config.py")
        self.txt_output_dir.setReadOnly(True)
        self.btn_pick_out_dir = QPushButton("Zmień folder zapisu...")
        self.btn_pick_out_dir.clicked.connect(self.pick_output_dir)
        dir_lay.addWidget(self.txt_output_dir)
        dir_lay.addWidget(self.btn_pick_out_dir)

        opts_lay.addLayout(param_lay)
        opts_lay.addLayout(dir_lay)

        top.addWidget(self.grp_opts, 1)

        self.toggle_dark = ToggleSwitch("Ciemny")
        self.toggle_dark.setChecked(True)
        self.toggle_dark.toggled.connect(self._apply_theme)
        top.addWidget(self.toggle_dark)

        self.btn_start = QPushButton("▶ START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_analysis)
        top.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        top.addWidget(self.btn_stop)

        splitter = QSplitter(Qt.Orientation.Vertical)
        left_layout.addWidget(splitter, 1)

        # Tabela z drag&drop i nowymi kolumnami
        self.table_results = DropTableWidget()
        self.table_results.files_dropped.connect(self._add_files)
        self.table_results.setColumnCount(5)
        self.table_results.setHorizontalHeaderLabels([
            "Plik", "Typ", "Status AI", "% AI w wideo", "Raport CSV"
        ])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setAlternatingRowColors(True)
        self.table_results.setToolTip("Przeciągnij i upuść pliki lub foldery tutaj (Drag & Drop)")
        splitter.addWidget(self.table_results)

        self.logView = QTextEdit()
        self.logView.setReadOnly(True)
        splitter.addWidget(self.logView)
        splitter.setSizes([350, 250])

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        left_layout.addLayout(bottom)

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("%p%")
        bottom.addWidget(self.progressBar, 1)

        self.btn_open_folder = QPushButton("📂 Otwórz folder wyników")
        self.btn_open_folder.setToolTip(
            "Otwiera folder z zapisanymi klatkami i raportami CSV z ostatniej analizy.\n"
            "Każdy plik ma własny podfolder z nazwą i znacznikiem czasu."
        )
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        bottom.addWidget(self.btn_open_folder)

        root.addWidget(left_panel, 2)

        # ---- RIGHT PANEL ----
        right_panel = QSplitter(Qt.Orientation.Vertical)

        preview_container = QWidget()
        preview_lay = QVBoxLayout(preview_container)
        preview_lay.setContentsMargins(0, 0, 0, 0)
        lbl_title = QLabel("<b>Klatka z kamery:</b>")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_lay.addWidget(lbl_title)
        self.lbl_preview = QLabel("Brak podglądu")
        self.lbl_preview.setObjectName("preview_label")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(400, 300)
        self.lbl_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        preview_lay.addWidget(self.lbl_preview, 1)
        right_panel.addWidget(preview_container)

        zoom_container = QWidget()
        zoom_lay = QVBoxLayout(zoom_container)
        zoom_lay.setContentsMargins(0, 10, 0, 0)
        lbl_zoom = QLabel("<b>Przybliżenie detekcji (Zoom):</b>")
        lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_lay.addWidget(lbl_zoom)
        self.lbl_zoom = QLabel("Brak detekcji")
        self.lbl_zoom.setObjectName("zoom_label")
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_zoom.setMinimumSize(400, 150)
        self.lbl_zoom.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        zoom_lay.addWidget(self.lbl_zoom, 1)
        right_panel.addWidget(zoom_container)

        right_panel.setSizes([450, 150])
        root.addWidget(right_panel, 1)

        self.status = self.statusBar()
        self._apply_theme(True)

    # -------------------- Theme --------------------

    def _apply_theme(self, dark: bool) -> None:
        app = QApplication.instance()
        if not app:
            return
        app.setStyle("Fusion")  # type: ignore[union-attr]
        app.setStyleSheet(_DARK_QSS if dark else _LIGHT_QSS)  # type: ignore[union-attr]

    # -------------------- Helpers --------------------

    def append_log(self, text: str) -> None:
        self.logView.append(text)
        self.logView.moveCursor(QTextCursor.MoveOperation.End)
        self.status.showMessage(text, 4000)

    def _add_files(self, paths: list) -> None:
        added = 0
        for p in paths:
            if not p or not is_supported_file(p):
                continue
            ap = os.path.abspath(p)
            if ap in self.files_set:
                continue
            self.files_set.add(ap)
            self.files.append(ap)

            row = self.table_results.rowCount()
            self.table_results.insertRow(row)

            fname = os.path.basename(ap)
            ext = os.path.splitext(fname)[1].lower()
            file_type = "Video" if ext in {".mp4", ".mkv", ".avi", ".webm", ".mov"} else "Image"

            self.table_results.setItem(row, 0, QTableWidgetItem(fname))
            self.table_results.setItem(row, 1, QTableWidgetItem(file_type))

            status_item = QTableWidgetItem("⏳ Oczekuje")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_results.setItem(row, 2, status_item)

            pct_item = QTableWidgetItem("-")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_results.setItem(row, 3, pct_item)

            self.table_results.setItem(row, 4, QTableWidgetItem("-"))

            added += 1

        if added:
            self.btn_start.setEnabled(True)
            self.append_log(f"> Dodano {added} plik(ów). Razem: {len(self.files)}.")

    @pyqtSlot(np.ndarray, list)
    def set_preview_image(self, frame_bgr: np.ndarray, detections: list) -> None:
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(
            self.lbl_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_preview.setPixmap(scaled_pixmap)

        if detections:
            det = detections[0]
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 10, 10))
            padding = 30
            crop_bgr = frame_bgr[
                max(0, y1 - padding):min(h, y2 + padding),
                max(0, x1 - padding):min(w, x2 + padding)
            ]
            if crop_bgr.size > 0:
                ch_c, cw_c, c_c = crop_bgr.shape
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crop_qt = QImage(crop_rgb.data, cw_c, ch_c, c_c * cw_c, QImage.Format.Format_RGB888)
                self.lbl_zoom.setPixmap(
                    QPixmap.fromImage(crop_qt).scaled(
                        self.lbl_zoom.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
        else:
            self.lbl_zoom.setText("Brak detekcji")

    # -------------------- Actions --------------------

    def pick_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder docelowy dla raportów")
        if folder:
            self.txt_output_dir.setText(os.path.abspath(folder))

    def pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz pliki do analizy", "",
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;Wszystkie pliki (*.*)",
        )
        if paths:
            self._add_files(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Wybierz folder z nagraniami/obrazami", "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not folder:
            return
        to_add = [
            os.path.join(root, name)
            for root, _, files in os.walk(folder)
            for name in files
            if is_supported_file(name)
        ]
        if not to_add:
            self.append_log("> W wybranym folderze nie znaleziono obsługiwanych plików.")
            return
        self._add_files(sorted(to_add))

    def start_analysis(self) -> None:
        if not self.files:
            QMessageBox.warning(self, "Brak plików", "Najpierw dodaj pliki lub folder do analizy.")
            return

        self._file_frame_data = {}

        for btn in (self.btn_start, self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progressBar.setValue(0)
        self.lbl_preview.setText("Analiza w toku...")
        self.lbl_zoom.setText("Analiza w toku...")

        conf_val = self.spin_conf.value()
        sample_val = self.spin_sample.value()
        out_dir = self.txt_output_dir.text().strip()
        detailed_val = self.toggle_detailed.isChecked()

        self.worker = WatermarkWorker(self.files, conf_val, sample_val, out_dir, detailed_val, parent=self)
        self.worker.progress.connect(self.on_progress)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.log_line.connect(self.append_log)
        self.worker.frame_detected.connect(self.set_preview_image)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def stop_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("> Przerywam analizę…")

    def open_output_folder(self) -> None:
        if not self.current_run_dir or not os.path.isdir(self.current_run_dir):
            QMessageBox.information(self, "Brak Outputu", "Najpierw przeprowadź analizę.")
            return
        path_to_open = os.path.abspath(self.current_run_dir)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path_to_open)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                QtCore.QProcess.startDetached("open", [path_to_open])
            else:
                QtCore.QProcess.startDetached("xdg-open", [path_to_open])
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć: {e}")

    # -------------------- Slots --------------------

    @pyqtSlot(int, int)
    def on_progress(self, curr: int, tot: int) -> None:
        if tot > 0:
            self.progressBar.setValue(max(0, min(100, int(curr * 100 / max(1, tot)))))

    @pyqtSlot(int, str, int)
    def on_file_started(self, idx: int, name: str, total_frames: int) -> None:
        self.progressBar.setValue(0)
        self._file_frame_data[idx] = {"total_frames": total_frames, "detected_frames": 0}
        # Ustaw status na "w trakcie"
        status_item = self.table_results.item(idx, 2)
        if status_item:
            status_item.setText("🔍 Analiza...")

    @pyqtSlot(int, dict)
    def on_file_finished(self, idx: int, details: dict) -> None:
        folder = details.get("watermark_folder")
        if folder:
            self.report_paths[idx] = folder
            self.current_run_dir = os.path.dirname(folder)
            self.btn_open_folder.setEnabled(True)

        count = details.get("watermark_count", 0) or 0
        total_frames = details.get("total_frames", 0) or 0
        sample_rate = self.spin_sample.value()

        # Szacunkowa liczba przeanalizowanych klatek
        sampled_frames = max(1, total_frames // sample_rate) if total_frames > 0 else 1

        # --- Kolumna Status AI ---
        if count > 0:
            types = ", ".join(details.get("watermark_types", []))
            status_text = f"🔴 AI DETECTED\n{types}"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QtGui.QBrush(QtGui.QColor("#f38ba8")))
        else:
            status_item = QTableWidgetItem("✅ AI CLEAR")
            status_item.setForeground(QtGui.QBrush(QtGui.QColor("#a6e3a1")))
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_results.setItem(idx, 2, status_item)

        # --- Kolumna % AI w wideo ---
        if total_frames > 0 and count > 0:
            # count = liczba klatek z detekcją, sampled_frames = ile przeskanowano
            ai_pct = min(100.0, count / sampled_frames * 100)
            clear_pct = 100.0 - ai_pct
            pct_text = f"🔴 {ai_pct:.0f}% AI  |  ✅ {clear_pct:.0f}% CLEAR"
            pct_item = QTableWidgetItem(pct_text)
            if ai_pct >= 50:
                pct_item.setForeground(QtGui.QBrush(QtGui.QColor("#f38ba8")))
            else:
                pct_item.setForeground(QtGui.QBrush(QtGui.QColor("#fab387")))
        else:
            pct_item = QTableWidgetItem("✅ 100% CLEAR")
            pct_item.setForeground(QtGui.QBrush(QtGui.QColor("#a6e3a1")))
        pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_results.setItem(idx, 3, pct_item)

        # --- Kolumna CSV ---
        report_file = details.get("csv_path", folder if folder else "Brak")
        self.table_results.setItem(idx, 4, QTableWidgetItem(str(report_file)))

        self.append_log(f"   ➔ Plik zakończony. Znaleziono {count} watermarków.")
        self.table_results.resizeRowsToContents()

    @pyqtSlot()
    def on_all_done(self) -> None:
        self.append_log("> Analiza wszystkich plików zakończona.")
        self.progressBar.setValue(0)
        for btn in (self.btn_pick_files, self.btn_pick_folder):
            btn.setEnabled(True)
        self.btn_start.setEnabled(len(self.files) > 0)
        self.btn_stop.setEnabled(False)
        self.worker = None


# ============================ entry point ============================

def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
