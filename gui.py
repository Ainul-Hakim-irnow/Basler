import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
QApplication, QWidget, QLabel, QPushButton, QSlider, QLineEdit,
QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QSizePolicy
)

from pypylon import pylon

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    error       = Signal(str)
    defaults_loaded = Signal(float, float, float)  # exposure, brightness, contrast


def __init__(self, parent=None):
    super().__init__(parent)
    self._running = False


    # live‑updatable parameters
    self.exposure   = 1000          # µs
    self.brightness = 0.0           # ‑1 … 1
    self.contrast   = 0.0           # ‑1 … 1


def update_params(self, exposure, brightness, contrast):
    self.exposure, self.brightness, self.contrast = exposure, brightness, contrast


def load_defaults(self):
    if not hasattr(self, "_cam"):
        return
    try:
        self._cam.UserSetSelector.SetValue("Default")
        self._cam.UserSetLoad.Execute()
        self.exposure   = self._cam.ExposureTime.GetValue()
        self.brightness = self._cam.BslBrightness.GetValue()
        self.contrast   = self._cam.BslContrast.GetValue()
        self.defaults_loaded.emit(self.exposure, self.brightness, self.contrast)
    except Exception as e:
        self.error.emit(str(e))


def stop(self):
    self._running = False


# ---------------- QThread.run() ---------------- #
def run(self):
    try:
        self._cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self._cam.Open()
        self._cam.AcquisitionFrameRateEnable.SetValue(True)
        self._cam.AcquisitionFrameRate.SetValue(30)


        fmt = pylon.ImageFormatConverter()
        fmt.OutputPixelFormat   = pylon.PixelType_BGR8packed
        fmt.OutputBitAlignment  = pylon.OutputBitAlignment_MsbAligned


        self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._running = True


        while self._running and self._cam.IsGrabbing():
            grab = self._cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                # apply latest parameters (they're cheap to set)
                try:
                    self._cam.ExposureTime.SetValue(self.exposure)
                    self._cam.BslBrightness.SetValue(self.brightness)
                    self._cam.BslContrast.SetValue(self.contrast)
                except Exception as e:
                    # ignore momentary failures (e.g. while stopping)
                    pass


                frame = fmt.Convert(grab).GetArray()
                self.frame_ready.emit(frame)
            grab.Release()


    except Exception as e:
        self.error.emit(str(e))
    finally:
        if hasattr(self, "_cam") and self._cam.IsGrabbing():
            self._cam.StopGrabbing()
        self._cam.Close()




# ------------------------------ Main window ------------------------------ #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real‑Time Image Format Control")
        self.cam_thread: CameraThread | None = None


        self._build_ui()


    # ---------- UI construction ---------- #
    def _build_ui(self):
        # --- video area (right) ---
        self.video_lbl = QLabel("Camera feed")
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background-color:black;")
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # --- controls (left) ---
        ctrl_layout = QFormLayout()


        # Exposure
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(2, 1_000_000)
        self.exp_slider.setValue(1000)
        self.exp_edit   = QLineEdit("1000")
        self.exp_edit.setFixedWidth(70)
        ctrl_layout.addRow("Exposure (µs)",
                            self._hbox(self.exp_slider, self.exp_edit))


        # Brightness  (slider uses ‑1000 … 1000 to represent ‑1 … 1)
        self.bri_slider = QSlider(Qt.Horizontal)
        self.bri_slider.setRange(-1000, 1000)
        self.bri_slider.setValue(0)
        self.bri_edit   = QLineEdit("0.000")
        self.bri_edit.setFixedWidth(70)
        ctrl_layout.addRow("Brightness",
                            self._hbox(self.bri_slider, self.bri_edit))


        # Contrast
        self.con_slider = QSlider(Qt.Horizontal)
        self.con_slider.setRange(-1000, 1000)
        self.con_slider.setValue(0)
        self.con_edit   = QLineEdit("0.000")
        self.con_edit.setFixedWidth(70)
        ctrl_layout.addRow("Contrast",
                            self._hbox(self.con_slider, self.con_edit))


        # Buttons
        self.start_btn   = QPushButton("Start Camera")
        self.stop_btn    = QPushButton("Stop Camera")
        self.default_btn = QPushButton("Default Settings")
        self.stop_btn.setEnabled(False)


        btn_col = QVBoxLayout()
        btn_col.addWidget(self.start_btn)
        btn_col.addWidget(self.stop_btn)
        btn_col.addWidget(self.default_btn)
        btn_col.addStretch()


        left_col = QVBoxLayout()
        left_col.addLayout(ctrl_layout)
        left_col.addLayout(btn_col)


        # --- wire up signals ---
        self.exp_slider.valueChanged.connect(
            lambda v: self.exp_edit.setText(str(v)))
        self.exp_edit.editingFinished.connect(
            lambda: self._sync_edit_to_slider(self.exp_edit, self.exp_slider, 2, 1_000_000, int))


        self.bri_slider.valueChanged.connect(
            lambda v: self.bri_edit.setText(f"{v/1000:.3f}"))
        self.bri_edit.editingFinished.connect(
            lambda: self._sync_edit_to_slider(self.bri_edit, self.bri_slider, -1.0, 1.0, float, 1000))


        self.con_slider.valueChanged.connect(
            lambda v: self.con_edit.setText(f"{v/1000:.3f}"))
        self.con_edit.editingFinished.connect(
            lambda: self._sync_edit_to_slider(self.con_edit, self.con_slider, -1.0, 1.0, float, 1000))


        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.default_btn.clicked.connect(self.load_defaults)


        # --- main layout ---
        root_layout = QHBoxLayout(self)
        root_layout.addLayout(left_col, 0)   # minimal width
        root_layout.addWidget(self.video_lbl, 1)


    # ---------- helpers ---------- #
    @staticmethod
    def _hbox(*widgets):
        lay = QHBoxLayout()
        for w in widgets:
            lay.addWidget(w)
        return lay


    def _sync_edit_to_slider(self, edit: QLineEdit, slider: QSlider,
                                lo, hi, typ, scale=1):
        """Validate edit text and push value to slider (scale converts float->int)."""
        try:
            val = typ(edit.text())
            if not (lo <= val <= hi):
                raise ValueError
            slider.setValue(int(val * scale))
        except ValueError:
            QMessageBox.warning(self, "Invalid value",
                                f"Value must be between {lo} and {hi}")
            # revert to slider's current value
            if scale == 1:
                edit.setText(str(slider.value()))
            else:
                edit.setText(f"{slider.value()/scale:.3f}")


    # ---------- camera control ---------- #
    def start_camera(self):
        if self.cam_thread:   # already running
            return
        self.cam_thread = CameraThread()
        self.cam_thread.frame_ready.connect(self.update_frame)
        self.cam_thread.error.connect(self._show_error)
        self.cam_thread.defaults_loaded.connect(self._apply_defaults_from_cam)
        self.cam_thread.start()


        self._push_params_to_thread()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)


        # update params to thread every 100 ms
        self.param_timer = QTimer(self)
        self.param_timer.timeout.connect(self._push_params_to_thread)
        self.param_timer.start(100)


    def stop_camera(self):
        if not self.cam_thread:
            return
        self.param_timer.stop()
        self.cam_thread.stop()
        self.cam_thread.wait()
        self.cam_thread = None


        self.video_lbl.setPixmap(QPixmap())
        self.video_lbl.setText("Camera feed")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


    def _push_params_to_thread(self):
        if self.cam_thread:
            exposure   = self.exp_slider.value()
            brightness = self.bri_slider.value() / 1000
            contrast   = self.con_slider.value() / 1000
            self.cam_thread.update_params(exposure, brightness, contrast)


    def load_defaults(self):
        if self.cam_thread:
            self.cam_thread.load_defaults()


    def _apply_defaults_from_cam(self, exposure, brightness, contrast):
        self.exp_slider.setValue(int(exposure))
        self.bri_slider.setValue(int(brightness * 1000))
        self.con_slider.setValue(int(contrast * 1000))


    # ---------- display frames ---------- #
    def update_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)


        # scale to label size while keeping aspect ratio
        self.video_lbl.setPixmap(
            pix.scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.video_lbl.pixmap():
            self.video_lbl.setPixmap(
                self.video_lbl.pixmap().scaled(
                    self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


    # ---------- misc ---------- #
    def _show_error(self, msg):
        QMessageBox.critical(self, "Camera error", msg)
        self.stop_camera()


    def closeEvent(self, ev):
        self.stop_camera()
        ev.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()   # start full‑screen
    sys.exit(app.exec())

