import sys
import cv2
from pypylon import pylon

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QLineEdit,
    QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QSizePolicy,
    QMainWindow, QTabWidget, QCheckBox, QGridLayout
)
from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtGui import QPixmap, QImage
import numpy as np

class ControlPanel(QWidget):
    params_changed = Signal(float, float, float) # exp, bri, con
    contrast_mode_changed = Signal(bool)      # use_scurve
    load_defaults_requested = Signal()

    def __init__(self):
        super().__init__()
        
        main_layout = QVBoxLayout(self)
        params_form = QFormLayout()
        
        self.exp_slider = QSlider(Qt.Orientation.Horizontal); self.exp_slider.setRange(2, 50_000); self.exp_slider.setValue(1000)
        self.exp_edit = QLineEdit("1000")
        params_form.addRow("Exposure (µs)", self._create_hbox(self.exp_slider, self.exp_edit))
        
        self.bri_slider = QSlider(Qt.Orientation.Horizontal); self.bri_slider.setRange(-1000, 1000); self.bri_slider.setValue(0)
        self.bri_edit = QLineEdit("0.000")
        params_form.addRow("Brightness", self._create_hbox(self.bri_slider, self.bri_edit))

        self.con_slider = QSlider(Qt.Orientation.Horizontal); self.con_slider.setRange(-1000, 1000); self.con_slider.setValue(0)
        self.con_edit = QLineEdit("0.000")
        params_form.addRow("Contrast", self._create_hbox(self.con_slider, self.con_edit))
        
        self.use_scurve_checkbox = QCheckBox("Use S-Curve Contrast")
        params_form.addRow("", self.use_scurve_checkbox)

        main_layout.addLayout(params_form)

        self.defaults_btn = QPushButton("Load Defaults")
        main_layout.addWidget(self.defaults_btn)
        main_layout.addStretch()

        self._connect_signals()

    def _connect_signals(self):
        self.exp_slider.valueChanged.connect(self._emit_param_change)
        self.bri_slider.valueChanged.connect(self._emit_param_change)
        self.con_slider.valueChanged.connect(self._emit_param_change)
        self.use_scurve_checkbox.stateChanged.connect(lambda: self.contrast_mode_changed.emit(self.use_scurve_checkbox.isChecked()))
        self.defaults_btn.clicked.connect(self.load_defaults_requested.emit)
        
        self._connect_slider_edit(self.exp_slider, self.exp_edit, is_float=False)
        self._connect_slider_edit(self.bri_slider, self.bri_edit, factor=1000.0)
        self._connect_slider_edit(self.con_slider, self.con_edit, factor=1000.0)

    def _emit_param_change(self):
        self.params_changed.emit(self.exp_slider.value(), self.bri_slider.value() / 1000.0, self.con_slider.value() / 1000.0)

    def set_to_defaults(self):
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: w.blockSignals(True)
        self.exp_slider.setValue(1000); self.bri_slider.setValue(0); self.con_slider.setValue(0); self.use_scurve_checkbox.setChecked(False)
        self.exp_edit.setText("1000"); self.bri_edit.setText("0.000"); self.con_edit.setText("0.000")
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: w.blockSignals(False)
        self._emit_param_change()
        self.contrast_mode_changed.emit(False)

    def _create_hbox(self, w1, w2): layout = QHBoxLayout(); layout.addWidget(w1); w2.setFixedWidth(70); layout.addWidget(w2); return layout
    def _connect_slider_edit(self, slider, edit, is_float=True, factor=1.0):
        slider.valueChanged.connect(lambda val: edit.setText(f"{val / factor:.3f}" if is_float else str(val)))
        edit.editingFinished.connect(lambda: slider.setValue(int(float(edit.text()) * factor) if is_float else int(edit.text())))
        edit.editingFinished.connect(self._emit_param_change)

class Camera_Thread(QThread):
    frame_ready = Signal(str, np.ndarray)
    error = Signal(str, str)

    def __init__(self, device_info):
        super().__init__()
        self.device_info = device_info
        self.serial = self.device_info.GetSerialNumber()
        self._running = False
        self.exposure, self.brightness, self.contrast = 1000.0, 0.0, 0.0
        self.use_scurve = False

    def stop(self): self._running = False
    def update_params(self, exp, bri, con): self.exposure, self.brightness, self.contrast = exp, bri, con
    def update_contrast_mode(self, use_scurve): self.use_scurve = use_scurve

    def run(self):
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info))
            cam.Open()
            cam.AcquisitionFrameRateEnable.SetValue(True); cam.AcquisitionFrameRate.SetValue(30)
            fmt = pylon.ImageFormatConverter(); fmt.OutputPixelFormat = pylon.PixelType_BGR8packed
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly); self._running = True

            while self._running and cam.IsGrabbing():
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    try:
                        cam.BslContrastMode.SetValue("SCurve" if self.use_scurve else "Linear")
                        cam.ExposureTime.SetValue(self.exposure)
                        cam.BslBrightness.SetValue(self.brightness)
                        cam.BslContrast.SetValue(self.contrast)
                    except Exception: pass
                    self.frame_ready.emit(self.serial, fmt.Convert(grab).GetArray())
                grab.Release()
        except Exception as e: self.error.emit(self.serial, str(e))
        finally:
            if 'cam' in locals() and cam.IsGrabbing(): cam.StopGrabbing()
            if 'cam' in locals() and cam.IsOpen(): cam.Close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera GUI")
        self.camera_threads = {}
        self.video_panels = {}
        self.control_panels = {}

        self.detect_and_setup_ui()

    def detect_and_setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # MODIFIED: Main layout is now horizontal
        main_layout = QHBoxLayout(central_widget)

        # --- Left Section: Tabbed Controls ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        control_tabs_widget = QTabWidget()
        left_layout.addWidget(control_tabs_widget)
        
        # --- Right Section: Video Feeds Grid ---
        video_grid_widget = QWidget()
        video_grid_layout = QGridLayout(video_grid_widget)
        
        # Add widgets to main layout
        main_layout.addWidget(left_widget, 35) # 35% of horizontal space
        main_layout.addWidget(video_grid_widget, 65) # 65% of horizontal space

        # --- Camera Detection and UI Population ---
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices:
            video_grid_layout.addWidget(QLabel("No Basler cameras detected."))
        else:
            positions = [(i, j) for i in range(2) for j in range(2)] # 2x2 grid
            for i, dev in enumerate(devices):
                if i >= 4: break # Limit to 4 cameras
                serial = dev.GetSerialNumber()

                # -- Create Video Panel (Right Side) --
                panel_container = QWidget()
                panel_layout = QVBoxLayout(panel_container)
                
                title_lbl = QLabel(f"<b>Camera {i+1}</b> ({serial})")
                title_lbl.setAlignment(Qt.AlignCenter)
                
                video_lbl = QLabel("Disconnected")
                video_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                video_lbl.setAlignment(Qt.AlignCenter)
                video_lbl.setStyleSheet("background-color:black; color:white;")
                
                panel_layout.addWidget(title_lbl)
                panel_layout.addWidget(video_lbl)
                row, col = positions[i]
                video_grid_layout.addWidget(panel_container, row, col)
                self.video_panels[serial] = video_lbl

                # -- Create Control Tab (Left Side) --
                control_panel = ControlPanel()
                self.control_panels[serial] = control_panel
                
                side_tabs = QTabWidget()
                side_tabs.setTabPosition(QTabWidget.TabPosition.West)
                side_tabs.addTab(control_panel, "Image")
                
                model_tab_placeholder = QLabel("Model Inference Placeholder")
                model_tab_placeholder.setAlignment(Qt.AlignCenter)
                side_tabs.addTab(model_tab_placeholder, "Model")
                
                control_tabs_widget.addTab(side_tabs, f"Cam {i+1}")

                # -- Connect Signals for this specific camera --
                control_panel.params_changed.connect(
                    lambda exp, bri, con, s=serial: self.push_params_to_thread(s, exp, bri, con)
                )
                control_panel.contrast_mode_changed.connect(
                    lambda use_scurve, s=serial: self.push_contrast_mode_to_thread(s, use_scurve)
                )
                control_panel.load_defaults_requested.connect(control_panel.set_to_defaults)
                
        # --- Global Control Buttons (Bottom Left) ---
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start All Cameras")
        self.stop_btn = QPushButton("Stop All Cameras")
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        left_layout.addLayout(buttons_layout)
        
        if not devices:
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        else:
            self.start_btn.clicked.connect(self.start_all_cameras)
            self.stop_btn.clicked.connect(self.stop_all_cameras)
            self.stop_btn.setEnabled(False)
            for panel in self.control_panels.values(): panel.setEnabled(False)

    def start_all_cameras(self):
        if self.camera_threads: return
        
        devices = {dev.GetSerialNumber(): dev for dev in pylon.TlFactory.GetInstance().EnumerateDevices()}

        for serial in self.control_panels.keys():
            if serial in devices:
                thread = Camera_Thread(devices[serial])
                thread.frame_ready.connect(self.update_frame)
                thread.error.connect(self.show_error)
                thread.start()
                self.camera_threads[serial] = thread
        
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        for panel in self.control_panels.values(): panel.setEnabled(True)

    def stop_all_cameras(self):
        for thread in self.camera_threads.values():
            thread.stop(); thread.wait()
        self.camera_threads.clear()

        for panel in self.video_panels.values():
            panel.setPixmap(QPixmap())
            panel.setText("Disconnected")

        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        for panel in self.control_panels.values(): panel.setEnabled(False)
    
    def update_frame(self, serial, frame):
        if serial in self.video_panels:
            video_lbl = self.video_panels[serial]
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            video_lbl.setPixmap(pixmap.scaled(video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def push_params_to_thread(self, serial, exp, bri, con):
        if serial in self.camera_threads: self.camera_threads[serial].update_params(exp, bri, con)
    def push_contrast_mode_to_thread(self, serial, use_scurve):
        if serial in self.camera_threads: self.camera_threads[serial].update_contrast_mode(use_scurve)
    def show_error(self, serial, message): QMessageBox.warning(self, f"Camera Error ({serial})", message)
    def closeEvent(self, event): self.stop_all_cameras(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())