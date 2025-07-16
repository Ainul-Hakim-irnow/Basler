import sys
import os
import cv2
import csv
import json
import time
import torch
import numpy as np
import torchvision
from pypylon import pylon
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QLineEdit, QComboBox,
    QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QSizePolicy, QGroupBox,
    QMainWindow, QTabWidget, QCheckBox, QGridLayout, QFileDialog, QListWidget
)
from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtGui import QPixmap, QImage

from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# --- IMPORTANT: DEFINE YOUR PYTORCH MODEL ARCHITECTURE HERE ---
def get_my_maskrcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
# --- END OF MODEL DEFINITION ---


# --- Helper Functions ---
def overlay_masks_boxes_labels_predict(image, masks, boxes, class_colors, labels, alpha=0.3):
	overlay = image.convert("RGBA")
	for mask, label in zip(masks, labels):
		mask_array = mask.cpu().numpy().astype("uint8") * 255
		mask_pil = Image.fromarray(mask_array)
		class_name = label.split()[0] if (label and isinstance(label, str)) else "unknown"
		color = tuple(class_colors.get(class_name, (255, 255, 255))) + (int(255 * alpha),)
		colored_mask = Image.new("RGBA", overlay.size, color); mask_image = Image.new("RGBA", overlay.size)
		mask_image.paste(colored_mask, (0, 0), mask_pil); overlay = Image.alpha_composite(overlay, mask_image)
	overlay = overlay.convert("RGB"); draw = ImageDraw.Draw(overlay)
	image_width, image_height = image.size
	dynamic_font_size = max(20, min(image_width, image_height) // 50)
	try: font = ImageFont.load_default(dynamic_font_size)
	except IOError: font = ImageFont.load_default()
	for box, label in zip(boxes, labels):
		x_min, y_min, x_max, y_max = box; class_name = label.split()[0]; color = tuple(class_colors.get(class_name, (255, 255, 255)))
		draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=5)
		text_size = draw.textbbox((0, 0), label, font=font); label_width = text_size[2] - text_size[0]; label_height = text_size[3] - text_size[1]
		if y_min - label_height - 12 < 0:
			label_position = (x_min, y_max + 12); label_bg_position = [(x_min, y_max + 12), (x_min + label_width + 12, y_max + label_height + 16)]
		else:
			label_position = (x_min + 4, y_min - label_height - 12); label_bg_position = [(x_min, y_min - label_height - 12), (x_min + label_width + 12, y_min)]
		draw.rectangle(label_bg_position, fill=color); draw.text(label_position, label, fill="black", font=font)
	return overlay

def resize_and_binarize_masks(masks, image_size):
	import torch.nn.functional as torch_F; resized_masks = []
	for mask in masks:
		if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
		elif mask.dim() == 3 and mask.size(0) == 1: mask = mask.unsqueeze(0)
		resized_mask = torch_F.interpolate(mask, size=image_size, mode="bilinear", align_corners=False)
		binarized_mask = (resized_mask > 0.5).float().squeeze(0).squeeze(0)
		resized_masks.append(binarized_mask.byte())
	return resized_masks

def filter_overlapping_detections(boxes, scores, labels, masks, threshold=0.25):
    if len(boxes) == 0: return torch.empty(0, 4), torch.empty(0), [], []
    keep = []; indices = torch.argsort(scores, descending=True)
    while len(indices) > 0:
        current_idx = indices[0].item(); keep.append(current_idx)
        if len(indices) == 1: break
        current_box = boxes[current_idx:current_idx+1]; other_indices = indices[1:]; other_boxes = boxes[other_indices]
        iou = torchvision.ops.box_iou(current_box, other_boxes)[0]
        current_label_val = labels[current_idx]; other_labels_vals = [labels[i.item()] for i in other_indices]
        mask = torch.tensor([ (current_label_val != other_label) or (iou_val <= threshold) for other_label, iou_val in zip(other_labels_vals, iou)], dtype=torch.bool)
        indices = other_indices[mask]
    keep = torch.tensor(keep, dtype=torch.long)
    return boxes[keep], scores[keep], [labels[i] for i in keep], [masks[i] for i in keep]


# --- PySide6 UI Classes ---
class ControlPanel(QWidget):
    params_changed = Signal(float, float, float); contrast_mode_changed = Signal(bool); load_defaults_requested = Signal()
    def __init__(self):
        super().__init__(); main_layout = QVBoxLayout(self); params_form = QFormLayout()
        self.exp_slider = QSlider(Qt.Orientation.Horizontal); self.exp_slider.setRange(2, 50_000); self.exp_slider.setValue(1000)
        self.exp_edit = QLineEdit("1000"); params_form.addRow("Exposure (µs)", self._create_hbox(self.exp_slider, self.exp_edit))
        self.bri_slider = QSlider(Qt.Orientation.Horizontal); self.bri_slider.setRange(-1000, 1000); self.bri_slider.setValue(0)
        self.bri_edit = QLineEdit("0.000"); params_form.addRow("Brightness", self._create_hbox(self.bri_slider, self.bri_edit))
        self.con_slider = QSlider(Qt.Orientation.Horizontal); self.con_slider.setRange(-1000, 1000); self.con_slider.setValue(0)
        self.con_edit = QLineEdit("0.000"); params_form.addRow("Contrast", self._create_hbox(self.con_slider, self.con_edit))
        self.use_scurve_checkbox = QCheckBox("Use S-Curve Contrast"); params_form.addRow("", self.use_scurve_checkbox)
        main_layout.addLayout(params_form); self.defaults_btn = QPushButton("Load Defaults"); main_layout.addWidget(self.defaults_btn); main_layout.addStretch(); self._connect_signals()
    def _connect_signals(self):
        self.exp_slider.valueChanged.connect(self._emit_param_change); self.bri_slider.valueChanged.connect(self._emit_param_change); self.con_slider.valueChanged.connect(self._emit_param_change)
        self.use_scurve_checkbox.stateChanged.connect(lambda: self.contrast_mode_changed.emit(self.use_scurve_checkbox.isChecked())); self.defaults_btn.clicked.connect(self.load_defaults_requested.emit)
        self._connect_slider_edit(self.exp_slider, self.exp_edit, is_float=False); self._connect_slider_edit(self.bri_slider, self.bri_edit, factor=1000.0); self._connect_slider_edit(self.con_slider, self.con_edit, factor=1000.0)
    def _emit_param_change(self): self.params_changed.emit(self.exp_slider.value(), self.bri_slider.value() / 1000.0, self.con_slider.value() / 1000.0)
    def set_to_defaults(self):
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: w.blockSignals(True)
        self.exp_slider.setValue(1000); self.bri_slider.setValue(0); self.con_slider.setValue(0); self.use_scurve_checkbox.setChecked(False)
        self.exp_edit.setText("1000"); self.bri_edit.setText("0.000"); self.con_edit.setText("0.000")
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: w.blockSignals(False)
        self._emit_param_change(); self.contrast_mode_changed.emit(False)
    def _create_hbox(self, w1, w2): layout = QHBoxLayout(); layout.addWidget(w1); w2.setFixedWidth(70); layout.addWidget(w2); return layout
    def _connect_slider_edit(self, slider, edit, is_float=True, factor=1.0):
        slider.valueChanged.connect(lambda val: edit.setText(f"{val / factor:.3f}" if is_float else str(val)))
        edit.editingFinished.connect(lambda: slider.setValue(int(float(edit.text()) * factor) if is_float else int(edit.text()))); edit.editingFinished.connect(self._emit_param_change)

class ModelPanel(QWidget):
    model_load_requested = Signal(); model_stop_requested = Signal(); settings_changed = Signal(dict)
    def __init__(self):
        super().__init__(); self.class_names = []; self.class_checkboxes = {}; self.good_checkboxes = {}; self.ng_checkboxes = {}
        main_layout = QVBoxLayout(self); form_layout = QFormLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal); self.conf_slider.setRange(0, 100); self.conf_edit = QLineEdit("0.50"); form_layout.addRow("Confidence:", self._create_hbox(self.conf_slider, self.conf_edit))
        self.align_slider = QSlider(Qt.Orientation.Horizontal); self.align_slider.setRange(1, 100); self.align_edit = QLineEdit("0.02"); form_layout.addRow("Alignment:", self._create_hbox(self.align_slider, self.align_edit))
        self.bottle_combo = QComboBox(); self.logo_combo = QComboBox(); form_layout.addRow("Major Class (bottle):", self.bottle_combo); form_layout.addRow("Target Class (logo):", self.logo_combo)
        main_layout.addLayout(form_layout)
        self.detect_group = QGroupBox("Classes to Detect"); detect_layout = QVBoxLayout(); self.detect_group.setLayout(detect_layout)
        self.good_group = QGroupBox("Good Classes"); good_layout = QVBoxLayout(); self.good_group.setLayout(good_layout)
        self.ng_group = QGroupBox("Not Good Classes"); ng_layout = QVBoxLayout(); self.ng_group.setLayout(ng_layout)
        classes_layout = QHBoxLayout(); classes_layout.addWidget(self.good_group); classes_layout.addWidget(self.ng_group)
        main_layout.addWidget(self.detect_group); main_layout.addLayout(classes_layout)
        btn_layout = QHBoxLayout(); self.load_btn = QPushButton("Load Model"); self.stop_btn = QPushButton("Stop/Unload Model")
        btn_layout.addWidget(self.load_btn); btn_layout.addWidget(self.stop_btn); main_layout.addLayout(btn_layout); main_layout.addStretch(); self._connect_signals()
    def _connect_signals(self):
        self.load_btn.clicked.connect(self.model_load_requested.emit); self.stop_btn.clicked.connect(self.model_stop_requested.emit)
        for w in [self.conf_slider, self.align_slider, self.bottle_combo, self.logo_combo]:
            if isinstance(w, QSlider): w.valueChanged.connect(self._emit_settings_change)
            else: w.currentIndexChanged.connect(self._emit_settings_change)
    def update_class_lists(self, class_names):
        self.class_names = class_names; self.bottle_combo.clear(); self.bottle_combo.addItems(class_names); self.logo_combo.clear(); self.logo_combo.addItems(class_names)
        for group_box, checkbox_dict in [(self.detect_group, self.class_checkboxes), (self.good_group, self.good_checkboxes), (self.ng_group, self.ng_checkboxes)]:
            layout = group_box.layout()
            while layout.count(): layout.takeAt(0).widget().deleteLater()
            checkbox_dict.clear()
            for name in class_names: cb = QCheckBox(name); cb.setChecked(True); cb.stateChanged.connect(self._emit_settings_change); layout.addWidget(cb); checkbox_dict[name] = cb
        self._emit_settings_change()
    def _emit_settings_change(self):
        self.conf_edit.setText(f"{self.conf_slider.value() / 100.0:.2f}"); self.align_edit.setText(f"{self.align_slider.value() / 100.0:.2f}")
        settings = {"ConfidenceThreshold": self.conf_slider.value() / 100.0, "AlignmentTolerance": self.align_slider.value() / 100.0, "BottleClass": self.bottle_combo.currentText(), "LogoClass": self.logo_combo.currentText(), "ClassesToDetect": {name: cb.isChecked() for name, cb in self.class_checkboxes.items()}, "Good": [name for name, cb in self.good_checkboxes.items() if cb.isChecked()], "NotGood": [name for name, cb in self.ng_checkboxes.items() if cb.isChecked()],}; self.settings_changed.emit(settings)
    def _create_hbox(self, w1, w2): layout = QHBoxLayout(); layout.addWidget(w1); w2.setFixedWidth(50); layout.addWidget(w2); return layout

class ROIPanel(QWidget):
    roi_changed = Signal(dict)
    def __init__(self):
        super().__init__(); main_layout = QVBoxLayout(self); form_layout = QFormLayout()
        self.width_slider = QSlider(Qt.Orientation.Horizontal); self.width_slider.setRange(10, 100); self.width_slider.setValue(100)
        self.height_slider = QSlider(Qt.Orientation.Horizontal); self.height_slider.setRange(10, 100); self.height_slider.setValue(100)
        self.shiftx_slider = QSlider(Qt.Orientation.Horizontal); self.shiftx_slider.setRange(-100, 100); self.shiftx_slider.setValue(0)
        self.shifty_slider = QSlider(Qt.Orientation.Horizontal); self.shifty_slider.setRange(-100, 100); self.shifty_slider.setValue(0)
        form_layout.addRow("Width (%)", self.width_slider); form_layout.addRow("Height (%)", self.height_slider)
        form_layout.addRow("Shift X", self.shiftx_slider); form_layout.addRow("Shift Y", self.shifty_slider)
        main_layout.addLayout(form_layout); main_layout.addStretch()
        for w in [self.width_slider, self.height_slider, self.shiftx_slider, self.shifty_slider]: w.valueChanged.connect(self._emit_change)
    def _emit_change(self):
        settings = {'width': self.width_slider.value()/100.0, 'height': self.height_slider.value()/100.0, 'shift_x': self.shiftx_slider.value()/100.0, 'shift_y': self.shifty_slider.value()/100.0}
        self.roi_changed.emit(settings)

class TransformPanel(QWidget):
    transform_changed = Signal(dict)
    def __init__(self):
        super().__init__(); main_layout = QVBoxLayout(self); form_layout = QFormLayout()
        self.rotate_combo = QComboBox(); self.rotate_combo.addItems(["0", "90", "180", "-90"])
        self.flip_ud_cb = QCheckBox("Flip Up/Down"); self.flip_lr_cb = QCheckBox("Flip Left/Right")
        form_layout.addRow("Rotation:", self.rotate_combo); form_layout.addRow(self.flip_ud_cb); form_layout.addRow(self.flip_lr_cb)
        main_layout.addLayout(form_layout); main_layout.addStretch()
        self.rotate_combo.currentIndexChanged.connect(self._emit_change)
        self.flip_ud_cb.stateChanged.connect(self._emit_change); self.flip_lr_cb.stateChanged.connect(self._emit_change)
    def _emit_change(self):
        settings = {'rotate': int(self.rotate_combo.currentText()), 'flip_ud': self.flip_ud_cb.isChecked(), 'flip_lr': self.flip_lr_cb.isChecked()}
        self.transform_changed.emit(settings)

class MediaCapturePanel(QGroupBox):
    capture_image_requested = Signal(str); toggle_recording_requested = Signal(str)
    def __init__(self):
        super().__init__("Media Capture (All Feeds)"); main_layout = QVBoxLayout(self)
        self.folder_edit = QLineEdit("capture_session"); self.folder_edit.setPlaceholderText("Enter folder name...")
        self.history_list = QListWidget()
        self.capture_img_btn = QPushButton("Capture Image"); self.record_btn = QPushButton("Start Recording")
        main_layout.addWidget(self.folder_edit); main_layout.addWidget(self.history_list); 
        btn_layout = QHBoxLayout(); btn_layout.addWidget(self.capture_img_btn); btn_layout.addWidget(self.record_btn)
        main_layout.addLayout(btn_layout)
        self.capture_img_btn.clicked.connect(lambda: self.capture_image_requested.emit(self.folder_edit.text()))
        self.record_btn.clicked.connect(lambda: self.toggle_recording_requested.emit(self.folder_edit.text()))
    def add_history(self, message): self.history_list.insertItem(0, message)
    def set_recording_state(self, is_recording): self.record_btn.setText("Stop Recording" if is_recording else "Start Recording")

class Camera_Thread(QThread):
    frame_ready = Signal(str, np.ndarray); error = Signal(str, str); model_loaded = Signal(str, list)
    def __init__(self, device_info):
        super().__init__()
        self.device_info, self.serial = device_info, device_info.GetSerialNumber()
        self._running = False
        self.exposure, self.brightness, self.contrast = 1000.0, 0.0, 0.0; self.use_scurve = False
        self.model, self.model_metadata, self.inference_enabled, self.model_settings = None, None, False, {}
        self.roi_settings = {'width': 1.0, 'height': 1.0, 'shift_x': 0.0, 'shift_y': 0.0}
        self.transform_settings = {'rotate': 0, 'flip_ud': False, 'flip_lr': False}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def stop(self): self._running = False
    def update_params(self, exp, bri, con): self.exposure, self.brightness, self.contrast = exp, bri, con
    def update_contrast_mode(self, use_scurve): self.use_scurve = use_scurve
    def update_model_settings(self, settings): self.model_settings = settings
    def update_roi_settings(self, settings): self.roi_settings = settings
    def update_transform_settings(self, settings): self.transform_settings = settings

    def load_model(self, model_path, metadata_path):
        try:
            with open(metadata_path, 'r') as f: self.model_metadata = json.load(f)
            class_names = self.model_metadata.get("class_names", []); num_classes = len(class_names) + 1
            self.model = get_my_maskrcnn_model(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict); self.model.to(self.device).eval()
            self.inference_enabled = True; self.model_loaded.emit(self.serial, class_names)
        except Exception as e: self.error.emit(self.serial, f"Failed to load model: {e}")

    def unload_model(self): self.model, self.model_metadata, self.inference_enabled = None, None, False

    def run(self):
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info)); cam.Open()
            cam.AcquisitionFrameRateEnable.SetValue(True); cam.AcquisitionFrameRate.SetValue(30)
            fmt = pylon.ImageFormatConverter(); fmt.OutputPixelFormat = pylon.PixelType_BGR8packed
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly); self._running = True

            while self._running and cam.IsGrabbing():
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if not grab.GrabSucceeded(): continue
                
                frame_np = fmt.Convert(grab).GetArray()
                
                # Apply Transformations
                rot = self.transform_settings['rotate']; ud = self.transform_settings['flip_ud']; lr = self.transform_settings['flip_lr']
                if rot == 90: frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180: frame_np = cv2.rotate(frame_np, cv2.ROTATE_180)
                elif rot == -90: frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if ud: frame_np = cv2.flip(frame_np, 0)
                if lr: frame_np = cv2.flip(frame_np, 1)

                h, w, _ = frame_np.shape
                roi_w = int(w * self.roi_settings['width']); roi_h = int(h * self.roi_settings['height'])
                roi_cx = int(w/2 + self.roi_settings['shift_x'] * w/2); roi_cy = int(h/2 + self.roi_settings['shift_y'] * h/2)
                roi_x1, roi_y1 = max(0, roi_cx - roi_w//2), max(0, roi_cy - roi_h//2)
                roi_x2, roi_y2 = min(w, roi_x1 + roi_w), min(h, roi_y1 + roi_h)

                final_frame = frame_np.copy()
                if self.inference_enabled and self.model:
                    pil_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                    img_tensor = F.to_tensor(pil_img).to(self.device)
                    with torch.no_grad(): predictions = self.model([img_tensor]); pred = predictions[0]
                    conf = self.model_settings.get("ConfidenceThreshold", 0.5); idx = pred["scores"] > conf
                    boxes, scores, masks, labels_idx = pred["boxes"][idx], pred["scores"][idx], pred["masks"][idx].squeeze(1), pred["labels"][idx]

                    # Filter by ROI
                    roi_filter = (boxes[:, 0] >= roi_x1) & (boxes[:, 1] >= roi_y1) & (boxes[:, 2] <= roi_x2) & (boxes[:, 3] <= roi_y2)
                    boxes, scores, masks, labels_idx = boxes[roi_filter], scores[roi_filter], masks[roi_filter], labels_idx[roi_filter]
                    
                    if len(boxes) > 0:
                        class_names = self.model_metadata.get("class_names", [])
                        labels_text = [class_names[i-1] for i in labels_idx]
                        filtered_boxes, filtered_scores, filtered_labels_text, filtered_masks = filter_overlapping_detections(boxes, scores, labels_text, masks)
                        display_labels = [f"{name} {score:.2f}" for name, score in zip(filtered_labels_text, filtered_scores)]
                        resized_masks = resize_and_binarize_masks(filtered_masks, pil_img.size)
                        class_colors = self.model_metadata.get("class_colors", {})
                        overlay_img = overlay_masks_boxes_labels_predict(pil_img, resized_masks, filtered_boxes, class_colors, display_labels)
                        final_frame = cv2.cvtColor(np.array(overlay_img), cv2.COLOR_RGB2BGR)
                
                cv2.rectangle(final_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
                self.frame_ready.emit(self.serial, final_frame)
                grab.Release()
        except Exception as e: self.error.emit(self.serial, str(e))
        finally:
            if 'cam' in locals() and cam.IsGrabbing(): cam.StopGrabbing()
            if 'cam' in locals() and cam.IsOpen(): cam.Close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Multi-Camera GUI with Inference")
        self.camera_threads = {}; self.video_panels = {}; self.control_panels = {}; self.model_panels = {}
        self.is_recording = False; self.video_writers = {}
        self.detect_and_setup_ui()

    def detect_and_setup_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget); main_layout = QHBoxLayout(central_widget)
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        control_tabs_widget = QTabWidget()
        left_layout.addWidget(control_tabs_widget)
        
        video_grid_widget = QWidget(); video_grid_layout = QGridLayout(video_grid_widget)
        main_layout.addWidget(left_widget, 35); main_layout.addWidget(video_grid_widget, 65)

        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices: video_grid_layout.addWidget(QLabel("No Basler cameras detected."))
        else:
            positions = [(i, j) for i in range(2) for j in range(2)]
            for i, dev in enumerate(devices):
                if i >= 4: break
                serial = dev.GetSerialNumber()

                panel_container = QWidget(); panel_layout = QVBoxLayout(panel_container)
                title_lbl = QLabel(f"<b>Camera {i+1}</b> ({serial})"); title_lbl.setAlignment(Qt.AlignCenter)
                video_lbl = QLabel("Disconnected"); video_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                video_lbl.setAlignment(Qt.AlignCenter); video_lbl.setStyleSheet("background-color:black; color:white;")
                panel_layout.addWidget(title_lbl); panel_layout.addWidget(video_lbl); row, col = positions[i]; video_grid_layout.addWidget(panel_container, row, col)
                self.video_panels[serial] = video_lbl

                # Create control panels
                control_panel = ControlPanel(); self.control_panels[serial] = control_panel
                model_panel = ModelPanel(); self.model_panels[serial] = model_panel
                roi_panel = ROIPanel(); transform_panel = TransformPanel()
                
                side_tabs = QTabWidget(); side_tabs.setTabPosition(QTabWidget.TabPosition.West)
                side_tabs.addTab(control_panel, "Image"); side_tabs.addTab(model_panel, "Model")
                side_tabs.addTab(roi_panel, "ROI"); side_tabs.addTab(transform_panel, "Transform")
                control_tabs_widget.addTab(side_tabs, f"Cam {i+1}")

                # Connect signals
                control_panel.params_changed.connect(lambda exp, bri, con, s=serial: self.push_params_to_thread(s, exp, bri, con))
                control_panel.contrast_mode_changed.connect(lambda use_scurve, s=serial: self.push_contrast_mode_to_thread(s, use_scurve))
                control_panel.load_defaults_requested.connect(control_panel.set_to_defaults)
                model_panel.model_load_requested.connect(lambda s=serial: self.load_model_for_camera(s))
                model_panel.model_stop_requested.connect(lambda s=serial: self.stop_model_for_camera(s))
                model_panel.settings_changed.connect(lambda settings, s=serial: self.push_model_settings_to_thread(s, settings))
                roi_panel.roi_changed.connect(lambda settings, s=serial: self.push_roi_settings_to_thread(s, settings))
                transform_panel.transform_changed.connect(lambda settings, s=serial: self.push_transform_settings_to_thread(s, settings))
        
        # Add global panels
        self.media_panel = MediaCapturePanel(); left_layout.addWidget(self.media_panel)
        self.media_panel.capture_image_requested.connect(self.capture_image)
        self.media_panel.toggle_recording_requested.connect(self.toggle_recording)

        buttons_layout = QHBoxLayout(); self.start_btn = QPushButton("Start All"); self.stop_btn = QPushButton("Stop All")
        buttons_layout.addWidget(self.start_btn); buttons_layout.addWidget(self.stop_btn); left_layout.addLayout(buttons_layout)
        
        if not devices: self.start_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        else:
            self.start_btn.clicked.connect(self.start_all_cameras)
            self.stop_btn.clicked.connect(self.stop_all_cameras)
            self.stop_btn.setEnabled(False)
            for panel in self.control_panels.values(): panel.setEnabled(False)
            for panel in self.model_panels.values(): panel.setEnabled(False)

    def start_all_cameras(self):
        if self.camera_threads: return
        devices = {dev.GetSerialNumber(): dev for dev in pylon.TlFactory.GetInstance().EnumerateDevices()}
        for serial in self.control_panels.keys():
            if serial in devices:
                thread = Camera_Thread(devices[serial]); thread.frame_ready.connect(self.update_frame); thread.error.connect(self.show_error); thread.model_loaded.connect(self.on_model_loaded)
                thread.start(); self.camera_threads[serial] = thread
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        for panel in self.control_panels.values(): panel.setEnabled(True)
        for panel in self.model_panels.values(): panel.setEnabled(True)

    def stop_all_cameras(self):
        self.toggle_recording("", force_stop=True) # Stop recording if active
        for thread in self.camera_threads.values(): thread.stop(); thread.wait()
        self.camera_threads.clear()
        for serial, panel in self.video_panels.items(): panel.setPixmap(QPixmap()); panel.setText("Disconnected")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        for panel in self.control_panels.values(): panel.setEnabled(False)
        for panel in self.model_panels.values(): panel.setEnabled(False)

    def load_model_for_camera(self, serial):
        model_path, _ = QFileDialog.getOpenFileName(self, f"Select Model for {serial}", "", "PyTorch Models (*.pth)");
        if not model_path: return
        metadata_path, _ = QFileDialog.getOpenFileName(self, f"Select Metadata for {serial}", "", "JSON files (*.json)");
        if not metadata_path: return
        if serial in self.camera_threads: self.camera_threads[serial].load_model(model_path, metadata_path)

    def stop_model_for_camera(self, serial):
        if serial in self.camera_threads: self.camera_threads[serial].unload_model(); QMessageBox.information(self, "Model Unloaded", f"Model has been unloaded for camera {serial}.")
    
    def on_model_loaded(self, serial, class_names):
        if serial in self.model_panels: self.model_panels[serial].update_class_lists(class_names)
        QMessageBox.information(self, "Model Loaded", f"Model loaded successfully for camera {serial}.")

    def update_frame(self, serial, frame):
        if serial in self.video_panels:
            video_lbl = self.video_panels[serial]; h, w, ch = frame.shape; q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img); video_lbl.setPixmap(pixmap.scaled(video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            if self.is_recording and serial in self.video_writers:
                writer = self.video_writers[serial]
                if writer.isOpened():
                    # Ensure frame matches writer dimensions
                    frame_h, frame_w, _ = frame.shape
                    writer_w, writer_h = writer.get(cv2.CAP_PROP_FRAME_WIDTH), writer.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if frame_w != writer_w or frame_h != writer_h:
                        frame = cv2.resize(frame, (writer_w, writer_h))
                    writer.write(frame)

    def capture_image(self, folder_name):
        if not self.camera_threads: QMessageBox.warning(self, "Warning", "Cameras are not running."); return
        if not folder_name.strip(): QMessageBox.warning(self, "Warning", "Please enter a folder name."); return
        
        for serial, panel in self.video_panels.items():
            pixmap = panel.pixmap()
            if pixmap and not pixmap.isNull():
                base_dir = os.path.join(os.path.dirname(__file__), "media", folder_name.strip(), serial)
                os.makedirs(base_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(base_dir, f"capture_{timestamp}.png")
                pixmap.save(file_path)
                self.media_panel.add_history(f"Saved: {os.path.relpath(file_path)}")

    def toggle_recording(self, folder_name, force_stop=False):
        if self.is_recording or force_stop:
            for writer in self.video_writers.values(): writer.release()
            self.video_writers.clear(); self.is_recording = False
            self.media_panel.set_recording_state(False)
            if not force_stop: self.media_panel.add_history("Recording stopped.")
        else: # Start recording
            if not self.camera_threads: QMessageBox.warning(self, "Warning", "Cameras are not running."); return
            if not folder_name.strip(): QMessageBox.warning(self, "Warning", "Please enter a folder name."); return
            self.is_recording = True
            for serial, panel in self.video_panels.items():
                pixmap = panel.pixmap()
                if pixmap and not pixmap.isNull():
                    base_dir = os.path.join(os.path.dirname(__file__), "media", folder_name.strip(), serial)
                    os.makedirs(base_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = os.path.join(base_dir, f"video_{timestamp}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID'); fps = 5.0
                    dims = (pixmap.width(), pixmap.height())
                    self.video_writers[serial] = cv2.VideoWriter(file_path, fourcc, fps, dims)
            self.media_panel.set_recording_state(True)
            self.media_panel.add_history("Recording started...")

    def push_params_to_thread(self, s, exp, bri, con):
        if s in self.camera_threads: self.camera_threads[s].update_params(exp, bri, con)
    def push_contrast_mode_to_thread(self, s, use_scurve):
        if s in self.camera_threads: self.camera_threads[s].update_contrast_mode(use_scurve)
    def push_model_settings_to_thread(self, s, settings):
        if s in self.camera_threads: self.camera_threads[s].update_model_settings(settings)
    def push_roi_settings_to_thread(self, s, settings):
        if s in self.camera_threads: self.camera_threads[s].update_roi_settings(settings)
    def push_transform_settings_to_thread(self, s, settings):
        if s in self.camera_threads: self.camera_threads[s].update_transform_settings(settings)

    def show_error(self, serial, message): QMessageBox.warning(self, f"Camera Error ({serial})", message)
    def closeEvent(self, event): self.stop_all_cameras(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())