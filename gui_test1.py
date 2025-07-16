import os
import cv2
import sys
import csv
import json
import time
import torch
import queue
import serial
import threading
import torchvision
import numpy as np

from pypylon import pylon
from datetime import datetime
from serial.tools import list_ports  
from PIL import Image, ImageTk, ImageDraw, ImageFont

from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, resnet18, ResNet18_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor, BackboneWithFPN

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QLineEdit, QComboBox,
    QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QSizePolicy, QGroupBox,
    QMainWindow, QTabWidget, QCheckBox, QGridLayout, QFileDialog
)

############################################
# Helpers & Overlay Functions
############################################
def parse_resolution(resolution_str):
	return tuple(map(int, resolution_str.split()[0].split("x")))

def overlay_masks_boxes_labels_predict(image, masks, boxes, class_colors, labels, alpha=0.3):
	overlay = image.convert("RGBA")
	for mask, label in zip(masks, labels):
		mask_array = mask.cpu().numpy().astype("uint8") * 255
		mask_pil = Image.fromarray(mask_array)
		if label and isinstance(label, str):
			class_name = label.split()[0]
		else:
			class_name = "unknown"
		color = tuple(class_colors.get(class_name, (255, 255, 255))) + (int(255 * alpha),)
		colored_mask = Image.new("RGBA", overlay.size, color)
		mask_image = Image.new("RGBA", overlay.size)
		mask_image.paste(colored_mask, (0, 0), mask_pil)
		overlay = Image.alpha_composite(overlay, mask_image)

	overlay = overlay.convert("RGB")
	draw = ImageDraw.Draw(overlay)
	image_width, image_height = image.size
	dynamic_font_size = max(20, min(image_width, image_height) // 50)

	try:
		font = ImageFont.load_default(dynamic_font_size)
	except IOError:
		font = ImageFont.load_default()
		print("Default font is being used. Text size may be small.")

	for box, label in zip(boxes, labels):
		x_min, y_min, x_max, y_max = box
		class_name = label.split()[0]
		color = tuple(class_colors.get(class_name, (255, 255, 255)))
		draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=5)
		text_size = draw.textbbox((0, 0), label, font=font)
		label_width = text_size[2] - text_size[0]
		label_height = text_size[3] - text_size[1]
		if y_min - label_height - 12 < 0:
			label_position = (x_min, y_max + 12)
			label_bg_position = [
				(x_min, y_max + 12),
				(x_min + label_width + 12, y_max + label_height + 16)
			]
		else:
			label_position = (x_min + 4, y_min - label_height - 12)
			label_bg_position = [
				(x_min, y_min - label_height - 12),
				(x_min + label_width + 12, y_min)
			]
		draw.rectangle(label_bg_position, fill=color)
		draw.text(label_position, label, fill="black", font=font)

	return overlay

def resize_and_binarize_masks(masks, image_size):
	import torch.nn.functional as torch_F
	resized_masks = []
	for mask in masks:
		if mask.dim() == 2:
			mask = mask.unsqueeze(0).unsqueeze(0)
		elif mask.dim() == 3 and mask.size(0) == 1:
			mask = mask.unsqueeze(0)
		resized_mask = torch_F.interpolate(mask, size=image_size, mode="bilinear", align_corners=False)
		binarized_mask = (resized_mask > 0.5).float().squeeze(0).squeeze(0)
		resized_masks.append(binarized_mask.byte())
	return resized_masks

def filter_overlapping_detections(boxes, scores, labels, masks, threshold=0.25):
	keep = []
	indices = torch.argsort(scores, descending=True)
	while len(indices) > 0:
		current = indices[0]
		keep.append(current)
		if len(indices) == 1:
			break
		current_box = boxes[current]
		current_label = labels[current]
		other_indices = indices[1:]
		other_boxes = boxes[other_indices]
		other_labels = labels[other_indices]
		x1 = torch.max(current_box[0], other_boxes[:, 0])
		y1 = torch.max(current_box[1], other_boxes[:, 1])
		x2 = torch.min(current_box[2], other_boxes[:, 2])
		y2 = torch.min(current_box[3], other_boxes[:, 3])
		inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
		current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
		other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
		union_area = current_area + other_areas - inter_area
		iou = inter_area / union_area
		mask = (other_labels != current_label) | (iou <= threshold)
		indices = other_indices[mask]
	keep = torch.tensor(keep, dtype=torch.long)
	filtered_boxes = boxes[keep]
	filtered_scores = scores[keep]
	filtered_labels = labels[keep]
	filtered_masks = [masks[i] for i in keep.tolist()]
	return filtered_boxes, filtered_scores, filtered_labels, filtered_masks

##############################################
# Sub-Notebook Approach for Each Controller
##############################################

class ImagePreprocessingTab(QWidget):
    params_changed = Signal(float, float, float)
    contrast_mode_changed = Signal(bool)
    load_defaults_requested = Signal()
    
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        params_form = QFormLayout()
        title_lbl = QLabel("<b>Image Preprocessing Parameters</b>")
        main_layout.addWidget(title_lbl)
        main_layout.addLayout(params_form)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet("font-size: 20pt; font-weight: bold;")
        
        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(2, 50_000)
        self.exp_slider.setValue(1000)
        self.exp_edit = QLineEdit("1000")
        params_form.addRow("Exposure (µs)", self._create_hbox(self.exp_slider, self.exp_edit))
        
        self.bri_slider = QSlider(Qt.Orientation.Horizontal)
        self.bri_slider.setRange(-1000, 1000)
        self.bri_slider.setValue(0)
        self.bri_edit = QLineEdit("0.000")
        params_form.addRow("Brightness", self._create_hbox(self.bri_slider, self.bri_edit))

        self.con_slider = QSlider(Qt.Orientation.Horizontal)
        self.con_slider.setRange(-1000, 1000)
        self.con_slider.setValue(0)
        self.con_edit = QLineEdit("0.000")
        params_form.addRow("Contrast", self._create_hbox(self.con_slider, self.con_edit))
        
        self.use_scurve_checkbox = QCheckBox("Use S-Curve Contrast")
        params_form.addRow("", self.use_scurve_checkbox)

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
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: 
            w.blockSignals(True)
        self.exp_slider.setValue(1000)
        self.bri_slider.setValue(0)
        self.con_slider.setValue(0)
        self.use_scurve_checkbox.setChecked(False)
        self.exp_edit.setText("1000")
        self.bri_edit.setText("0.000")
        self.con_edit.setText("0.000")
        
        for w in [self.exp_slider, self.bri_slider, self.con_slider, self.use_scurve_checkbox]: 
            w.blockSignals(False)
        self._emit_param_change()
        self.contrast_mode_changed.emit(False)

    def _create_hbox(self, w1, w2): 
        layout = QHBoxLayout()
        layout.addWidget(w1)
        w2.setFixedWidth(70)
        layout.addWidget(w2)
        return layout
    
    def _connect_slider_edit(self, slider, edit, is_float=True, factor=1.0):
        slider.valueChanged.connect(lambda val: edit.setText(f"{val / factor:.3f}" if is_float else str(val)))
        edit.editingFinished.connect(lambda: slider.setValue(int(float(edit.text()) * factor) if is_float else int(edit.text())))
        edit.editingFinished.connect(self._emit_param_change)

class ModelPanel(QWidget):
    model_load_requested = Signal()
    model_stop_requested = Signal()
    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.class_names = []
        self.class_checkboxes = {}
        self.good_checkboxes = {}
        self.ng_checkboxes = {}

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Confidence
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_edit = QLineEdit("0.50")
        form_layout.addRow("Confidence:", self._create_hbox(self.conf_slider, self.conf_edit))
        
        # Alignment
        self.align_slider = QSlider(Qt.Orientation.Horizontal)
        self.align_slider.setRange(1, 100)
        self.align_edit = QLineEdit("0.02")
        form_layout.addRow("Alignment:", self._create_hbox(self.align_slider, self.align_edit))

        # Major/Target Classes
        self.bottle_combo = QComboBox()
        self.logo_combo = QComboBox()
        form_layout.addRow("Major Class (bottle):", self.bottle_combo)
        form_layout.addRow("Target Class (logo):", self.logo_combo)
        
        main_layout.addLayout(form_layout)

        # Class Toggles
        self.detect_group = QGroupBox("Classes to Detect")
        detect_layout = QVBoxLayout()
        self.detect_group.setLayout(detect_layout)
        self.good_group = QGroupBox("Good Classes")
        good_layout = QVBoxLayout()
        self.good_group.setLayout(good_layout)
        self.ng_group = QGroupBox("Not Good Classes")
        ng_layout = QVBoxLayout()
        self.ng_group.setLayout(ng_layout)

        classes_layout = QHBoxLayout()
        classes_layout.addWidget(self.good_group)
        classes_layout.addWidget(self.ng_group)

        main_layout.addWidget(self.detect_group)
        main_layout.addLayout(classes_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Model")
        self.stop_btn = QPushButton("Stop/Unload Model")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

        self._connect_signals()

    def _connect_signals(self):
        self.load_btn.clicked.connect(self.model_load_requested.emit)
        self.stop_btn.clicked.connect(self.model_stop_requested.emit)
        # Connect all controls to emit settings change
        for w in [self.conf_slider, self.align_slider, self.bottle_combo, self.logo_combo]:
            if isinstance(w, QSlider): w.valueChanged.connect(self._emit_settings_change)
            else: w.currentIndexChanged.connect(self._emit_settings_change)
    
    def update_class_lists(self, class_names):
        self.class_names = class_names
        self.bottle_combo.clear()
        self.bottle_combo.addItems(class_names)
        self.logo_combo.clear()
        self.logo_combo.addItems(class_names)

        for group_box, checkbox_dict in [(self.detect_group, self.class_checkboxes), (self.good_group, self.good_checkboxes), (self.ng_group, self.ng_checkboxes)]:
            layout = group_box.layout()
            while layout.count():
                layout.takeAt(0).widget().deleteLater()
            checkbox_dict.clear()
            for name in class_names:
                cb = QCheckBox(name)
                cb.setChecked(True)
                cb.stateChanged.connect(self._emit_settings_change)
                layout.addWidget(cb)
                checkbox_dict[name] = cb
        self._emit_settings_change()

    def _emit_settings_change(self):
        # Update line edits from sliders
        self.conf_edit.setText(f"{self.conf_slider.value() / 100.0:.2f}")
        self.align_edit.setText(f"{self.align_slider.value() / 100.0:.2f}")

        settings = {
            "ConfidenceThreshold": self.conf_slider.value() / 100.0,
            "AlignmentTolerance": self.align_slider.value() / 100.0,
            "BottleClass": self.bottle_combo.currentText(),
            "LogoClass": self.logo_combo.currentText(),
            "ClassesToDetect": {name: cb.isChecked() for name, cb in self.class_checkboxes.items()},
            "Good": [name for name, cb in self.good_checkboxes.items() if cb.isChecked()],
            "NotGood": [name for name, cb in self.ng_checkboxes.items() if cb.isChecked()],
        }
        self.settings_changed.emit(settings)
        
    def _create_hbox(self, w1, w2):
        layout = QHBoxLayout()
        layout.addWidget(w1)
        w2.setFixedWidth(50)
        layout.addWidget(w2)
        return layout

class Camera_Thread(QThread):
    frame_ready = Signal(str, np.ndarray)
    error = Signal(str, str)
    model_loaded = Signal(str, list) # serial, class_names

    def __init__(self, device_info):
        super().__init__()
        self.device_info = device_info
        self.serial = device_info.GetSerialNumber()
        self._running = False
        self.exposure, self.brightness, self.contrast = 1000.0, 0.0, 0.0
        self.use_scurve = False
        
        # Model-related attributes
        self.model, self.model_metadata = None, None
        self.inference_enabled = False
        self.model_settings = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def stop(self): 
        self._running = False
        
    def update_params(self, exp, bri, con): 
        self.exposure, self.brightness, self.contrast = exp, bri, con
        
    def update_contrast_mode(self, use_scurve): 
        self.use_scurve = use_scurve
        
    def update_model_settings(self, settings): 
        self.model_settings = settings

    def load_model(self, model_path, metadata_path):
        try:
            # STEP 1: Load metadata
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)

            # STEP 2: Rebuild the model architecture based on metadata
            model_type = self.model_metadata.get("modelType", 0)
            # Use num_classes from metadata, provide a fallback for safety
            num_classes = len(self.model_metadata.get("class_names", ["BG", "FG"]))

            if model_type == 0:
                print("Loading model with ResNet-50 backbone...")
                model = maskrcnn_resnet50_fpn(weights=None)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            elif model_type == 1:
                print("Loading model with MobileNetV3 backbone...")
                weights = MobileNet_V3_Large_Weights.DEFAULT
                mobilenet_backbone = mobilenet_v3_large(weights=weights).features
                
                # Define which layers to extract for the FPN
                # These indices might need adjustment based on your specific MobileNetV3 architecture
                layers_to_extract = [3, 6, 12]
                feature_channels = [mobilenet_backbone[i][-1].out_channels for i in layers_to_extract]

                fpn = FeaturePyramidNetwork(in_channels_list=feature_channels, out_channels=256)

                class CustomBackboneWithFPN(torch.nn.Module):
                    def __init__(self, body, fpn):
                        super().__init__()
                        self.body = body
                        self.fpn = fpn
                        self.out_channels = 256
                    
                    def forward(self, x):
                        features = []
                        for idx, module in enumerate(self.body.children()):
                            x = module(x)
                            if idx in layers_to_extract:
                                features.append(x)
                        features_dict = {str(i): features[i] for i in range(len(features))}
                        return self.fpn(features_dict)

                backbone_with_fpn = CustomBackboneWithFPN(mobilenet_backbone, fpn)

                anchor_generator = AnchorGenerator(
                    sizes=((32,), (64,), (128,)),
                    aspect_ratios=((0.5, 1.0, 2.0),) * len(layers_to_extract)
                )
                model = MaskRCNN(
                    backbone_with_fpn,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator
                )

            elif model_type == 2:
                print("Loading model with ResNet-18 backbone...")
                weights = ResNet18_Weights.DEFAULT
                resnet_model = resnet18(weights=weights)
                backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
                backbone.out_channels = 512 # ResNet-18's last conv block has 512 channels
                
                anchor_generator = AnchorGenerator(
                    sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
                )
                roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=['0'], output_size=7, sampling_ratio=2
                )
                mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=['0'], output_size=14, sampling_ratio=2
                )
                
                model = MaskRCNN(
                    backbone=backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    mask_roi_pool=mask_roi_pooler
                )
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                
            else:
                raise ValueError(f"Unsupported modelType {model_type} in metadata.")

            # STEP 3: Load the state_dict
            state_dict = torch.load(model_path, map_location=self.device)
            corrected_state_dict = { key.replace("roi_heads.box_predictor.predictor.", "roi_heads.box_predictor."): value for key, value in state_dict.items() }
            model.load_state_dict(corrected_state_dict)

            # STEP 4: Move model to the correct device and set to evaluation mode
            model.to(self.device).eval()
            
            # Use .half() for faster inference if your GPU supports it
            if self.device.type == 'cuda':
                model.half()

            # STEP 5: Assign the loaded model and enable inference
            self.model = model
            self.inference_enabled = True

            # Emit signal to notify the main window that the model is ready
            self.model_loaded.emit(self.serial, self.model_metadata.get("class_names", []))

        except Exception as e:
            # Emit an error signal if anything goes wrong
            self.error.emit(self.serial, f"Failed to load model: {e}")

    def unload_model(self):
        self.model, self.model_metadata, self.inference_enabled = None, None, False

    def run(self):
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info))
            cam.Open()
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(30)
            fmt = pylon.ImageFormatConverter()
            fmt.OutputPixelFormat = pylon.PixelType_BGR8packed
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._running = True

            while self._running and cam.IsGrabbing():
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if not grab.GrabSucceeded(): 
                    continue
                
                if grab.GrabSucceeded():
                    try:
                        cam.BslContrastMode.SetValue("SCurve" if self.use_scurve else "Linear")
                        cam.ExposureTime.SetValue(self.exposure)
                        cam.BslBrightness.SetValue(self.brightness)
                        cam.BslContrast.SetValue(self.contrast)
                    except Exception: 
                        pass
                
                frame_np = fmt.Convert(grab).GetArray()
                
                if self.inference_enabled and self.model:
                    pil_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                    img_tensor = F.to_tensor(pil_img).to(self.device)
                    
                    with torch.no_grad(): predictions = self.model([img_tensor])
                    pred = predictions[0]

                    conf = self.model_settings.get("ConfidenceThreshold", 0.5)
                    idx = pred["scores"] > conf
                    boxes = pred["boxes"][idx]
                    scores = pred["scores"][idx]
                    masks = pred["masks"][idx].squeeze(1)
                    labels_idx = pred["labels"][idx]
                    class_names = self.model_metadata.get("class_names", [])
                    labels = [f"{class_names[i]} {s:.2f}" for i, s in zip(labels_idx, scores)]

                    filtered_boxes, _, filtered_labels, filtered_masks = filter_overlapping_detections(boxes, scores, labels, masks)
                    resized_masks = resize_and_binarize_masks(filtered_masks, pil_img.size)

                    class_colors = self.model_metadata.get("class_colors", {})
                    overlay_img = overlay_masks_boxes_labels_predict(pil_img, resized_masks, filtered_boxes, class_colors, filtered_labels)
                    
                    final_frame = cv2.cvtColor(np.array(overlay_img), cv2.COLOR_RGB2BGR)
                    self.frame_ready.emit(self.serial, final_frame)
                else:
                    self.frame_ready.emit(self.serial, frame_np)
                
                grab.Release()
        except Exception as e: self.error.emit(self.serial, str(e))
        finally:
            if 'cam' in locals() and cam.IsGrabbing(): cam.StopGrabbing()
            if 'cam' in locals() and cam.IsOpen(): cam.Close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera GUI with Inference")
        self.camera_threads = {}
        self.video_panels = {}
        self.control_panels = {}
        self.model_panels = {}

        self.detect_and_setup_ui()

    def detect_and_setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        control_tabs_widget = QTabWidget()
        control_tabs_widget.setStyleSheet("font-size: 17pt")
        left_layout.addWidget(control_tabs_widget)

        video_grid_widget = QWidget()
        video_grid_layout = QGridLayout(video_grid_widget)

        main_layout.addWidget(left_widget, 20)
        main_layout.addWidget(video_grid_widget, 80)

        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices:
            video_grid_layout.addWidget(QLabel("No Basler cameras detected."))
        else:
            positions = [(i, j) for i in range(2) for j in range(2)]
            for i, dev in enumerate(devices):
                if i >= 4: 
                    break
                serial = dev.GetSerialNumber()

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

                control_panel = ImagePreprocessingTab()
                self.control_panels[serial] = control_panel
                model_panel = ModelPanel()
                self.model_panels[serial] = model_panel

                side_tabs = QTabWidget()
                side_tabs.setTabPosition(QTabWidget.TabPosition.West)
                side_tabs.setStyleSheet("font-size: 17pt")
                side_tabs.addTab(control_panel, "Image")
                side_tabs.addTab(model_panel, "Model")
                control_tabs_widget.addTab(side_tabs, f"Cam {i+1}")

                control_panel.params_changed.connect(lambda exp, bri, con, s=serial: self.push_params_to_thread(s, exp, bri, con))
                control_panel.contrast_mode_changed.connect(lambda use_scurve, s=serial: self.push_contrast_mode_to_thread(s, use_scurve))
                control_panel.load_defaults_requested.connect(control_panel.set_to_defaults)
                model_panel.model_load_requested.connect(lambda s=serial: self.load_model_for_camera(s))
                model_panel.model_stop_requested.connect(lambda s=serial: self.stop_model_for_camera(s))
                model_panel.settings_changed.connect(lambda settings, s=serial: self.push_model_settings_to_thread(s, settings))

        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start All")
        self.stop_btn = QPushButton("Stop All")
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        left_layout.addLayout(buttons_layout)

        if not devices:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
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
                thread = Camera_Thread(devices[serial])
                thread.frame_ready.connect(self.update_frame)
                thread.error.connect(self.show_error)
                thread.model_loaded.connect(self.on_model_loaded) # Connect new signal
                thread.start()
                self.camera_threads[serial] = thread
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        for panel in self.control_panels.values(): panel.setEnabled(True)
        for panel in self.model_panels.values(): panel.setEnabled(True)

    def stop_all_cameras(self):
        for thread in self.camera_threads.values():
            thread.stop()
            thread.wait()
        self.camera_threads.clear()
        for panel in self.video_panels.values():
            panel.setPixmap(QPixmap())
            panel.setText("Disconnected")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        for panel in self.control_panels.values():
            panel.setEnabled(False)
        for panel in self.model_panels.values():
            panel.setEnabled(False)

    def load_model_for_camera(self, serial):
        model_path, _ = QFileDialog.getOpenFileName(self, f"Select Model for {serial}", "", "PyTorch Models (*.pth)")
        if not model_path:
            return

        metadata_path, _ = QFileDialog.getOpenFileName(self, f"Select Metadata for {serial}", "", "JSON files (*.json)")
        if not metadata_path:
            return

        # Pass the file paths to the correct camera thread for loading
        if serial in self.camera_threads:
            print(f"Requesting model load for camera {serial}...")
            self.camera_threads[serial].load_model(model_path, metadata_path)
        else:
            self.show_error("Load Error", "Cannot load model. Camera is not running.")

    def stop_model_for_camera(self, serial):
        if serial in self.camera_threads:
            self.camera_threads[serial].unload_model()
            QMessageBox.information(self, "Model Unloaded", f"Model has been unloaded for camera {serial}.")
    
    def on_model_loaded(self, serial, class_names):
        if serial in self.model_panels:
            self.model_panels[serial].update_class_lists(class_names)
        QMessageBox.information(self, "Model Loaded", f"Model loaded successfully for camera {serial}.")

    def update_frame(self, serial, frame):
        if serial in self.video_panels:
            video_lbl = self.video_panels[serial]
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            video_lbl.setPixmap(pixmap.scaled(video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def push_params_to_thread(self, s, exp, bri, con):
        if s in self.camera_threads: self.camera_threads[s].update_params(exp, bri, con)
    def push_contrast_mode_to_thread(self, s, use_scurve):
        if s in self.camera_threads: self.camera_threads[s].update_contrast_mode(use_scurve)
    def push_model_settings_to_thread(self, s, settings):
        if s in self.camera_threads: self.camera_threads[s].update_model_settings(settings)
    def show_error(self, serial, message): QMessageBox.warning(self, f"Camera Error ({serial})", message)
    def closeEvent(self, event): self.stop_all_cameras(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())