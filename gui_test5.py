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
    QMainWindow, QTabWidget, QCheckBox, QGridLayout, QFileDialog, QListWidget
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
	# FIX: PIL.Image.size is (width, height), but torch.interpolate needs (height, width).
	# We must reverse the tuple.
	output_size = (image_size[1], image_size[0]) 

	for mask in masks:
		# Ensure mask is 4D for interpolate: (N, C, H, W)
		if mask.dim() == 2:
			mask = mask.unsqueeze(0).unsqueeze(0)
		elif mask.dim() == 3 and mask.size(0) == 1:
			mask = mask.unsqueeze(0)
            
        # Use the corrected (height, width) output size
		resized_mask = torch_F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
		
		# Squeeze back down to a 2D mask
		binarized_mask = (resized_mask > 0.5).float().squeeze(0).squeeze(0)
		resized_masks.append(binarized_mask.byte())
	return resized_masks

def resize_frame_based_on_resolution(frame, new_height, aspect_ratio):
    if aspect_ratio > 0:
        new_width = int(new_height * aspect_ratio)
        return cv2.resize(frame, (new_width, new_height))
    return frame

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
        self.load_btn.clicked.connect(self._handle_load_button_clicked) # Connect to a new handler
        self.stop_btn.clicked.connect(self._handle_stop_button_clicked) # Connect to a new handler
        # Connect all controls to emit settings change
        for w in [self.conf_slider, self.align_slider, self.bottle_combo, self.logo_combo]:
            if isinstance(w, QSlider): w.valueChanged.connect(self._emit_settings_change)
            else: w.currentIndexChanged.connect(self._emit_settings_change)
    
    def update_class_lists(self, class_names):
        self.class_names = class_names
        
        # Update dropdowns
        self.bottle_combo.clear()
        self.bottle_combo.addItems(class_names)
        self.logo_combo.clear()
        self.logo_combo.addItems(class_names)

        # 1. "Classes to Detect" group: Default to CHECKED
        detect_layout = self.detect_group.layout()
        while detect_layout.count(): detect_layout.takeAt(0).widget().deleteLater()
        self.class_checkboxes.clear()
        for name in class_names:
            cb = QCheckBox(name)
            # Default to checked, but you can add exceptions like 'background'
            cb.setChecked(name.lower() != 'background')
            cb.stateChanged.connect(self._emit_settings_change)
            detect_layout.addWidget(cb)
            self.class_checkboxes[name] = cb

        # 2. "Good" and "Not Good" groups: Default to UNCHECKED
        for group_box, checkbox_dict in [(self.good_group, self.good_checkboxes), (self.ng_group, self.ng_checkboxes)]:
            layout = group_box.layout()
            while layout.count(): layout.takeAt(0).widget().deleteLater()
            checkbox_dict.clear()
            for name in class_names:
                cb = QCheckBox(name)
                cb.setChecked(False)  # Set to unchecked by default
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
    
    def _handle_load_button_clicked(self):
        self.load_btn.setProperty("active_button", "true")
        self.stop_btn.setProperty("active_button", "false")
        self.style().polish(self.load_btn)
        self.style().polish(self.stop_btn)
        self.model_load_requested.emit()
        
    def _handle_stop_button_clicked(self):
        self.stop_btn.setProperty("active_button", "true")
        self.load_btn.setProperty("active_button", "false")
        self.style().polish(self.stop_btn)
        self.style().polish(self.load_btn)
        self.model_stop_requested.emit()
        
    def reset_model_button_states(self):
        self.load_btn.setProperty("active_button", "false")
        self.stop_btn.setProperty("active_button", "false")
        self.style().polish(self.load_btn)
        self.style().polish(self.stop_btn)
        
class ROIPanel(QWidget):
    roi_changed = Signal(dict)
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(10, 100)
        self.width_slider.setValue(100)
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(10, 100)
        self.height_slider.setValue(100)
        self.shiftx_slider = QSlider(Qt.Orientation.Horizontal)
        self.shiftx_slider.setRange(-100, 100)
        self.shiftx_slider.setValue(0)
        self.shifty_slider = QSlider(Qt.Orientation.Horizontal)
        self.shifty_slider.setRange(-100, 100)
        self.shifty_slider.setValue(0)
        form_layout.addRow("Width (%)", self.width_slider)
        form_layout.addRow("Height (%)", self.height_slider)
        form_layout.addRow("Shift X", self.shiftx_slider)
        form_layout.addRow("Shift Y", self.shifty_slider)
        main_layout.addLayout(form_layout)
        main_layout.addStretch()
        for w in [self.width_slider, self.height_slider, self.shiftx_slider, self.shifty_slider]:
            w.valueChanged.connect(self._emit_change)

    def _emit_change(self):
        settings = {'width': self.width_slider.value()/100.0, 'height': self.height_slider.value()/100.0, 'shift_x': self.shiftx_slider.value()/100.0, 'shift_y': self.shifty_slider.value()/100.0}
        self.roi_changed.emit(settings)
        
class TransformPanel(QWidget):
    transform_changed = Signal(dict)
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.rotate_combo = QComboBox()
        self.rotate_combo.addItems(["0", "90", "180", "-90"])
        self.flip_ud_cb = QCheckBox("Flip Up/Down")
        self.flip_lr_cb = QCheckBox("Flip Left/Right")
        form_layout.addRow("Rotation:", self.rotate_combo)
        form_layout.addRow(self.flip_ud_cb)
        form_layout.addRow(self.flip_lr_cb)
        main_layout.addLayout(form_layout)
        main_layout.addStretch()
        self.rotate_combo.currentIndexChanged.connect(self._emit_change)
        self.flip_ud_cb.stateChanged.connect(self._emit_change)
        self.flip_lr_cb.stateChanged.connect(self._emit_change)
    def _emit_change(self):
        settings = {'rotate': int(self.rotate_combo.currentText()), 'flip_ud': self.flip_ud_cb.isChecked(), 'flip_lr': self.flip_lr_cb.isChecked()}
        self.transform_changed.emit(settings)
        
class MediaCapturePanel(QGroupBox):
    capture_image_requested = Signal(str)
    toggle_recording_requested = Signal(str)
    def __init__(self):
        super().__init__("Media Capture (All Feeds)")
        main_layout = QVBoxLayout(self)
        self.folder_edit = QLineEdit("capture_session")
        self.folder_edit.setPlaceholderText("Enter folder name...")
        self.history_list = QListWidget()
        self.capture_img_btn = QPushButton("Capture Image")
        self.record_btn = QPushButton("Start Recording")
        main_layout.addWidget(self.folder_edit)
        main_layout.addWidget(self.history_list)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.capture_img_btn)
        btn_layout.addWidget(self.record_btn)
        main_layout.addLayout(btn_layout)
        self.capture_img_btn.clicked.connect(lambda: self.capture_image_requested.emit(self.folder_edit.text()))
        self.record_btn.clicked.connect(lambda: self.toggle_recording_requested.emit(self.folder_edit.text()))
    def add_history(self, message): self.history_list.insertItem(0, message)
    def set_recording_state(self, is_recording): self.record_btn.setText("Stop Recording" if is_recording else "Start Recording")

class Camera_Thread(QThread):
    frame_ready = Signal(str, np.ndarray)
    error = Signal(str, str)
    model_loaded = Signal(str, list) # serial, class_names
    image_saved = Signal(str, str)

    def __init__(self, device_info):
        super().__init__()
        self.device_info = device_info
        self.serial = device_info.GetSerialNumber()
        self._running = False
        self.exposure, self.brightness, self.contrast = 1000.0, 0.0, 0.0
        self.use_scurve = False
        self.aspect_ratio = 16 / 9  # Default aspect ratio
        self.processing_height = 720  # Default processing height

        # Model-related attributes
        self.model, self.model_metadata = None, None
        self.inference_enabled = False
        self.model_settings = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lock = threading.Lock()
        self.transform_settings = {'rotate': 0, 'flip_ud': False, 'flip_lr': False}
        self.capture_lock = threading.Lock()
        self._capture_folder = None
        
    def request_capture(self, folder_name):
        with self.capture_lock:
            self._capture_folder = folder_name
        
    def set_processing_resolution(self, height):
        self.processing_height = height
        
    def update_transform_settings(self, settings):
        self.transform_settings = settings

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
                print(f"Model metadata loaded: {self.model_metadata}")

            # STEP 2: Rebuild the model architecture based on metadata
            model_type = self.model_metadata.get("modelType", 0)
            # Use num_classes from metadata, provide a fallback for safety
            num_classes = len(self.model_metadata.get("class_names", ["BG", "FG"]))
            print(f"Model type: {model_type}, Number of classes: {num_classes}")

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
                # feature_channels = [mobilenet_backbone[i][-1].out_channels for i in layers_to_extract]
                feature_channels = [mobilenet_backbone[i].out_channels for i in layers_to_extract]

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
                # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                #     featmap_names=['0'], output_size=14, sampling_ratio=2
                # )
                
                model = MaskRCNN(
                    backbone=backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    # mask_roi_pool=mask_roi_pooler
                )
                # in_features = model.roi_heads.box_predictor.cls_score.in_features
                # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                
            else:
                raise ValueError(f"Unsupported modelType {model_type} in metadata.")

            # STEP 3: Load the state_dict
            state_dict = torch.load(model_path, map_location=self.device)
            corrected_state_dict = { key.replace("roi_heads.box_predictor.predictor.", "roi_heads.box_predictor."): value for key, value in state_dict.items() }
            model.load_state_dict(corrected_state_dict)
            # model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {model_path}")

            # STEP 4: Move model to the correct device and set to evaluation mode
            model.to(self.device).eval()
            print(f"Model moved to {self.device} and set to evaluation mode.")
            
            # Use .half() for faster inference if your GPU supports it
            if self.device.type == 'cuda':
                model.half()
                
            print("Model is ready for inference.")

            # STEP 5: Assign the loaded model and enable inference
            with self.model_lock:
                self.model = model
                self.inference_enabled = True

            # Emit signal to notify the main window that the model is ready
            self.model_loaded.emit(self.serial, self.model_metadata.get("class_names", []))
            print(f"Model loaded and ready for camera {self.serial}: {self.model_metadata.get('class_names', [])}")

        except Exception as e:
            # Emit an error signal if anything goes wrong
            self.error.emit(self.serial, f"Failed to load model: {e}")
            print(f"Error loading model for camera {self.serial}: {e}")

    def unload_model(self):
        with self.model_lock:
            self.model, self.model_metadata, self.inference_enabled = None, None, False

    def run(self):
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info))
            cam.Open()
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(30)
            fmt = pylon.ImageFormatConverter()
            fmt.OutputPixelFormat = pylon.PixelType_BGR8packed
            
            print("Starting camera grabbing...")
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._running = True
            is_first_frame = True
            
            prev_time = time.time()
            frame_count = 0
            display_fps = 0
            
            font_face_fps = cv2.FONT_HERSHEY_SIMPLEX
            font_face_status = cv2.FONT_HERSHEY_SIMPLEX
            font_color_status = (255, 255, 255)
            color_ok = (0, 150, 0)
            color_ng = (0, 0, 200)

            while self._running and cam.IsGrabbing():
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if not grab.GrabSucceeded(): 
                    continue
                
                folder_to_capture = None
                with self.capture_lock:
                    if self._capture_folder:
                        folder_to_capture = self._capture_folder
                        self._capture_folder = None # Reset request
                        
                # This block will be used to save the overlay later if needed
                saved_full_res_info = {}
                if folder_to_capture:
                    # 1. Save the full, unprocessed image
                    full_res_frame_bgr = fmt.Convert(grab).GetArray()
                    base_dir = os.path.join(os.getcwd(), "media", folder_to_capture.strip(), self.serial)
                    os.makedirs(base_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds
                    full_res_path = os.path.join(base_dir, f"capture_{timestamp}_full_res.png")
                    cv2.imwrite(full_res_path, full_res_frame_bgr)
                    self.image_saved.emit(self.serial, f"Saved full-res: {os.path.relpath(full_res_path)}")
                    saved_full_res_info = {"base_dir": base_dir, "timestamp": timestamp}
                    
                try:
                    cam.BslContrastMode.SetValue("SCurve" if self.use_scurve else "Linear")
                    cam.ExposureTime.SetValue(self.exposure)
                    cam.BslBrightness.SetValue(self.brightness)
                    cam.BslContrast.SetValue(self.contrast)
                except Exception: 
                    pass
                
                with self.model_lock:
                    inference_enabled_local = self.inference_enabled
                    model_local = self.model
                    model_metadata_local = self.model_metadata

                if inference_enabled_local and model_local and model_metadata_local:
                    frame_np = fmt.Convert(grab).GetArray()
                    # Convert and resize the frame
                    if is_first_frame:
                        h, w, _ = frame_np.shape
                        print(f"Initial frame resolution for {self.serial}: {w}x{h}")
                        if h > 0: self.aspect_ratio = w / h
                        is_first_frame = False
                        
                    frame_np = resize_frame_based_on_resolution(frame_np, self.processing_height, self.aspect_ratio)
                    
                    # Transformation
                    rot, ud, lr = self.transform_settings['rotate'], self.transform_settings['flip_ud'], self.transform_settings['flip_lr']
                    if rot == 90: frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180: frame_np = cv2.rotate(frame_np, cv2.ROTATE_180)
                    elif rot == -90: frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if ud: frame_np = cv2.flip(frame_np, 0)
                    if lr: frame_np = cv2.flip(frame_np, 1)
                    
                    # Inference and overlay
                    pil_img = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                    img_tensor = F.to_tensor(pil_img).to(self.device)
                    if self.device.type == 'cuda':
                        img_tensor = img_tensor.half()

                    with torch.no_grad(): 
                        predictions = self.model([img_tensor])
                        
                    pred = predictions[0]
                    conf = self.model_settings.get("ConfidenceThreshold", 0.5)
                    idx = pred["scores"] > conf
                    
                    filtered_boxes, filtered_scores, filtered_labels_idx, filtered_masks = filter_overlapping_detections(
                        pred["boxes"][idx], pred["scores"][idx], pred["labels"][idx], pred["masks"][idx].squeeze(1)
                    )
                    
                    class_names = model_metadata_local.get("class_names", [])
                    enabled_classes = self.model_settings.get("ClassesToDetect", {})
                    
                    # Create temporary lists to hold the filtered results
                    final_boxes, final_masks, final_labels_list = [], [], []
                    # Iterate through the predictions and keep only the ones that are checked in the UI
                    for i in range(len(filtered_labels_idx)):
                        label_index = filtered_labels_idx[i].item()
                        class_name = class_names[label_index]
                        
                        # Only include the detection if its class is enabled (or if no setting exists)
                        if enabled_classes.get(class_name, True): # Default to True if not found
                            final_boxes.append(filtered_boxes[i])
                            final_masks.append(filtered_masks[i])
                            final_labels_list.append(f"{class_name} {filtered_scores[i]:.2f}")

                    # Convert lists back to tensors where necessary
                    filtered_boxes = torch.stack(final_boxes) if final_boxes else torch.empty((0, 4))
                    resized_masks = resize_and_binarize_masks(final_masks, pil_img.size)
                    
                    class_colors = model_metadata_local.get("class_colors", {})
                    overlay_img = overlay_masks_boxes_labels_predict(pil_img, resized_masks, filtered_boxes, class_colors, final_labels_list)
                    output_frame = cv2.cvtColor(np.array(overlay_img), cv2.COLOR_RGB2BGR) # This is now our base frame to draw on
                    
                else:
                    frame_np = fmt.Convert(grab).GetArray()
                    if is_first_frame: # Ensure aspect ratio is set on first frame
                        h, w, _ = frame_np.shape
                        print(f"Initial frame resolution for {self.serial}: {w}x{h}")
                        if h > 0: self.aspect_ratio = w / h
                        is_first_frame = False
                    output_frame = resize_frame_based_on_resolution(frame_np, self.processing_height, self.aspect_ratio)

                status = None
                if inference_enabled_local and model_local:
                    detected_class_names = [label.split()[0] for label in final_labels_list]
                    if detected_class_names: # Proceed only if there are detections
                        not_good_classes = self.model_settings.get("NotGood", [])
                        good_classes = self.model_settings.get("Good", [])
                        if any(name in not_good_classes for name in detected_class_names):
                            status = "NG"
                        elif any(name in good_classes for name in detected_class_names):
                            status = "OK"

                if status:
                    text = "OK" if status == "OK" else "NG"
                    color = color_ok if status == "OK" else color_ng
                    (text_w, text_h), baseline = cv2.getTextSize(text, font_face_status, 1.8, 3)
                    margin = 20
                    frame_h, frame_w, _ = output_frame.shape
                    rect_x1, rect_y1 = frame_w - text_w - 2 * margin, margin
                    rect_x2, rect_y2 = frame_w - margin, margin + text_h + baseline + margin
                    text_x, text_y = frame_w - text_w - margin - (margin // 2), margin + text_h + (baseline // 2)
                    cv2.rectangle(output_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
                    cv2.putText(output_frame, text, (text_x, text_y), font_face_status, 1.8, font_color_status, 3, cv2.LINE_AA)
                    
                if folder_to_capture and self.inference_enabled:
                    info = saved_full_res_info
                    overlay_path = os.path.join(info["base_dir"], f"capture_{info['timestamp']}_overlay.png")
                    cv2.imwrite(overlay_path, output_frame)
                    self.image_saved.emit(self.serial, f"Saved overlay: {os.path.relpath(overlay_path)}")
                
                frame_count += 1
                current_time = time.time()
                if (current_time - prev_time) > 1.0:
                    display_fps = frame_count / (current_time - prev_time)
                    prev_time, frame_count = current_time, 0
                
                cv2.putText(output_frame, f"FPS: {display_fps:.2f}", (10, 30), font_face_fps, 1.2, (0, 255, 0), 2)
                self.frame_ready.emit(self.serial, output_frame)
                
            grab.Release()
        except Exception as e: 
            self.error.emit(self.serial, str(e))
        finally:
            if 'cam' in locals() and cam.IsOpen():
                if cam.IsGrabbing():
                    cam.StopGrabbing()
                cam.Close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera GUI with Inference")
        self.camera_threads = {}
        self.video_panels = {}
        self.control_panels = {}
        self.model_panels = {}
        self.roi_panels = {}
        self.transform_panels = {}
        self.media_capture_panels = {}
        self.is_recording = {}
        self.video_writers = {}
        
        self.processing_height = 720  # Default processing height
        self.resolution_options = {
            "360 (nHD)": 360,
            "480 (FWVGA)": 480,
            "540 (qHD)": 540,
            "720 (HD)": 720,
            "1080 (Full HD)": 1080,
            "1440 (QHD)": 1440
        }
        
        self.setStyleSheet("""
            /* Style for QTabWidget tabs when selected */
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #C2C7CB;
            }

            QTabBar::tab {
                background: lightgray;
                border: 1px solid gray;
                border-bottom-color: #C2C7CB; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 25px;
                padding: 8px;
                font-size: 17pt; /* Keep your existing font size for inactive */
                color: black; /* Default text color for tabs */
            }

            QTabBar::tab:selected {
                background: #000055; /* Dark blue for active tab */
                color: white; /* White text for active tab */
                font-weight: bold; /* Bold text for active tab */
                border-color: #000055;
                border-bottom-color: #000055; /* Same as selected tab color */
            }

            QTabBar::tab:hover:!selected {
                background: #E0E0E0; /* Slightly darker gray on hover for inactive tabs */
            }

            /* Base style for QPushButton (default state) */
            QPushButton {
                background-color: lightgray; /* Default background */
                color: black; /* Default text color */
                border: 1px solid gray;
                border-radius: 4px;
                padding: 6px;
            }

            /* Style for QPushButton when it has the 'active_button' property set to "true" */
            QPushButton[active_button="true"] {
                background-color: #000055; /* Dark blue for active button */
                color: black; /* White text for active button */
                font-weight: bold; /* Bold text for active button */
                border: 1px solid #000055; /* Darker border for active state */
            }

            /* Style for an active QPushButton when pressed */
            QPushButton[active_button="true"]:pressed {
                background-color: #000033; /* Even darker blue when active and pressed */
                border: 1px solid #000011;
            }

            /* Style for an inactive QPushButton when pressed */
            QPushButton:!active_button:pressed {
                background-color: #000033; /* Dark blue when inactive and pressed */
                color: white; /* White text when pressed */
                font-weight: bold; /* Bold text when pressed */
                border: 1px solid #000011;
            }

            /* Style for QPushButton when hovered (applies to any button not active or pressed) */
            QPushButton:hover:!active_button:!pressed {
                background-color: #E0E0E0; /* Lighter gray on hover for inactive buttons */
            }

            /* Style for QSlider handle (the movable part) */
            QSlider::handle:horizontal {
                background: #000055; /* Dark blue for slider handle */
                width: 18px;
                margin: -2px 0; /* expand outside the groove */
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #000088; /* Slightly lighter blue on hover */
            }
            QSlider::handle:horizontal:pressed {
                background: #000033; /* Even darker blue when pressed */
            }

            /* Optional: Style the slider groove */
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #000055;
                height: 10px;
                border-radius: 4px;
            }
        """)


        self.detect_and_setup_ui()

    def detect_and_setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        top_control_bar = QWidget()
        top_control_layout = QHBoxLayout(top_control_bar)
        top_control_layout.setContentsMargins(0,0,0,0)
        left_layout.addWidget(top_control_bar)
        
        control_tabs_widget = QTabWidget()
        control_tabs_widget.setStyleSheet("font-size: 17pt")
        left_layout.addWidget(control_tabs_widget)

        video_grid_widget = QWidget()
        video_grid_layout = QGridLayout(video_grid_widget)
        
        top_control_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(self.resolution_options.keys())
        self.resolution_combo.setCurrentText("720 (HD)")
        self.resolution_combo.currentTextChanged.connect(self.update_resolution)
        top_control_layout.addWidget(self.resolution_combo)
        top_control_layout.addStretch()
        
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        top_control_layout.addLayout(buttons_layout)

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
                roi_panel = ROIPanel()
                self.roi_panels[serial] = roi_panel
                transform_panel = TransformPanel()
                self.transform_panels[serial] = transform_panel
                media_panel = MediaCapturePanel()
                self.media_capture_panels[serial] = media_panel

                side_tabs = QTabWidget()
                side_tabs.setTabPosition(QTabWidget.TabPosition.West)
                side_tabs.setStyleSheet("font-size: 17pt")
                side_tabs.addTab(model_panel, "Model")
                side_tabs.addTab(control_panel, "Image")
                side_tabs.addTab(roi_panel, "ROI")
                side_tabs.addTab(transform_panel, "Transform")
                side_tabs.addTab(media_panel, "Capture")
                control_tabs_widget.addTab(side_tabs, f"Cam {i+1}")

                control_panel.params_changed.connect(lambda exp, bri, con, s=serial: self.push_params_to_thread(s, exp, bri, con))
                control_panel.contrast_mode_changed.connect(lambda use_scurve, s=serial: self.push_contrast_mode_to_thread(s, use_scurve))
                control_panel.load_defaults_requested.connect(control_panel.set_to_defaults)
                model_panel.model_load_requested.connect(lambda s=serial: self.load_model_for_camera(s))
                model_panel.model_stop_requested.connect(lambda s=serial: self.stop_model_for_camera(s))
                model_panel.settings_changed.connect(lambda settings, s=serial: self.push_model_settings_to_thread(s, settings))
                transform_panel.transform_changed.connect(lambda settings, s=serial: self.camera_threads[s].update_transform_settings(settings))
                roi_panel.roi_changed.connect(lambda settings, s=serial: self.camera_threads[s].update_roi_settings(settings))
                media_panel.capture_image_requested.connect(lambda folder, s=serial: self.capture_single_image(folder, s))
                media_panel.toggle_recording_requested.connect(lambda folder, s=serial: self.toggle_single_recording(folder, s))

        if not devices:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
        else:
            self.start_btn.clicked.connect(self.start_all_cameras)
            self.stop_btn.clicked.connect(self.stop_all_cameras)
            self.stop_btn.setEnabled(False)
            for panel in self.control_panels.values(): panel.setEnabled(False)
            for panel in self.model_panels.values(): panel.setEnabled(False)
            for panel in self.roi_panels.values(): panel.setEnabled(False)
            for panel in self.transform_panels.values(): panel.setEnabled(False)

    def update_resolution(self, text):
        self.processing_height = self.resolution_options.get(text, 720)
        print(f"Processing resolution height set to: {self.processing_height}px")
        # Push the new resolution to all active camera threads
        for thread in self.camera_threads.values():
            thread.set_processing_resolution(self.processing_height)

    def start_all_cameras(self):
        if self.camera_threads: return
        devices = {dev.GetSerialNumber(): dev for dev in pylon.TlFactory.GetInstance().EnumerateDevices()}
        for serial in self.control_panels.keys():
            if serial in devices:
                thread = Camera_Thread(devices[serial])
                thread.set_processing_resolution(self.processing_height)
                thread.frame_ready.connect(self.update_frame)
                thread.error.connect(self.show_error)
                thread.model_loaded.connect(self.on_model_loaded) # Connect new signal
                thread.image_saved.connect(self.on_image_saved)
                thread.start()
                self.camera_threads[serial] = thread
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Set the 'active_button' property for the start button
        self.start_btn.setProperty("active_button", "true")
        self.stop_btn.setProperty("active_button", "false") # Ensure stop button is inactive
        self.style().polish(self.start_btn) # Re-apply stylesheet
        self.style().polish(self.stop_btn) # Re-apply stylesheet
        
        for panel in self.control_panels.values(): panel.setEnabled(True)
        for panel in self.model_panels.values(): panel.setEnabled(True)
        for panel in self.roi_panels.values(): panel.setEnabled(True)
        for panel in self.transform_panels.values(): panel.setEnabled(True)
        for panel in self.media_capture_panels.values(): panel.setEnabled(True)

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
        
        # Set the 'active_button' property for the stop button
        self.stop_btn.setProperty("active_button", "true")
        self.start_btn.setProperty("active_button", "false") # Ensure start button is inactive
        self.style().polish(self.stop_btn) # Re-apply stylesheet
        self.style().polish(self.start_btn) # Re-apply stylesheet
        
        for panel in self.control_panels.values():
            panel.setEnabled(False)
        for panel in self.model_panels.values():
            panel.setEnabled(False)
        for panel in self.roi_panels.values():
            panel.setEnabled(False)
        for panel in self.transform_panels.values():
            panel.setEnabled(False)
        for panel in self.media_capture_panels.values():
            panel.setEnabled(False)

    def load_model_for_camera(self, serial):
        model_path, _ = QFileDialog.getOpenFileName(self, f"Select Model for {serial}", "", "PyTorch Models (*.pth)")
        if not model_path:
            if serial in self.model_panels:
                self.model_panels[serial].reset_model_button_states()
            return

        metadata_path, _ = QFileDialog.getOpenFileName(self, f"Select Metadata for {serial}", "", "JSON files (*.json)")
        if not metadata_path:
            if serial in self.model_panels:
                self.model_panels[serial].reset_model_button_states()
            return

        # Pass the file paths to the correct camera thread for loading
        if serial in self.camera_threads:
            print(f"Requesting model load for camera {serial}...")
            self.camera_threads[serial].load_model(model_path, metadata_path)
        else:
            self.show_error("Load Error", "Cannot load model. Camera is not running.")
            if serial in self.model_panels:
                self.model_panels[serial].reset_model_button_states()

    def stop_model_for_camera(self, serial):
        if serial in self.camera_threads:
            self.camera_threads[serial].unload_model()
            QMessageBox.information(self, "Model Unloaded", f"Model has been unloaded for camera {serial}.")
            if serial in self.model_panels:
                self.model_panels[serial].reset_model_button_states() # This is the missing piece
    
    def on_model_loaded(self, serial, class_names):
        if serial in self.model_panels:
            self.model_panels[serial].update_class_lists(class_names)
            # The _handle_load_button_clicked already sets it, but good to ensure
            # self.model_panels[serial].load_btn.setProperty("active_button", "true")
            # self.model_panels[serial].stop_btn.setProperty("active_button", "false")
            # self.model_panels[serial].style().polish(self.model_panels[serial].load_btn)
            # self.model_panels[serial].style().polish(self.model_panels[serial].stop_btn)
        QMessageBox.information(self, "Model Loaded", f"Model loaded successfully for camera {serial}.")

    # def update_frame(self, serial, frame):
    #     if serial in self.video_panels:
    #         video_lbl = self.video_panels[serial]
    #         h, w, ch = frame.shape
    #         q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
    #         pixmap = QPixmap.fromImage(q_img)
    #         video_lbl.setPixmap(pixmap.scaled(video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            
    def update_frame(self, serial, frame):
        if serial in self.video_panels:
            # Update the GUI
            video_lbl = self.video_panels[serial]
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            video_lbl.setPixmap(pixmap.scaled(video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            # Write frame to video file if recording is active for this serial
            if self.is_recording.get(serial, False) and serial in self.video_writers:
                # The video writer needs a specific frame dimension. Let's resize the frame to match.
                writer = self.video_writers[serial]
                height, width, _ = frame.shape
                
                # Get the expected dimensions from the writer
                expected_width = int(writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                expected_height = int(writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if (width, height) != (expected_width, expected_height):
                    # Resize frame to match the video writer's dimensions
                    frame_to_write = cv2.resize(frame, (expected_width, expected_height))
                else:
                    frame_to_write = frame

                writer.write(frame_to_write)
            
    def capture_single_image(self, folder_name, serial):
        if not self.camera_threads:
            QMessageBox.warning(self, "Warning", "Cameras are not running.")
            return
        if not folder_name.strip():
            QMessageBox.warning(self, "Warning", "Please enter a folder name.")
            return

        self.camera_threads[serial].request_capture(folder_name.strip())
        
    def on_image_saved(self, serial, message):
        if serial in self.media_capture_panels:
            self.media_capture_panels[serial].add_history(message)

    def toggle_single_recording(self, folder_name, serial):
        is_currently_recording = self.is_recording.get(serial, False)

        media_panel = self.media_capture_panels.get(serial)

        if is_currently_recording:
            # Stop recording for this specific camera
            if serial in self.video_writers:
                self.video_writers[serial].release()
                del self.video_writers[serial]
            self.is_recording[serial] = False
            if media_panel:
                media_panel.set_recording_state(False)
                media_panel.add_history("Recording stopped.")
        else:
            # Start recording for this specific camera
            if not self.camera_threads:
                QMessageBox.warning(self, "Warning", "Cameras are not running.")
                return
            if not folder_name.strip():
                QMessageBox.warning(self, "Warning", "Please enter a folder name.")
                return

            panel = self.video_panels.get(serial)
            if panel:
                pixmap = panel.pixmap()
                if pixmap and not pixmap.isNull():
                    base_dir = os.path.join(os.getcwd(), "media", folder_name.strip(), serial)
                    os.makedirs(base_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = os.path.join(base_dir, f"video_{timestamp}.avi")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 10.0  # Or another desired framerate
                    dims = (pixmap.width(), pixmap.height())
                    
                    self.video_writers[serial] = cv2.VideoWriter(file_path, fourcc, fps, dims)
                    self.is_recording[serial] = True
                    
                    if media_panel:
                        media_panel.set_recording_state(True)
                        media_panel.add_history(f"Recording to: {os.path.relpath(file_path)}")
            
    def push_roi_settings_to_thread(self, s, settings):
        if s in self.camera_threads: 
            self.camera_threads[s].update_roi_settings(settings)
            if s in self.roi_panels:
                self.roi_panels[s].update_roi_settings(settings)
    
    def push_params_to_thread(self, s, exp, bri, con):
        if s in self.camera_threads: 
            self.camera_threads[s].update_params(exp, bri, con)
            
    def push_contrast_mode_to_thread(self, s, use_scurve):
        if s in self.camera_threads: 
            self.camera_threads[s].update_contrast_mode(use_scurve)
            
    def push_model_settings_to_thread(self, s, settings):
        if s in self.camera_threads: 
            self.camera_threads[s].update_model_settings(settings)
            
    def push_transform_settings_to_thread(self, s, settings):
        if s in self.camera_threads: 
            self.camera_threads[s].update_transform_settings(settings)
            
    def show_error(self, serial, message): 
        QMessageBox.warning(self, f"Camera Error ({serial})", message)
        if "Failed to load model" in message or "Cannot load model" in message:
            if serial in self.model_panels: 
                self.model_panels[serial].reset_model_button_states()
        
    def closeEvent(self, event): 
        self.stop_all_cameras(); 
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())