import sys
import numpy as np
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QSlider, QLabel, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

# ----------------- SliceView -----------------
class SliceView(QWidget):
    def __init__(self, viewer, plane):
        super().__init__()
        self.viewer = viewer        # Reference to main Viewer
        self.plane = plane          # "ax", "cor", "sag"

        self.idx = 0                # Current slice index
        
        self._orig_qimage = None

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(200, 200)  # or any reasonable value

        # Slider + counter
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slice_text = QLabel()

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider, stretch=1)
        slider_layout.addWidget(self.slice_text)

        # Buttons
        btn_rotate_ccw = QPushButton("-90°")
        btn_rotate_cw = QPushButton("+90°")
        btn_flip_x = QPushButton("-X")
        btn_flip_y = QPushButton("-Y")
        btn_reset = QPushButton("Reset")

        btn_rotate_ccw.clicked.connect(self.viewer.rotate_ccw)
        btn_rotate_cw.clicked.connect(self.viewer.rotate_cw)
        btn_flip_x.clicked.connect(self.viewer.toggle_flip_x)
        btn_flip_y.clicked.connect(self.viewer.toggle_flip_y)
        btn_reset.clicked.connect(self.viewer.reset_transforms)

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_rotate_ccw)
        button_layout.addWidget(btn_rotate_cw)
        button_layout.addWidget(btn_flip_x)
        button_layout.addWidget(btn_flip_y)
        button_layout.addWidget(btn_reset)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=1)  # <-- stretch=1 for image
        layout.addLayout(slider_layout, stretch=0)     # buttons/sliders: stretch=0
        layout.addLayout(button_layout, stretch=0)

        self.setLayout(layout)
        self.update_slider_range()
        self.update_image()

    def slider_changed(self, value):
        self.idx = value
        self.viewer.slice_idx[self.plane] = value
        self.update_image()

    def update_slider_range(self):
        shape = self.viewer.vol_transformed.shape
        if self.plane == "ax":
            self.slider.setMaximum(shape[0] - 1)
        elif self.plane == "cor":
            self.slider.setMaximum(shape[1] - 1)
        elif self.plane == "sag":
            self.slider.setMaximum(shape[2] - 1)

    def set_slice(self, idx):
        self.idx = idx
        self.slider.setValue(idx)
        self.update_image()

    def update_image(self):
        vol = self.viewer.vol_transformed
        rot_k = self.viewer.rot_k
        flip_x = self.viewer.flip_x
        flip_y = self.viewer.flip_y

        # Extract slice
        if self.plane == "ax":
            img = vol[self.idx, :, :]
        elif self.plane == "cor":
            img = vol[:, self.idx, :]
        elif self.plane == "sag":
            img = vol[:, :, self.idx]

        # Apply global transforms
        if rot_k:
            img = np.rot90(img, rot_k)
        if flip_x:
            img = img[:, ::-1]
        if flip_y:
            img = img[::-1, :]

        # Normalize for display
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self._orig_qimage = qimg  # store for resizing
        pixmap = QPixmap.fromImage(qimg)
        self._set_pixmap(pixmap)

        # Update slice counter
        self.slice_text.setText(f"{self.idx}/{self.slider.maximum()}")

    def _set_pixmap(self, pixmap):
        if not self._orig_qimage:
            return
        label_w = max(1, self.image_label.width())
        label_h = max(1, self.image_label.height())
        pixmap = QPixmap.fromImage(self._orig_qimage)
        scaled = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.image_label.pixmap():
            self._set_pixmap(self.image_label.pixmap())
        super().resizeEvent(event)

# ----------------- Viewer -----------------
class Viewer(QWidget):
    def __init__(self, filepath, transpose):
        super().__init__()

        self.vol_original = np.load(filepath, mmap_mode="c")
        self.vol_transformed = self.vol_original.transpose(transpose)
        self.shape = self.vol_transformed.shape

        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False

        # Slice indices per plane
        self.slice_idx = {
            "ax": self.shape[0] // 2,
            "cor": self.shape[1] // 2,
            "sag": self.shape[2] // 2,
        }

        # Create SliceViews
        self.slice_views = {}
        planes = ["ax", "cor", "sag"]
        h_layout = QHBoxLayout()
        for plane in planes:
            view = SliceView(self, plane)
            view.set_slice(self.slice_idx[plane])
            self.slice_views[plane] = view
            h_layout.addWidget(view)

        # Transpose buttons
        btn_transpose_012 = QPushButton("012")
        btn_transpose_120 = QPushButton("120")
        btn_transpose_201 = QPushButton("201")
        btn_transpose_021 = QPushButton("021")
        btn_transpose_102 = QPushButton("102")
        btn_transpose_210 = QPushButton("210")
        btn_transpose_012.clicked.connect(lambda: self.apply_transpose([0,1,2]))
        btn_transpose_120.clicked.connect(lambda: self.apply_transpose([1,2,0]))
        btn_transpose_201.clicked.connect(lambda: self.apply_transpose([2,0,1]))
        btn_transpose_021.clicked.connect(lambda: self.apply_transpose([0,2,1]))
        btn_transpose_210.clicked.connect(lambda: self.apply_transpose([2,1,0]))
        btn_transpose_102.clicked.connect(lambda: self.apply_transpose([1,0,2]))
        transpose_layout = QHBoxLayout()
        transpose_layout.addWidget(btn_transpose_012)
        transpose_layout.addWidget(btn_transpose_120)
        transpose_layout.addWidget(btn_transpose_201)
        transpose_layout.addWidget(btn_transpose_021)
        transpose_layout.addWidget(btn_transpose_210)
        transpose_layout.addWidget(btn_transpose_102)

        main_layout = QVBoxLayout()
        main_layout.addLayout(transpose_layout)
        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)
        self.setWindowTitle("NumPy Viewer")
        self.resize(1200, 600)

    # ----------------- global transforms -----------------
    def rotate_ccw(self):
        self.rot_k = (self.rot_k + 1) % 4
        self.update_all_views()

    def rotate_cw(self):
        self.rot_k = (self.rot_k - 1) % 4
        self.update_all_views()

    def toggle_flip_x(self):
        self.flip_x = not self.flip_x
        self.update_all_views()

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y
        self.update_all_views()

    def reset_transforms(self):
        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False
        self.update_all_views()

    def apply_transpose(self, axes):
        self.vol_transformed = self.vol_original.transpose(axes)
        self.shape = self.vol_transformed.shape
        # Reset slices
        self.slice_idx = {
            "ax": self.shape[0] // 2,
            "cor": self.shape[1] // 2,
            "sag": self.shape[2] // 2,
        }
        self.update_all_views()

    def update_all_views(self):
        for plane, view in self.slice_views.items():
            view.update_slider_range()
            view.set_slice(self.slice_idx[plane])

# ----------------- entry -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("i", help="Input .npy file")
    parser.add_argument("-t", default="012", help="Initial transpose axes")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    transpose = [int(i) for i in args.t]
    viewer = Viewer(args.i, transpose)
    viewer.show()
    sys.exit(app.exec_())

