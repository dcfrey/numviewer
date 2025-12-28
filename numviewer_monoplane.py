import sys
import numpy as np
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QSlider, QLabel, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QSizePolicy



class Viewer(QWidget):
    def __init__(self, filepath, transpose):
        super().__init__()

        self.data_original = np.load(filepath, mmap_mode="c")

        if self.data_original.ndim == 3:
            self.data_original = self.data_original.transpose(transpose)
            self.total_slices = self.data_original.shape[0]
        elif self.data_original.ndim == 2:
            self.total_slices = 1
        else:
            raise ValueError("Expected 2D or 3D array")

        # ---- transform state ----
        self.view = "ax"   # "ax", "cor", "sag"
        self.shape = self.data_original.shape
        self.slice_idx = {
            "ax": self.shape[0] // 2,
            "cor": self.shape[1] // 2,
            "sag": self.shape[2] // 2,
        }

        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False

        # ---- UI ----
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored
        )

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_slices - 1)
        self.slider.valueChanged.connect(self.update_image)
        self.update_slider_range()

        # Buttons
        btn_ax = QPushButton("Ax")
        btn_cor = QPushButton("Cor")
        btn_sag = QPushButton("Sag")

        btn_rotate_ccw = QPushButton("Rotate -90°")
        btn_rotate_cw = QPushButton("Rotate +90°")
        btn_flip_x = QPushButton("Flip X")
        btn_flip_y = QPushButton("Flip Y")
        btn_reset = QPushButton("Reset")

        btn_ax.clicked.connect(lambda: self.set_view("ax"))
        btn_cor.clicked.connect(lambda: self.set_view("cor"))
        btn_sag.clicked.connect(lambda: self.set_view("sag"))

        btn_rotate_ccw.clicked.connect(self.rotate_ccw)
        btn_rotate_cw.clicked.connect(self.rotate_cw)
        btn_flip_x.clicked.connect(self.toggle_flip_x)
        btn_flip_y.clicked.connect(self.toggle_flip_y)
        btn_reset.clicked.connect(self.reset_transforms)

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_ax)
        button_layout.addWidget(btn_cor)
        button_layout.addWidget(btn_sag)
        button_layout.addStretch()
        button_layout.addWidget(btn_rotate_ccw)
        button_layout.addWidget(btn_rotate_cw)
        button_layout.addWidget(btn_flip_x)
        button_layout.addWidget(btn_flip_y)
        button_layout.addWidget(btn_reset)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)

        if self.data_original.ndim == 3:
            layout.addWidget(self.slider)

        self.setLayout(layout)
        self.setWindowTitle("NumPy Array Viewer")
        self.resize(800, 800)

        self.update_image(0)
        self.show()

    # ----------------- transforms -----------------
    
    def set_view(self, view):
        self.view = view
#        self.current_slice = 0
        self.update_slider_range()
#        self.slider.setValue(0)
        self.update_image(self.current_slice)

    def update_slider_range(self):
        if self.data_original.ndim != 3:
            return

        if self.view == "ax":
            self.slider.setMaximum(self.shape[0] - 1)
        elif self.view == "cor":
            self.slider.setMaximum(self.shape[1] - 1)
        elif self.view == "sag":
            self.slider.setMaximum(self.shape[2] - 1)

    def rotate_ccw(self):
        self.rot_k = (self.rot_k + 1) % 4
        self.update_image(self.current_slice)
    
    def rotate_cw(self):
        self.rot_k = (self.rot_k - 1) % 4
        self.update_image(self.current_slice)

    def toggle_flip_x(self):
        self.flip_x = not self.flip_x
        self.update_image(self.current_slice)

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y
        self.update_image(self.current_slice)

    def reset_transforms(self):
        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False
        self.update_image(self.current_slice)

    # ----------------- rendering -----------------

    def get_slice(self, idx):
        if self.data_original.ndim == 2:
            img = self.data_original

        else:
            if self.view == "ax":
                img = self.data_original[idx, :, :]
            elif self.view == "cor":
                img = self.data_original[:, idx, :]
            elif self.view == "sag":
                img = self.data_original[:, :, idx]

        if self.rot_k:
            img = np.rot90(img, self.rot_k)
        if self.flip_x:
            img = img[:, ::-1]
        if self.flip_y:
            img = img[::-1, :]

        return img


    def update_image(self, idx):
        self.current_slice = idx
        img = self.get_slice(idx)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        self._set_pixmap(pixmap)

    def _set_pixmap(self, pixmap):
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.image_label.pixmap():
            self._set_pixmap(self.image_label.pixmap())
        super().resizeEvent(event)


# ----------------- entry -----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("i", help="Input .npy file")
    parser.add_argument("-t", default="012", help="Transpose axes")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    transpose = [int(i) for i in args.t]
    viewer = Viewer(args.i, transpose)
    sys.exit(app.exec_())

