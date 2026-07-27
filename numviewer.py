import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QFont, QFontMetrics, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QSlider, QLabel, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QSizePolicy, QDoubleSpinBox, 
    QMessageBox, QToolBar, QAction, QComboBox, QShortcut, 
    QDialog, QTextEdit, QTextBrowser
)


DEFAULT_TRANSPOSE = [0, 1, 2]
PMIN, PMAX = 20, 97.5


class HelpDialog(QDialog):
    def __init__(self, markdown_text, parent=None):
        super().__init__(parent)

        self.setWindowTitle("NumViewer Help")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setMarkdown(markdown_text)

        self.text.setFrameStyle(QTextBrowser.NoFrame)

        self.text.setStyleSheet("""
            QTextBrowser {
                background-color: #111;
                color: #ddd;
                border: none;
                font-size: 12px;
            }
        """)

        layout.addWidget(self.text)

        # --- auto-size to content ---
        doc = self.text.document()
        doc.setTextWidth(500)

        height = int(doc.size().height()) + 30

        self.resize(540, min(height, 900))


# ----------------- SliceView -----------------
class SliceView(QWidget):
    def __init__(self, viewer, plane):
        super().__init__()
        self.viewer = viewer
        self.plane = plane

        self.idx = 0
        self._orig_qimage = None
        
        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False
        self.dragging_wl = False
        self.last_mouse_pos = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        
        self.zooming = False
        self.zoom_anchor_y = None
        self.zoom_anchor_zoom = None
        
        self.is_active = False
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(20, 20)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                
        self.image_label.installEventFilter(self)
        
        # Mouse interactions
        self.setFocusPolicy(Qt.StrongFocus)
        
        # ---------------- Image frame ----------------
        self.image_frame = QWidget()
        self.image_frame.setLayout(QVBoxLayout())
        self.image_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.image_frame.layout().setSpacing(0)
        self.image_frame.layout().addWidget(self.image_label)

        # ---------------- Slice slider ----------------
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slice_text = QLabel()

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider, stretch=1)
        slider_layout.addWidget(self.slice_text)

        # ---------------- Buttons ----------------
        btn_rotate_ccw = QPushButton("-90°")
        btn_rotate_cw = QPushButton("+90°")
        btn_flip_x = QPushButton("-X")
        btn_flip_y = QPushButton("-Y")
        btn_reset = QPushButton("Reset")

        btn_rotate_ccw.clicked.connect(self.rotate_ccw)
        btn_rotate_cw.clicked.connect(self.rotate_cw)
        btn_flip_x.clicked.connect(self.toggle_flip_x)
        btn_flip_y.clicked.connect(self.toggle_flip_y)
        btn_reset.clicked.connect(self.reset_transforms)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_rotate_ccw)
        button_layout.addWidget(btn_rotate_cw)
        button_layout.addWidget(btn_flip_x)
        button_layout.addWidget(btn_flip_y)
        button_layout.addWidget(btn_reset)

        # ---------------- Plane title ----------------
        plane_names = {
            "ax": "Axial",
            "cor": "Coronal",
            "sag": "Sagittal",
        }
        self.title_label = QLabel(plane_names[self.plane])
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ddd;
                font-weight: bold;
                font-size: 13px;
                padding: 2px;
            }
        """)

        # ---------------- Main layout ----------------
        layout = QVBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_frame, stretch=1)

        layout.addLayout(slider_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.update_slider_range()
        self.update_image()
        self.update_border()

    # --------------------------------------------------------
    
    def update_border(self):
        color = "#3a99f2"

        if self.is_active:
            self.image_frame.setStyleSheet(
                f"border: 3px solid {color};"
            )

            self.title_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-weight: bold;
                    font-size: 12px;
                    padding: 2px;
                }}
            """)

        else:
            self.image_frame.setStyleSheet(
                "border: 3px solid #222;"
            )

            self.title_label.setStyleSheet("""
                QLabel {
                    color: #222;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 2px;
                }
            """)
    
    def set_active(self, active):
        self.is_active = active
        self.update_border()
    
    def slider_changed(self, value):
        self.idx = value
        self.viewer.slice_idx[self.plane] = value
        self.update_image()

    def window_changed(self):
        # Prevent inversion
        if self.window_min_slider.value() >= self.window_max_slider.value():
            self.window_max_slider.setValue(
                self.window_min_slider.value() + 1
            )

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
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.viewer.slice_idx[self.plane] = idx
        self.update_image()

    def update_image(self):
        vol = self.viewer.vol_transformed

        rot_k = self.rot_k
        flip_x = self.flip_x
        flip_y = self.flip_y

        # ---------------- Extract slice ----------------
        if self.plane == "ax":
            img = vol[self.idx, :, :]
        elif self.plane == "cor":
            img = vol[:, self.idx, :]
        elif self.plane == "sag":
            img = vol[:, :, self.idx]

        img = img.astype(np.float32)

        # ---------------- Apply transforms ----------------
        if rot_k:
            img = np.rot90(img, rot_k)

        if flip_x:
            img = img[:, ::-1]

        if flip_y:
            img = img[::-1, :]

        # ---------------- Windowing ----------------
        win_min = self.viewer.window_low
        win_max = self.viewer.window_high
        
        img = np.clip(img, win_min, win_max)

        img = ((img - win_min) / (win_max - win_min + 1e-8) * 255)
        img = np.ascontiguousarray(img.astype(np.uint8))

        # ---------------- Convert to Qt image ----------------
        h, w = img.shape
        qimg = QImage(
            img.data,
            w,
            h,
            w,
            QImage.Format_Grayscale8
        )
        self._orig_qimage = qimg
        pixmap = QPixmap.fromImage(qimg)
        self._set_pixmap(pixmap)

        # ---------------- Slice text ----------------
        self.slice_text.setText(
            f"{self.idx}/{self.slider.maximum()}"
        )

    def _set_pixmap(self, pixmap):
        if not self._orig_qimage:
            return

        label_w = max(1, self.image_label.width())
        label_h = max(1, self.image_label.height())

        pixmap = QPixmap.fromImage(self._orig_qimage)

        scaled = pixmap.scaled(
            int(label_w * self.zoom),
            int(label_h * self.zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)

        painter = QPainter(canvas)

        x = (label_w - scaled.width()) // 2 + self.pan_x
        y = (label_h - scaled.height()) // 2 + self.pan_y

        painter.drawPixmap(x, y, scaled)

        painter.end()

        self.image_label.setPixmap(canvas)

    def resizeEvent(self, event):
        if self.image_label.pixmap():
            self._set_pixmap(self.image_label.pixmap())

        super().resizeEvent(event)
    
    def rotate_ccw(self):
        self.rot_k = (self.rot_k + 1) % 4
        self.update_image()

    def rotate_cw(self):
        self.rot_k = (self.rot_k - 1) % 4
        self.update_image()

    def toggle_flip_x(self):
        self.flip_x = not self.flip_x
        self.update_image()

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y
        self.update_image()

    def reset_transforms(self):
        self.rot_k = 0
        self.flip_x = False
        self.flip_y = False
        self.dragging_wl = False
        self.last_mouse_pos = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        
        self.update_image()
    
    def wheelEvent(self, event):
        mods = QApplication.keyboardModifiers()

        delta = event.angleDelta().y()

        # ---------------- Zoom (Ctrl + wheel) ----------------
        if mods & Qt.ControlModifier:
            if delta > 0:
                self.zoom *= 1.1
            else:
                self.zoom /= 1.1

            self.update_image()
            return

        # ---------------- Slice navigation ----------------
        base_step = 1
        fast_step = 5

        # Shift = fast scroll
        if mods & Qt.ShiftModifier:
            step = max(fast_step, self.slider.maximum() // 20)
        else:
            step = base_step

        step = step if delta > 0 else -step

        idx = np.clip(
            self.idx + step,
            0,
            self.slider.maximum()
        )

        self.set_slice(int(idx))
    
    def eventFilter(self, obj, event):
        if obj == self.image_label:

            if event.type() == event.Wheel:
                self.wheelEvent(event)
                return True

            elif event.type() == event.MouseButtonPress:
                self.mousePressEvent(event)
                return True

            elif event.type() == event.MouseMove:
                self.mouseMoveEvent(event)
                return True

            elif event.type() == event.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True

        return super().eventFilter(obj, event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.viewer.set_active_plane(self.plane)
            self.zooming = True
            self.zoom_anchor_y = event.pos().y()
            self.zoom_anchor_zoom = self.zoom
            self.setFocus()

        elif event.button() == Qt.RightButton:
            self.dragging_wl = True
            self.last_mouse_pos = event.pos()
            self.setFocus()

        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.last_mouse_pos = event.pos()
            self.setFocus()

        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.zooming = False

        if event.button() == Qt.RightButton:
            self.dragging_wl = False

        if event.button() == Qt.MiddleButton:
            self.panning = False

        super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        
        # ---------------- Zoom (left drag) ----------------
        if self.zooming:
            dy = pos.y() - self.zoom_anchor_y

            # exponential feel (much smoother than linear)
            factor = 1.0 + (-dy * 0.01)

            factor = np.clip(factor, 0.1, 10.0)

            self.zoom = self.zoom_anchor_zoom * factor
            self.zoom = max(0.1, min(self.zoom, 20.0))

            self.update_image()
            return

        # ---------------- Pan ----------------
        if self.panning:
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()

            self.last_mouse_pos = pos

            self.pan_x += dx
            self.pan_y += dy

            self.update_image()

            return

        # ---------------- Window/Level ----------------
        if self.dragging_wl:
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()

            self.last_mouse_pos = pos

            width = (
                self.viewer.window_high -
                self.viewer.window_low
            )

            center = 0.5 * (
                self.viewer.window_high +
                self.viewer.window_low
            )

            # brightness
            center += dx * 0.005 * width

            # contrast
            width *= (1.0 + dy * 0.01)

            width = max(width, 1e-6)

            low = center - width / 2
            high = center + width / 2

            self.viewer.set_window(low, high)

            return
    
    def keyPressEvent(self, event):
        # ---------------- Slice navigation ----------------
        if event.key() == Qt.Key_A:
            self.set_slice(max(0, self.idx - 1))

        elif event.key() == Qt.Key_D:
            self.set_slice(
                min(self.slider.maximum(), self.idx + 1)
            )

        # ---------------- Center ----------------
        elif event.key() == Qt.Key_C:
            self.pan_x = 0
            self.pan_y = 0
            self.update_image()
        
        # ---------------- Flip ----------------
        elif event.key() == Qt.Key_X:
            self.toggle_flip_x()

        elif event.key() == Qt.Key_Y:
            self.toggle_flip_y()

        # ---------------- Rotate ----------------
        elif event.key() == Qt.Key_R and (event.modifiers() & Qt.ShiftModifier):
            self.rotate_ccw()

        elif event.key() == Qt.Key_R:
            self.rotate_cw()
        
        # ---------------- Reset transforms ----------------
        elif event.key() == Qt.Key_Q:
            self.reset_transforms()
        
        else:
            super().keyPressEvent(event)

# ----------------- Viewer -----------------
class Viewer(QWidget):
    def __init__(self, filepath, transpose):
        super().__init__()
        self.setWindowTitle(f"NumViewer  ({Path(filepath).name})  ")

        self.filepath = filepath
        self.loaded_data = np.load(filepath, mmap_mode="c")

        self.array_selector = None
        self.active_plane = "ax"
        self.plane_order = ["ax", "cor", "sag"]
        
        self.help_dialog = None
        
        self.installEventFilter(self)

        # ---------------- NPZ support ----------------
        if self.filepath.endswith(".npz"):
            self.arrays = {}

            for key in self.loaded_data.files:
                arr = np.squeeze(self.loaded_data[key])

                # only keep >=3D arrays
                if arr.ndim >= 3:
                    self.arrays[key] = arr

            if not self.arrays:
                raise ValueError("No >=3D arrays found in NPZ.")

            first_key = list(self.arrays.keys())[0]
            self.current_array_key = first_key
            self.vol_original = self.arrays[first_key]

        else:
            self.vol_original = np.squeeze(self.loaded_data)

            if self.vol_original.ndim < 3:
                raise ValueError(
                    f"Expected >=3D array, got shape {self.vol_original.shape}"
                )

            self.arrays = {"array": self.vol_original}
            self.current_array_key = "array"

        self.transpose_axes = tuple(transpose)
        self.vol_transformed = self.vol_original.transpose(
            self.transpose_axes
        )
        self.shape = self.vol_transformed.shape
        self.slice_idx = {
            "ax": self.shape[0] // 2,
            "cor": self.shape[1] // 2,
            "sag": self.shape[2] // 2,
        }

        # ---------------- Toolbar ----------------
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        
        if len(self.arrays) > 1:
            self.array_selector = QComboBox()

            for key, arr in self.arrays.items():
                self.array_selector.addItem(
                    f"{key} {arr.shape} | {arr.dtype}",
                    key
                )

            self.array_selector.currentIndexChanged.connect(
                self.change_array
            )

            toolbar.addSeparator()
            label_array = QLabel("Array:")
            label_array.setStyleSheet("margin-right: 6px; font-weight: bold;")
            toolbar.addWidget(label_array)
            toolbar.addWidget(self.array_selector)
            toolbar.addSeparator()
        
        label_axes = QLabel("Transpose volume:")
        label_axes.setStyleSheet("margin-right: 6px; font-weight: bold;")

        toolbar.addWidget(label_axes)
        self.transpose_actions = {}
        self.transpose_order = [
            (0,1,2),
            (1,2,0),
            (2,0,1),
            (0,2,1),
            (1,0,2),
            (2,1,0),
        ]

        for axes in self.transpose_order:
            txt = "".join(map(str, axes)) 
            act = QAction(txt, self)
            act.setCheckable(True)
            if tuple(axes) == self.transpose_axes:
                act.setChecked(True)
            act.triggered.connect(
                lambda _, a=axes: self.apply_transpose(a)
            )
            toolbar.addAction(act)
            self.transpose_actions[tuple(axes)] = act

#        spacer = QWidget()
#        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
#        toolbar.addWidget(spacer)
        
        help_act = QAction("Help (F1)", self)
        help_act.setToolTip("Hotkeys and controls (F1)")
        help_act.setShortcut("F1")
        help_act.triggered.connect(self.show_help)
        
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)

        help_act.setFont(font)
        toolbar.addSeparator()
        toolbar.addAction(help_act)
        toolbar.addSeparator()
        
        # ---------------- Global intensity controls ----------------

        # Full data range
        self.data_min = float(self.vol_original.min())
        self.data_max = float(self.vol_original.max())

        # ---------------- Initial display window ----------------
        uniq = np.unique(self.vol_original)

        # binary / boolean image
        if (
            self.vol_original.dtype == np.bool_
            or (
                len(uniq) <= 2
                and np.all(np.isin(uniq, [0, 1]))
            )
        ):
            p_low = 0.0
            p_high = 1.0

        # General intensity image
        else:
            p_low = float(np.percentile(self.vol_original, PMIN))
            p_high = float(np.percentile(self.vol_original, PMAX))
        
        self.default_window_low = p_low
        self.default_window_high = p_high
        self.window_low = p_low
        self.window_high = p_high
        
        self.intensity_low_slider = QSlider(Qt.Vertical)
        self.intensity_high_slider = QSlider(Qt.Vertical)

        for s in [self.intensity_low_slider, self.intensity_high_slider]:
            s.setRange(0, 1000)

        self.intensity_low_label = QDoubleSpinBox()
        self.intensity_high_label = QDoubleSpinBox()
        for s in [
            self.intensity_low_label,
            self.intensity_high_label
        ]:
            s.setRange(
                self.data_min,
                self.data_max
            )
            s.setKeyboardTracking(False)
            s.setButtonSymbols(QDoubleSpinBox.NoButtons)
            s.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            s.setFixedWidth(80)
            s.lineEdit().setAlignment(Qt.AlignCenter)
            
        self.intensity_low_label.setValue(self.window_low)
        self.intensity_high_label.setValue(self.window_high)
        self.sync_controls()
        
        self.intensity_low_slider.valueChanged.connect(self.intensity_changed)
        self.intensity_high_slider.valueChanged.connect(self.intensity_changed)

        # ---------------- Slice views ----------------
        self.slice_views = {}
        planes = ["ax", "cor", "sag"]
        h_layout = QHBoxLayout()

        slices_widget = QWidget()
        slices_widget.setLayout(h_layout)

        # IMPORTANT: slices define the main vertical geometry
        slices_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        for plane in planes:
            view = SliceView(self, plane)
            view.set_slice(self.slice_idx[plane])
            self.slice_views[plane] = view
            h_layout.addWidget(view)

        # ---------------- Intensity widget ----------------
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(5)

        intensity_layout.addWidget(QLabel("vmax"), alignment=Qt.AlignHCenter)
        self.intensity_high_slider.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        intensity_layout.addWidget(
            self.intensity_high_slider,
            stretch=1
        )
        intensity_layout.addWidget(
            self.intensity_high_label,
            alignment=Qt.AlignHCenter | Qt.AlignVCenter
        )

        intensity_layout.addSpacing(10)

        intensity_layout.addWidget(QLabel("vmin"), alignment=Qt.AlignHCenter)
        self.intensity_low_slider.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        intensity_layout.addWidget(
            self.intensity_low_slider,
            stretch=1
        )
        intensity_layout.addWidget(
            self.intensity_low_label,
            alignment=Qt.AlignHCenter | Qt.AlignVCenter
        )
        
        self.intensity_high_slider.setMinimumHeight(60)
        self.intensity_low_slider.setMinimumHeight(60)
        
        intensity_widget = QWidget()
        intensity_widget.setLayout(intensity_layout)

        intensity_widget.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Expanding
        )
        intensity_layout.addSpacing(9)
        
        # ---------------- Viewer row container ----------------
        viewer_container = QWidget()

        viewer_layout = QHBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(5)

        viewer_layout.addWidget(slices_widget)
        viewer_layout.addWidget(intensity_widget)

        viewer_container.setLayout(viewer_layout)

        viewer_container.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        # ---------------- Main layout ----------------
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(toolbar)
        main_layout.addWidget(viewer_container)

        self.setLayout(main_layout)
        self.resize(1400, 425)
        
        # ---------------- Array switching ----------------
        self.short_next_array = QShortcut(QKeySequence("Tab"), self)
        self.short_prev_array = QShortcut(QKeySequence("Shift+Tab"), self)
        self.short_next_array.activated.connect(self.next_array)
        self.short_prev_array.activated.connect(self.prev_array)
        
        # ---------------- Quit application ----------------
        self.short_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.short_quit.activated.connect(QApplication.quit)
        
        # ---------------- Slice view switching ----------------
        self.short_ax = QShortcut(QKeySequence("1"), self)
        self.short_cor = QShortcut(QKeySequence("2"), self)
        self.short_sag = QShortcut(QKeySequence("3"), self)
        self.short_ax.activated.connect(lambda: self.set_active_plane("ax"))
        self.short_cor.activated.connect(lambda: self.set_active_plane("cor"))
        self.short_sag.activated.connect(lambda: self.set_active_plane("sag"))
        
        # ---------------- Transposition cycling ----------------
        self.short_cycle_transpose = QShortcut(QKeySequence("T"), self)
        self.short_cycle_transpose.activated.connect(self.cycle_transpose)
        
        # ---------------- Save views ----------------
        self.short_save_current = QShortcut(QKeySequence("Ctrl+S"), self)
        self.short_save_all = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self.short_save_current.activated.connect(self.save_current_view)
        self.short_save_all.activated.connect(self.save_all_views)
        
        self.reset_window()

    # --------------------------------------------------------
    
    def load_help_text(self):
        return (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    
    def show_help(self):
        path = Path(__file__).resolve().parent / "README.md"
        text = path.read_text(encoding="utf-8")

        if self.help_dialog is None:
            self.help_dialog = HelpDialog(text, self)

        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()
    
    def cycle_transpose(self):
        current = tuple(self.transpose_axes)

        try:
            i = self.transpose_order.index(current)
        except ValueError:
            i = 0

        next_axes = self.transpose_order[(i + 1) % len(self.transpose_order)]
        self.apply_transpose(list(next_axes))
    
    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress:
            # ---------------- Auto-window ----------------
            if event.key() == Qt.Key_Space:
                self.reset_window()

            # ---------------- Next array ----------------
            if event.key() == Qt.Key_Tab and (event.modifiers() & Qt.ControlModifier):
                self.next_array()
                return True

            # ---------------- Previous array ----------------
            if event.key() == Qt.Key_Backtab:
                self.prev_array()
                return True

        return super().eventFilter(obj, event)

    def set_active_plane(self, plane):
        self.active_plane = plane
        for p, view in self.slice_views.items():
            active = (p == plane)
            view.set_active(active)

            if active:
                view.setFocus()

    def next_array(self):
        if not self.array_selector:
            return
        i = self.array_selector.currentIndex()
        i = (i + 1) % self.array_selector.count()
        self.array_selector.setCurrentIndex(i)

    def prev_array(self):
        if not self.array_selector:
            return
        i = self.array_selector.currentIndex()
        i = (i - 1) % self.array_selector.count()
        self.array_selector.setCurrentIndex(i)
    
    def sync_controls(self):
        rng = self.data_max - self.data_min + 1e-8

        low_slider = int(1000 * (self.window_low - self.data_min) / rng)
        high_slider = int(1000 * (self.window_high - self.data_min) / rng)

        # sliders
        self.intensity_low_slider.blockSignals(True)
        self.intensity_high_slider.blockSignals(True)

        self.intensity_low_slider.setValue(low_slider)
        self.intensity_high_slider.setValue(high_slider)

        self.intensity_low_slider.blockSignals(False)
        self.intensity_high_slider.blockSignals(False)

        # spinboxes
        self.intensity_low_label.blockSignals(True)
        self.intensity_high_label.blockSignals(True)

        self.intensity_low_label.setValue(self.window_low)
        self.intensity_high_label.setValue(self.window_high)

        self.intensity_low_label.blockSignals(False)
        self.intensity_high_label.blockSignals(False)
    
    def apply_transpose(self, axes):
        old_shape = self.vol_transformed.shape

        # Current normalized positions in displayed volume
        old_pos = [
            self.slice_idx["ax"] / max(old_shape[0] - 1, 1),
            self.slice_idx["cor"] / max(old_shape[1] - 1, 1),
            self.slice_idx["sag"] / max(old_shape[2] - 1, 1),
        ]

        # Store transpose
        self.transpose_axes = tuple(axes)

        # Update toolbar checkmarks
        for a in self.transpose_actions.values():
            a.setChecked(False)

        self.transpose_actions[self.transpose_axes].setChecked(True)

        # Apply transpose
        self.vol_transformed = self.vol_original.transpose(
            self.transpose_axes
        )
        new_shape = self.vol_transformed.shape

        # Re-map positions according to transpose axes
        new_pos = [0, 0, 0]

        for new_axis, old_axis in enumerate(self.transpose_axes):
            new_pos[new_axis] = old_pos[old_axis]

        self.shape = new_shape
        self.slice_idx = {
            "ax": int(new_pos[0] * (new_shape[0] - 1)),
            "cor": int(new_pos[1] * (new_shape[1] - 1)),
            "sag": int(new_pos[2] * (new_shape[2] - 1)),
        }
        self.update_all_views()

    def update_all_views(self):
        for plane, view in self.slice_views.items():
            # block recursive slider updates
            view.slider.blockSignals(True)
            view.update_slider_range()

            idx = self.slice_idx[plane]
            idx = np.clip(
                idx,
                0,
                view.slider.maximum()
            )

            view.idx = idx
            view.slider.setValue(idx)
            view.slider.blockSignals(False)

            view.update_image()
    
    def intensity_changed(self):
        low = self.intensity_low_slider.value()
        high = self.intensity_high_slider.value()

        if low >= high:
            return

        rng = self.data_max - self.data_min

        win_low = self.data_min + (low / 1000.0) * rng
        win_high = self.data_min + (high / 1000.0) * rng

        self.set_window(win_low, win_high)
    
    def set_window(self, low, high):
        if low >= high:
            return

        self.window_low = max(self.data_min, low)
        self.window_high = min(self.data_max, high)

        rng = self.data_max - self.data_min

        low_slider = int(
            1000 * (self.window_low - self.data_min) / rng
        )

        high_slider = int(
            1000 * (self.window_high - self.data_min) / rng
        )

        # spinboxes (sync UI without recursion)
        self.intensity_low_label.blockSignals(True)
        self.intensity_high_label.blockSignals(True)

        self.intensity_low_label.setValue(self.window_low)
        self.intensity_high_label.setValue(self.window_high)

        self.intensity_low_label.blockSignals(False)
        self.intensity_high_label.blockSignals(False)

        self.update_all_views()
        self.sync_controls()
        
    def spinbox_window_changed(self):
        low = self.intensity_low_label.value()
        high = self.intensity_high_label.value()

        if low >= high:
            return

        self.set_window(low, high)
    
    def reset_window(self):
        self.set_window(
            self.default_window_low,
            self.default_window_high
        )
    
    def set_volume(self, vol):
        self.vol_original = vol

        # if >3D take first container by default
        while self.vol_original.ndim > 3:
            self.vol_original = self.vol_original[0]

        self.vol_transformed = self.vol_original.transpose(
            self.transpose_axes
        )

        self.shape = self.vol_transformed.shape

        self.slice_idx = {
            "ax": self.shape[0] // 2,
            "cor": self.shape[1] // 2,
            "sag": self.shape[2] // 2,
        }

        # recompute window defaults
        self.data_min = float(self.vol_original.min())
        self.data_max = float(self.vol_original.max())

        uniq = np.unique(self.vol_original)

        # binary / boolean image
        if (
            self.vol_original.dtype == np.bool_
            or (
                len(uniq) <= 2
                and np.all(np.isin(uniq, [0, 1]))
            )
        ):
            p_low = 0.0
            p_high = 1.0

        # general image
        else:
            p_low = float(np.percentile(self.vol_original, PMIN))
            p_high = float(np.percentile(self.vol_original, PMAX))

        self.default_window_low = p_low
        self.default_window_high = p_high

        self.window_low = p_low
        self.window_high = p_high

        # update spinbox ranges for new array
        self.intensity_low_label.blockSignals(True)
        self.intensity_high_label.blockSignals(True)

        self.intensity_low_label.setRange(
            self.data_min,
            self.data_max
        )

        self.intensity_high_label.setRange(
            self.data_min,
            self.data_max
        )

        self.intensity_low_label.blockSignals(False)
        self.intensity_high_label.blockSignals(False)

        self.intensity_low_label.setValue(self.window_low)
        self.intensity_high_label.setValue(self.window_high)

        self.sync_controls()
        self.update_all_views()
    
    def change_array(self, idx):
        key = self.array_selector.itemData(idx)
        self.current_array_key = key
        vol = self.arrays[key]

        # handle >3D automatically
        while vol.ndim > 3:
            vol = vol[0]

        self.set_volume(vol)
    
    def save_current_view(self):
        out_dir = Path("exports")
        out_dir.mkdir(exist_ok=True)
        p = Path(self.filepath)
        name = p.name.removesuffix("".join(p.suffixes))
        key = self.current_array_key

        for plane, view in self.slice_views.items():
            pixmap = view.image_label.pixmap()
            if pixmap is None:
                continue

            filename = out_dir / f"{name}_{key}_{plane}{view.idx:04d}.png"
            pixmap.save(str(filename))

        print(f"Saved current view to {out_dir}.")
    
    def save_all_views(self):
        out_dir = Path("exports")
        out_dir.mkdir(exist_ok=True)
        p = Path(self.filepath)
        name = p.name.removesuffix("".join(p.suffixes))
        key = self.current_array_key

        vol = self.vol_transformed

        for plane, view in self.slice_views.items():
            max_idx = view.slider.maximum()

            for i in range(max_idx + 1):
                view.set_slice(i)
                QApplication.processEvents()  # ensure update_image runs

                pixmap = view.image_label.pixmap()
                if pixmap is None:
                    continue

                filename = out_dir / f"{name}_{key}_{plane}{i:04d}.png"
                pixmap.save(str(filename))

        print(f"Saved full volume to {out_dir}.")

# ----------------- entry -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "i",
        help="Input .npy file"
    )
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    viewer = Viewer(args.i, DEFAULT_TRANSPOSE)
    viewer.show()
    sys.exit(app.exec_())


