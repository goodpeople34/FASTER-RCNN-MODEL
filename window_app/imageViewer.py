# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations

from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QLabel,QMainWindow, QMessageBox, QScrollArea,QSizePolicy, QWidget,QPlainTextEdit, QHBoxLayout, QVBoxLayout)
from PySide6.QtGui import (QColorSpace, QGuiApplication, QImageReader, QImageWriter, QKeySequence, QPalette, QPainter, QPixmap)
from PySide6.QtCore import QDir, QStandardPaths, Qt, Slot

from fileDialog import FileDialog
from tessearctOcr import CallModel


ABOUT = """<p>The <b>Image Viewer</b> example shows how to combine QLabel
and QScrollArea to display an image. QLabel is typically used
for displaying a text, but it can also display an image.
QScrollArea provides a scrolling view around another widget.
If the child widget exceeds the size of the frame, QScrollArea
automatically provides scroll bars. </p><p>The example
demonstrates how QLabel's ability to scale its contents
(QLabel.scaledContents), and QScrollArea's ability to
automatically resize its contents
(QScrollArea.widgetResizable), can be used to implement
zooming and scaling features. </p><p>In addition the example
shows how to use QPainter to print an image.</p>
"""

DUMMY = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco
laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, 
sunt in culpa qui officia deserunt mollit anim id est laborum"""

MAX_SIZE = 800


class ImageViewer(QMainWindow, FileDialog, CallModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._scroll_area = QScrollArea()
        self._scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self._scroll_area.setWidgetResizable(True)  
        self.setCentralWidget(self._scroll_area)

        self._content = QWidget()
        self._image_container = QWidget()
        self._Hlayout = QHBoxLayout(self._content)
        self._Vlayout = QVBoxLayout()
        self._scroll_area.setWidget(self._content)

        self._scale_factor = 1.0
        self._first_file_dialog = True

        self._image_labels = []
        self._images = []

        # self._image_label = QLabel()
        # self._image_label.setBackgroundRole(QPalette.ColorRole.Base)
        # self._image_label.setScaledContents(True)
        # self._image_label.setSizePolicy(
        #     QSizePolicy.Policy.Expanding,
        #     QSizePolicy.Policy.Expanding
        # )

        self._text_view = QPlainTextEdit(DUMMY)
        self._Vlayout.addWidget(self._image_container)
        self._Hlayout.addLayout(self._Vlayout, 1)
        self._Hlayout.addWidget(self._text_view, 2)




        # self._scale_factor = 1.0
        # self._first_file_dialog = True
        # self._image_label = QLabel()
        # self._image_label.setBackgroundRole(QPalette.ColorRole.Base)
        # self._image_label.setSizePolicy(QSizePolicy.Policy.Ignored,
        #                                 QSizePolicy.Policy.Ignored)
        # self._image_label.setScaledContents(True)

        # self._scroll_area = QScrollArea()
        # self._scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        # self._scroll_area.setWidget(self._image_label)
        # self._scroll_area.setVisible(False)
        # self.setCentralWidget(self._scroll_area)

        self._create_actions()

        self.resize(QGuiApplication.primaryScreen().availableSize() * 3 / 5)

    def load_file(self, fileName):

        new_image = self._model(fileName)

        if new_image.isNull():
            raise ValueError("Generated image is null.")
            

        # reader = QImageReader(fileName)
        # reader.setAutoTransform(True)
        # new_image = reader.read()

        native_filename = QDir.toNativeSeparators(fileName)
        # if new_image.isNull():
        #     error = reader.errorString()
        #     QMessageBox.information(self, QGuiApplication.applicationDisplayName(),
        #                             f"Cannot load {native_filename}: {error}")
        #     return False

        max_size = MAX_SIZE

        pixmap = QPixmap.fromImage(new_image)

        resized_image = pixmap.scaled(max_size, max_size,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(resized_image)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("padding: 8px; border-bottom: 1px solid #ccc;")

        self._Vlayout.addWidget(label)
        self._image_labels.append(label)
        self._images.append(new_image)

        # w, h, d = new_image.width(), new_image.height(), new_image.depth()
        # color_space = new_image.colorSpace()
        # description = color_space.description() if color_space.isValid() else "unknown"
        # message = f'Opened "{native_filename}", {w}x{h}, Depth: {d} ({description})'
        message = f'successfully detect {native_filename}'
        self.statusBar().showMessage(message)

        return True

    def _set_image(self, new_image):
        self._image = new_image
        if self._image.colorSpace().isValid():
            color_space = QColorSpace(QColorSpace.NamedColorSpace.SRgb)
            self._image.convertToColorSpace(color_space)
        self._image_label.setPixmap(QPixmap.fromImage(self._image))
        self._scale_factor = 1.0

        self._scroll_area.setVisible(True)
        self._print_act.setEnabled(True)
        self._fit_to_window_act.setEnabled(True)
        self._update_actions()

        if not self._fit_to_window_act.isChecked():
            self._image_label.adjustSize()

    def _save_file(self, fileName):
        writer = QImageWriter(fileName)

        native_filename = QDir.toNativeSeparators(fileName)
        if not writer.write(self._image):
            error = writer.errorString()
            message = f"Cannot write {native_filename}: {error}"
            QMessageBox.information(self, QGuiApplication.applicationDisplayName(),
                                    message)
            return False
        self.statusBar().showMessage(f'Wrote "{native_filename}"')
        return True

    @Slot()
    def _print_(self):
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            with QPainter(printer) as painter:
                pixmap = self._image_label.pixmap()
                rect = painter.viewport()
                size = pixmap.size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
                painter.setWindow(pixmap.rect())
                painter.drawPixmap(0, 0, pixmap)

    @Slot()
    def _copy(self):
        QGuiApplication.clipboard().setImage(self._image)

    @Slot()
    def _paste(self):
        new_image = QGuiApplication.clipboard().image()
        if new_image.isNull():
            self.statusBar().showMessage("No image in clipboard")
        else:
            self._set_image(new_image)
            self.setWindowFilePath('')
            w = new_image.width()
            h = new_image.height()
            d = new_image.depth()
            message = f"Obtained image from clipboard, {w}x{h}, Depth: {d}"
            self.statusBar().showMessage(message)

    @Slot()
    def _zoom_in(self):
        self._scale_image(1.25)

    @Slot()
    def _zoom_out(self):
        self._scale_image(0.8)

    @Slot()
    def _normal_size(self):
        self._image_label.adjustSize()
        self._scale_factor = 1.0

    @Slot()
    def _fit_to_window(self):
        fit_to_window = self._fit_to_window_act.isChecked()
        self._scroll_area.setWidgetResizable(fit_to_window)
        if not fit_to_window:
            self._normal_size()
        self._update_actions()

    @Slot()
    def _about(self):
        QMessageBox.about(self, "About Image Viewer", ABOUT)

    def _create_actions(self):
        file_menu = self.menuBar().addMenu("&File")

        self._open_act = file_menu.addAction("&Open...")
        self._open_act.triggered.connect(self._open)
        self._open_act.setShortcut(QKeySequence.StandardKey.Open)

        self._save_as_act = file_menu.addAction("&Save As...")
        self._save_as_act.triggered.connect(self._saveFileAs)
        self._save_as_act.setEnabled(False)

        self._print_act = file_menu.addAction("&Print...")
        self._print_act.triggered.connect(self._print_)
        self._print_act.setShortcut(QKeySequence.StandardKey.Print)
        self._print_act.setEnabled(False)

        file_menu.addSeparator()

        self._exit_act = file_menu.addAction("E&xit")
        self._exit_act.triggered.connect(self.close)
        self._exit_act.setShortcut("Ctrl+Q")

        edit_menu = self.menuBar().addMenu("&Edit")

        self._copy_act = edit_menu.addAction("&Copy")
        self._copy_act.triggered.connect(self._copy)
        self._copy_act.setShortcut(QKeySequence.StandardKey.Copy)
        self._copy_act.setEnabled(False)

        self._paste_act = edit_menu.addAction("&Paste")
        self._paste_act.triggered.connect(self._paste)
        self._paste_act.setShortcut(QKeySequence.StandardKey.Paste)

        view_menu = self.menuBar().addMenu("&View")

        self._zoom_in_act = view_menu.addAction("Zoom &In (25%)")
        self._zoom_in_act.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self._zoom_in_act.triggered.connect(self._zoom_in)
        self._zoom_in_act.setEnabled(False)

        self._zoom_out_act = view_menu.addAction("Zoom &Out (25%)")
        self._zoom_out_act.triggered.connect(self._zoom_out)
        self._zoom_out_act.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self._zoom_out_act.setEnabled(False)

        self._normal_size_act = view_menu.addAction("&Normal Size")
        self._normal_size_act.triggered.connect(self._normal_size)
        self._normal_size_act.setShortcut("Ctrl+S")
        self._normal_size_act.setEnabled(False)

        view_menu.addSeparator()

        self._fit_to_window_act = view_menu.addAction("&Fit to Window")
        self._fit_to_window_act.triggered.connect(self._fit_to_window)
        self._fit_to_window_act.setEnabled(False)
        self._fit_to_window_act.setCheckable(True)
        self._fit_to_window_act.setShortcut("Ctrl+F")

        help_menu = self.menuBar().addMenu("&Help")

        about_act = help_menu.addAction("&About")
        about_act.triggered.connect(self._about)
        about_qt_act = help_menu.addAction("About &Qt")
        about_qt_act.triggered.connect(QApplication.aboutQt)

    def _update_actions(self):
        has_image = not self._image.isNull()
        self._save_as_act.setEnabled(has_image)
        self._copy_act.setEnabled(has_image)
        enable_zoom = not self._fit_to_window_act.isChecked()
        self._zoom_in_act.setEnabled(enable_zoom)
        self._zoom_out_act.setEnabled(enable_zoom)
        self._normal_size_act.setEnabled(enable_zoom)

    def _scale_image(self, factor):
        self._scale_factor *= factor
        new_size = self._scale_factor * self._image_label.pixmap().size()
        self._image_label.resize(new_size)

        self._adjust_scrollbar(self._scroll_area.horizontalScrollBar(), factor)
        self._adjust_scrollbar(self._scroll_area.verticalScrollBar(), factor)

        self._zoom_in_act.setEnabled(self._scale_factor < 3.0)
        self._zoom_out_act.setEnabled(self._scale_factor > 0.333)

    def _adjust_scrollbar(self, scrollBar, factor):
        pos = int(factor * scrollBar.value()
                  + ((factor - 1) * scrollBar.pageStep() / 2))
        scrollBar.setValue(pos)