from PySide6.QtWidgets import (
    QApplication, QWidget, QScrollArea,
    QHBoxLayout, QLabel, QPlainTextEdit, QMainWindow
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Scroll area (whole window) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        # --- Content widget ---
        content = QWidget()
        scroll.setWidget(content)

        # --- Horizontal layout inside scroll ---
        layout = QHBoxLayout(content)

        # Left: Image
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap("example.png"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #222;")

        # Right: Plain text (self-scrolling)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText("Lots of text...\n" * 50)

        # Add widgets
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.text_edit, 2)

        self.resize(900, 500)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
