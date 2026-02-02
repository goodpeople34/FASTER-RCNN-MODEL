from PySide6.QtWidgets import QFileDialog, QDialog
from PySide6.QtCore import QStandardPaths, QDir,QObject,Slot,Signal, QThread
from PySide6.QtGui import QImageWriter

class FileDialog:


    def _open(self) -> None:
        dialog = self._createDialog(
            title="open files",
            accept_mode=QFileDialog.AcceptMode.AcceptOpen
        )

        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._clear_images()
            self._extracted_text.clear()

            self.thread = QThread()
            self.worker = Worker(dialog.selectedFiles(), self._model)

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self._image_viewer)
            self.worker.error.connect(self._on_error)

            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.thread.start()



    def _saveFileAs(self) -> None:
        dialog = self._createDialog(
            title="save file as",
            accept_mode=QFileDialog.AcceptMode.AcceptSave
        )
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._save_file(dialog.selectedFiles()[0])

    def _createDialog(self,*, title, accept_mode):
        dialog = QFileDialog(self,title)
        self._initializeImageFileDialog(dialog, accept_mode)
        return dialog

    def _initializeImageFileDialog(self, dialog, accept_mode):
        self._setInitialDirectory(dialog)
        self._setImageFilter(dialog)
        dialog.AcceptMode(accept_mode)
        if accept_mode == QFileDialog.AcceptMode.AcceptSave:
            dialog.setDefaultSuffix("jpg")
    
    def _setInitialDirectory(self, dialog):
        if getattr(self, "_first_file_dialog", True):
            self._first_file_dialog = False
            locations = QStandardPaths.standardLocations(
                QStandardPaths.StandardLocation.PicturesLocation
            )
            dialog.setDirectory(locations[-1] if locations else QDir.currentPath())

    def _setImageFilter(self, dialog):
        mime_types = sorted([m.data().decode('utf-8') for m in QImageWriter.supportedMimeTypes()])

        dialog.setMimeTypeFilters(mime_types)
        dialog.selectMimeTypeFilter("image/jpeg")
    

    def loading(self, dialog):
        paths = dialog.selectedFiles()

        self.thread = QThread()
        self.worker = LoadWorker()

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(lambda: self.worker.run(paths))
        self.worker.progress.connect(self.on_file_loaded)
        self.worker.finished.connect(self.on_load_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_file_loaded(self, path):
        self.status_label.setText(f"Loaded: {path}")

    def on_load_finished(self):
        self.status_label.setText("All files loaded")
    
    def _on_error(self, msg):
        QMessageBox.critical(self, "OCR Error", msg)



class Worker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, paths, model):
        super().__init__()
        self.paths = paths
        self.model = model

    @Slot()
    def run(self):
        results = []

        try:
            for path in self.paths:
                image, text = self.model(path)
                results.append((image, text, path))
        except Exception as e:
            self.error.emit(str(e))
            return

        self.finished.emit(results)
