from PySide6.QtWidgets import QFileDialog, QDialog
from PySide6.QtCore import QStandardPaths, QDir
from PySide6.QtGui import QImageWriter

class FileDialog:

    def _open(self)->None:
        dialog = self._createDialog(
            title="open files",
            accept_mode=QFileDialog.AcceptMode.AcceptOpen
        )

        dialog.setFileMode(
            QFileDialog.FileMode.ExistingFiles
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            for path in dialog.selectedFiles():
                if not self.load_file(path):
                    break

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
        