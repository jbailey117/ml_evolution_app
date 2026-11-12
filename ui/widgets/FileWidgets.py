from PySide6.QtWidgets import (
    QListWidget,
    QDialog,
    QDialogButtonBox,
    QListWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    )
from PySide6.QtGui import (
    QCloseEvent,
    QFont,
    QShowEvent,
)
from PySide6.QtCore import (
    QSize,
    Qt,
)
import pandas as pd
from . import CSVPreview

class FileList(QListWidget):
    def __init__(self, files:list[str], parent:QDialog, button):
        super().__init__(parent)
        self.button = button
        self.setFont(QFont("Arial", 12))
        for file_name in files:
            self.addItem(QListWidgetItem(file_name.split("/")[-1]))

        self.currentRowChanged.connect(self.update_selected_file)

    def update_selected_file(self):
        self.button.setDisabled(False)

class FileLoadDialog(QDialog):
    def __init__(self, parent, files:list[str]):
        super().__init__(parent)
        self.files = files
        self.setWindowTitle('Select Data')
        self.setMinimumSize(QSize(400, 400))

        layout = QVBoxLayout()
        layout.setContentsMargins(5,5,5,5)

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.buttons()[0].setDisabled(True)
        self.file_list = FileList(files, self, self.buttonBox.buttons()[0])
        layout.addWidget(self.file_list)
        layout.addWidget(self.buttonBox, 0, Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

class LoadPrompt(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        text = QLabel("Load file or select loaded csv")
        layout.addWidget(text, alignment=Qt.AlignmentFlag.AlignCenter)
        buttonbox_layout = QHBoxLayout()
        file_button = QPushButton("Select File")
        file_button.pressed.connect(self.load_file)
        buttonbox_layout.addWidget(file_button)
        csv_button = QPushButton("Select CSV")
        df = self.get_dataframes()
        if df == None or len(df.keys()) == 0: # type: ignore
            csv_button.setDisabled(True)
        csv_button.pressed.connect(self.load_csv)
        buttonbox_layout.addWidget(csv_button)
        buttonbox = QWidget()
        buttonbox.setLayout(buttonbox_layout)
        layout.addWidget(buttonbox)
        self.setLayout(layout)

        self.data = pd.DataFrame()

    def load_file(self):
        self.hide()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSVs (*.csv)")
        try:
            data = pd.read_csv(file_path)
            self.data = data
            self.accept()
        except FileNotFoundError:
            self.reject()

    def load_csv(self):
        self.hide()
        dfs = self.get_dataframes()
        if dfs == None:
            return
        dlg = FileLoadDialog(self, dfs.keys()) # type: ignore
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        if dlg.exec():
            self.data = dfs[dlg.file_list.currentItem().text()] # type: ignore
            self.accept()

    def get_dataframes(self):
        object = self
        for _ in range(5):
            parent = object.parent()
            if parent == None:
                return None
            try:
                return parent.dataframes # type: ignore
            except:
                object = parent

class SaveData(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = ""
        layout = QVBoxLayout()

        self.enter_name = QLineEdit()
        self.enter_name.setPlaceholderText("Enter name to save data as")
        layout.addWidget(self.enter_name)

        buttonbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        layout.addWidget(buttonbox)

        self.setLayout(layout)

    def accept(self) -> None:
        self.name = self.enter_name.text()
        if self.name == "":
            return
        return super().accept()

class CloseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Would you like to save data to program before closing?"))
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

        self.setLayout(layout)

    def closeEvent(self, arg__1: QCloseEvent) -> None:
        self.done(2)
        arg__1.accept()

class DataEditWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self.setWindowTitle("View/Edit Data")
        self.setMinimumSize(1000, parent.size().height()-100)

        layout = QVBoxLayout()
        layout.setContentsMargins(5,5,5,5)

        content_layout = QHBoxLayout()

        # csv preview window
        self.csv_view = CSVPreview(self, pd.DataFrame())
        content_layout.addWidget(self.csv_view)

        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        layout.addWidget(content_widget)

        buttonlayout = QHBoxLayout()
        buttonlayout.setContentsMargins(5, 5, 5, 5)
        buttonlayout.setSpacing(20)
        open_button = QPushButton("Open Data")
        open_button.pressed.connect(self.change_data)
        buttonlayout.addWidget(open_button)
        save_button = QPushButton("Save Data")
        save_button.pressed.connect(self.save_data)
        buttonlayout.addWidget(save_button)
        close_button = QPushButton("Close")
        close_button.pressed.connect(self.close_button)
        buttonlayout.addWidget(close_button)
        buttonBox = QWidget()
        buttonBox.setLayout(buttonlayout)
        layout.addWidget(buttonBox)

        self.saved = True

        self.setLayout(layout)

    def showEvent(self, arg__1: QShowEvent) -> None:
        super().showEvent(arg__1)
        self.csv_view.graph_header.make_column_graphs()

    def get_data(self) -> pd.DataFrame | None:
        data_dlg = LoadPrompt(self.parent())
        data_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        if data_dlg.exec():
            return data_dlg.data
        else:
            return None

    def change_data(self, new:bool=False) -> None:
        if not (new or self.saved):
            warn = QMessageBox.question(self, "Data not saved", "Would you like to save current data?")
            if warn == QMessageBox.StandardButton.Yes:
                if not self.save_data():
                    return
        data = self.get_data()
        if type(data) != pd.DataFrame:
            return
        self.csv_view.preview_file(data)
        self._parent._layout.setCurrentWidget(self) # type: ignore

    def setSaved(self, saved:bool=True) -> None:
        self.saved = saved

    def save_data(self) -> bool:
        dlg = SaveData(self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        if dlg.exec():
            if dlg.name in self._parent.dataframes:
                warn = QMessageBox.question(self, "Name already saved", f"{dlg.name} already exists. Would you like to update it?")
                if warn != QMessageBox.StandardButton.Yes:
                    return False
            self._parent.dataframes[dlg.name] = self.csv_view.table_model._data # type: ignore
            self.setSaved()
            return True
        return False

    def close_button(self) -> None:
        if not self.saved:
            dlg = CloseDialog(self)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            result = dlg.exec()
            if result == 1:
                if not self.save_data():
                    return
            elif result != 0:
                return
        self._parent.defaultScreen() # type: ignore