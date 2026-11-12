from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableView,
    QHeaderView,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
)
from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QPoint,
    QItemSelectionModel,
)
from PySide6.QtGui import (
    QResizeEvent,
    QAction,
    QDoubleValidator,
)
import pandas as pd
from functools import wraps
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from typing import Any
from util import DataLoader
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import DataEditWidget

class CSVPreview(QWidget):
    def __init__(self, parent:"DataEditWidget", data:pd.DataFrame):
        super().__init__(parent)
        self._parent=parent
        layout = QVBoxLayout()

        self.table_model = CSVTable(self, data)
        self.table_view = QTableView()

        self.graph_header = CSVGraphHeader(self)
        self.dataloader = DialogDataLoader(data, self.table_model, self.table_view, self.graph_header, lambda: self._parent.setSaved(False))
        layout.addWidget(self.graph_header)

        header = self.table_view.horizontalHeader()
        header.setDefaultSectionSize(160)
        self.table_view.setBaseSize(160 * 9 + 50, 20)
        self.table_view.setModel(self.table_model)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table_view.horizontalScrollBar().valueChanged.connect(self.graph_header.adjust_graphs)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_context_menu)
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.table_view)
        self.setLayout(layout)

    def preview_file(self, data:pd.DataFrame):
        self.table_model.change_data(data)
        self.dataloader = DialogDataLoader(self.table_model._data, self.table_model, self.table_view, self.graph_header, lambda: self._parent.setSaved(False))
        self.update()
        self.graph_header.make_column_graphs()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.graph_header.adjust_graphs()
        return super().resizeEvent(event)

    def adjust_column(self, columns:set[str]):
        dlg = AdjustDialog(self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        if dlg.exec():
            self.dataloader.adjust_columns(columns, *dlg.adjustment)

    def get_columns(self) -> set[str]:
        return set([self.table_model._data.columns[index.column()] for index in self.table_view.selectionModel().selectedColumns()])

    def show_context_menu(self, pos:QPoint):
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        columns = self.get_columns()
        s = ""
        if len(columns) != 1:
            s = "s"
        idx = self.table_view.indexAt(pos)
        if idx.row() == 0:
            if len(columns) == 0:
                self.table_view.selectColumn(idx.column())
            else:
                select_model = self.table_view.selectionModel()
                select_model.select(self.table_view.model().index(0, idx.column()), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Columns)
            columns.add(self.dataloader.data.columns.to_list()[idx.column()])

        # Normalize
        normalize = QAction('Normalize', self)
        if len(columns) == 0 and type(self.dataloader.target_data) != pd.DataFrame:
            normalize.setDisabled(True)
            normalize.setToolTip("No columns selected, or target data not selected")
        else:
            normalize.triggered.connect(lambda: self.dataloader.normalize(columns))
            normalize.setToolTip(f"Normalizes data in selected column{s} by z-score")
        menu.addAction(normalize)
        menu.addSeparator()

        # Remove columns
        remove = QAction(f'Remove', self)
        if len(columns) == 0:
            remove.setDisabled(True)
            remove.setToolTip("No columns selected")
        else:
            remove.triggered.connect(lambda: self.dataloader.remove_columns(columns))
            remove.setToolTip(f"Removes selected column{s}")
        menu.addAction(remove)
        menu.addSeparator()

        # Adjust columns
        adjust = QAction(f'Adjust', self)
        if len(columns) == 0:
            adjust.setDisabled(True)
            adjust.setToolTip("No columns selected")
        else:
            adjust.triggered.connect(lambda: self.adjust_column(columns))
            adjust.setToolTip(f"Selects function and applies to selected column{s}")
        menu.addAction(adjust)
        menu.addSeparator()

        # Reset columns
        reset = QAction('Reset', self)
        if len(columns) == 0:
            reset.setDisabled(True)
            reset.setToolTip("No columns selected")
        else:
            reset.triggered.connect(lambda: self.dataloader.reset(columns))
            reset.setToolTip(f"Resets selected column{s} to original form")
        menu.addAction(reset)
        menu.addSeparator()

        # Undo button
        undo = QAction('Undo', self)
        if type(self.dataloader.previous_data) != pd.DataFrame:
            undo.setDisabled(True)
            undo.setToolTip("No actions to undo")
        else:
            undo.triggered.connect(self.dataloader.undo)
        menu.addAction(undo)

        menu.exec(self.table_view.viewport().mapToGlobal(pos))

class CSVTable(QAbstractTableModel):
    def __init__(self, parent, data:pd.DataFrame):
        super().__init__(parent)
        self._data = data

    def preview_file(self, file:str):
        try:
            data = pd.read_csv(file)
            data = data.fillna("None")
            self.change_data(data)
        except(FileNotFoundError):
            print("File not found")

    def change_data(self, data:pd.DataFrame) -> None:
        self.beginResetModel()
        self._data = data
        self.endResetModel() 

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            item = self._data.iloc[index.row(), index.column()]
            return f"{item:.2f}"

    def headerData(self, section: int, orientation: Qt.Orientation, role) -> Any:
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return self._data.columns[section]
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return self._data.index[section]
        return None

    def rowCount(self, index:QModelIndex) -> int:
        if index.isValid():
            return 0
        return self._data.shape[0]

    def columnCount(self, index:QModelIndex) -> int:
        if index.isValid():
            return 0
        return self._data.shape[1]

class CSVGraphHeader(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.graphs:list[MplCanvas] = []
        self.displayed_columns:list[int] = []

    def make_column_graphs(self):
        self.setMinimumHeight(160)
        self.remove_graphs([i for i, _ in enumerate(self.graphs)])
        table_widget:CSVPreview = self.parent() # type: ignore
        table = table_widget.table_model
        for col in table._data.columns:
            data = table._data[col]
            graph = MplCanvas(self)
            graph.hist(data)
            graph.setMinimumWidth(160)
            self.graphs.append(graph)
        self.adjust_graphs()

    def place_graph(self, col:int, pos:int, header:QHeaderView) -> None:
        self.graphs[col].move(pos, 0)
        if col not in self.displayed_columns:
            self.displayed_columns.append(col)
            self.graphs[col].setMinimumSize(header.sectionSize(col), header.sectionSize(col))
            self.graphs[col].show()

    def remove_graphs(self, cols:list[int]) -> None:
        cols.sort()
        cols.reverse()
        for col in cols:
            self.graphs[col].hide()
            self.graphs.pop(col)
            if col in self.displayed_columns:
                self.displayed_columns.remove(col)
        self.adjust_graphs()

    def adjust_graphs(self):
        table_widget:CSVPreview = self.parent() # type: ignore
        table_view = table_widget.table_view
        start_col = table_view.columnAt(0)
        end_col = table_view.columnAt(table_view.width())
        if end_col < 0:
            end_col = len(self.graphs) - 1
        header = table_view.horizontalHeader()
        self.setFixedWidth(table_view.width())
        for col in range(len(self.graphs)):
            if col >= start_col and col <= end_col:
                self.place_graph(col, header.sectionViewportPosition(col) - header.sectionViewportPosition(start_col) + 30, header)
            elif col in self.displayed_columns:
                self.displayed_columns.remove(col)
                self.graphs[col].hide()
        self.displayed_columns.sort()
        self.update()

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=1.6, height=1.6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.tick_params(axis='both', labelsize=5)
        super().__init__(fig)
        self.setParent(parent)
        
    def hist(self, data):
        self.axes.clear()
        self.axes.hist(data, 15)
        self.draw()

class AdjustDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Adjustment")
        self.setBaseSize(400, 200)

        self.adjustment = []
        layout = QVBoxLayout()

        self.function_list = QListWidget(self)
        for func in DataLoader.transformations:
            self.function_list.addItem(QListWidgetItem(func))

        layout.addWidget(self.function_list)

        self.b_value = QLineEdit()
        self.b_value.setPlaceholderText("Enter value for b")
        validator = QDoubleValidator(top=10000, bottom=-10000, decimals=4)
        self.b_value.setValidator(validator)
        layout.addWidget(self.b_value)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.ok)
        self.buttons.rejected.connect(self.close)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def ok(self):
        if len(self.function_list.selectedIndexes()) != 1:
            return
        self.adjustment = [self.function_list.currentItem().text(), self.b_value.text() if self.b_value.text() != "" else "0"]
        self.accept()

def callback_wrapper(func):
    @wraps(func)
    def wrapper(self, *args):
        func(self, *args)
        if self.callback:
            self.callback()
    return wrapper

class DialogDataLoader(DataLoader):
    def __init__(self, dataset:pd.DataFrame, table:CSVTable, view:QTableView, header:CSVGraphHeader, callback=None):
        super().__init__(dataset)
        self.table = table
        self.view = view
        self.header = header
        self.callback = callback

    def update_header(self, columns:set[str]) -> None:
        column_names = self.data.columns.to_list()
        for column in columns:
            idx = column_names.index(column)
            self.header.graphs[idx].hist(self.data[column])

    def update_table(self, columns:set[str]=set()) -> None:
        self.table.change_data(self.data)
        if len(columns) == 0:
            columns = set(self.data.columns.to_list())
        self.update_header(columns)

    @callback_wrapper
    def normalize(self, columns:set[str]) -> None:
        super().normalize(columns)
        self.update_table(columns)

    @callback_wrapper
    def remove_columns(self, columns: set[str]) -> None:
        column_names = self.table._data.columns.to_list()
        cols = [column_names.index(column) for column in columns]
        super().remove_columns(columns)
        self.table.change_data(self.data)
        self.view.parent().update() # type: ignore
        self.header.remove_graphs(cols)

    @callback_wrapper
    def adjust_columns(self, columns:set[str], adjustment: str, b: str) -> None:
        super().adjust_columns(columns, adjustment, b)
        self.update_table(columns)

    @callback_wrapper
    def reset(self, columns: set[str]) -> None:
        super().reset(columns)
        self.update_table(columns)

    @callback_wrapper
    def undo(self) -> None:
        super().undo()
        self.table.change_data(self.data)
        self.header.make_column_graphs()
        self.header.adjust_graphs()
