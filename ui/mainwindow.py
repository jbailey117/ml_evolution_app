from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QMainWindow,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QStackedLayout,
    QWidget,
    QGraphicsProxyWidget,
)
from PySide6.QtCore import (
    Qt,
    QSize,
    QPointF,
)
from PySide6.QtGui import (
    QFont,
    QAction,
    QResizeEvent,
    QShowEvent,
    QPixmap,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from .widgets import (
    DataEditWidget,
    EvolutionEnvironmentGUI,
    FileLoadDialog,
)
import pandas as pd
from pathlib import Path

import keras
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .widgets import EnvConfigMenu

class MainWindow(QMainWindow):
    def __init__(self, size:QSize):
        super().__init__()
        self.setMaximumSize(size)
        self.setMinimumSize(1600, 800)
        self.container = QWidget()
        self._layout = QStackedLayout(self.container)
        self.model:keras.models.Sequential = keras.Sequential(layers=[
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(units=8, activation='relu'),
            keras.layers.Dense(units=8, activation='relu'),
            keras.layers.Dense(units=8, activation='relu'),
            keras.layers.Dense(units=1, activation='leaky_relu'),
        ])
        self.env, self.map, self.scene, self.view = self.initialize_graphics()
        self._layout.addWidget(self.view)
        self.dataEditor = DataEditWidget(self)
        self._layout.addWidget(self.dataEditor)

        self.setCentralWidget(self.container)

        self.dataframes:dict[str, pd.DataFrame] = dict()
        for file in Path("util/data").iterdir():
            if not (file.is_file() or file.suffix != '.csv'):
                continue
            self.dataframes[file.name] = pd.read_csv(file)

        self.initialize_menu()

    def initialize_graphics(self) -> tuple[EvolutionEnvironmentGUI, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView]:
        map = QGraphicsPixmapItem()
        map.setPos(0,0)

        view = QGraphicsView()
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setMinimumSize(self.size().width(), self.size().height())
        #view.setViewport(QOpenGLWidget())
        view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        view.setViewportMargins(0,0,0,0)

        env = EvolutionEnvironmentGUI(map, view, QSize(1, 1), self.model, np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]]), np.array([1, 0]), min_org=2, epochs_per_cycle=10)

        scene = QGraphicsScene()
        view.setScene(scene)
        scene.addItem(map)

        return env, map, scene, view

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.env.pixmap = QPixmap(self.view.size())
        self.env.pixmap.fill(Qt.GlobalColor.black)
        self.map.setPixmap(self.env.pixmap)

    def initialize_menu(self) -> None:
        font = "Serif"
        menu = self.menuBar()
        menu.setFixedHeight(40)
        main_font = QFont(font, 20)
        menu.setFont(main_font)

        subfont = QFont(font, 14)

        ### file section of menubar ###
        ###############################

        file_menu = menu.addMenu('&File')

        # new instance button
        new = QAction('New', self)
        new.triggered.connect(self.new)
        file_menu.addAction(new)
        file_menu.addSeparator()

        # load data button
        load_data = QAction('Load Data', self)
        load_data.triggered.connect(self.load_data)
        file_menu.addAction(load_data)
        file_menu.addSeparator()

        # import model
        import_model = QAction('Import Model', self)
        import_model.triggered.connect(self.import_model)
        file_menu.addAction(import_model)
        file_menu.addSeparator()

        # export model
        export_model = QAction('Export Model', self)
        export_model.triggered.connect(self.export_model)
        file_menu.addAction(export_model)
        file_menu.addSeparator()

        file_menu.setFont(subfont)
        self.file_menu = file_menu

        ### Simulation section of menu bar ###
        ######################################
        sim_menu = menu.addMenu('&Simulation')

        # config env
        config_env = QAction('Configure', self)
        config_env.triggered.connect(self.config_env)
        sim_menu.addAction(config_env)
        sim_menu.addSeparator()

        # start env
        init_env = QAction('Initialize', self)
        init_env.triggered.connect(self.init_env)
        sim_menu.addAction(init_env)
        sim_menu.addSeparator()

        # run env
        run_env = QAction('Run', self)
        run_env.triggered.connect(self.run_env)
        run_env.setDisabled(True)
        sim_menu.addAction(run_env)

        sim_menu.setFont(subfont)
        self.sim_menu = sim_menu

    def new(self) -> None:
        print("creating new instance")

    def load_data(self) -> None:
        self.dataEditor.change_data(True)

    def import_model(self) -> None:
        print("Importing model....")

    def export_model(self) -> None:
        print("exporting model....")

    def config_env(self) -> None:
        config_menu = self.make_config_menu()
        proxy = self.ConfigProxyItem()
        proxy.setWidget(config_menu)
        self.sim_menu.actions()[0].setDisabled(True)
        self.sim_menu.actions()[2].setDisabled(True)
        config_menu.connect(lambda: self.scene.removeItem(proxy))
        config_menu.connect(lambda: self.sim_menu.actions()[0].setDisabled(False))
        config_menu.connect(lambda: self.sim_menu.actions()[2].setDisabled(False))
        if self.sim_menu.actions()[4].isEnabled():
            self.sim_menu.actions()[4].setDisabled(True)
            config_menu.connect(lambda: self.sim_menu.actions()[4].setDisabled(False))
        self.scene.addItem(proxy)

    def init_env(self) -> None:
        dlg = FileLoadDialog(self, list(self.dataframes.keys()))
        if not dlg.exec():
            return
        data = self.dataframes[dlg.file_list.selectedItems()[0].text() if len(dlg.file_list.selectedItems()) != 0 else dlg.file_list.item(0).text()]
        y = data[data.columns[-1]].values
        data = data.drop(columns=data.columns[-1]).values
        self.env.x, self.env.y = data, y # type: ignore
        self.env.start_model = self.model
        self.env.init_sim()
        self.sim_menu.actions()[4].setDisabled(False)

    def run_env(self) -> None:
        self.sim_menu.actions()[4].setDisabled(True)
        self.sim_menu.actions()[2].setDisabled(True)
        self.env.cycle(lambda: self.sim_menu.actions()[4].setDisabled(False), lambda: self.sim_menu.actions()[2].setDisabled(False))

    def defaultScreen(self) -> None:
        self._layout.setCurrentIndex(0)

    def make_config_menu(self) -> "EnvConfigMenu":
        widget = self.env.make_config_menu(parent=self)
        return widget

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.env.draw()

    class ConfigProxyItem(QGraphicsProxyWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setFlag(QGraphicsProxyWidget.GraphicsItemFlag.ItemIsMovable)
            self.setFlag(QGraphicsProxyWidget.GraphicsItemFlag.ItemIsSelectable)

            self.drag_start:QPointF
            self.is_dragging = False

        def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
            super().mousePressEvent(event)

            if not event.isAccepted():
                event.setAccepted(True)
                self.drag_start = event.pos()
                self.is_dragging = True
                super(QGraphicsProxyWidget, self).mousePressEvent(event)

        def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
            if self.is_dragging:
                new_pos = self.pos() + event.pos() - self.drag_start

                rect = self.scene().sceneRect()
                new_x = new_pos.x()
                if new_x > rect.width() - self.rect().width():
                    new_x = rect.width() - self.rect().width()
                elif new_x < 0:
                    new_x = 0
                new_y = new_pos.y()
                if new_y > rect.height() - self.rect().height():
                    new_y = rect.height() - self.rect().height()
                elif new_y < rect.top():
                    new_y = rect.top()
                self.setPos(QPointF(new_x, new_y))
                event.setAccepted(True)
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
            if self.is_dragging:
                self.is_dragging = False
                event.setAccepted(True)
            else:
                super().mouseReleaseEvent(event)