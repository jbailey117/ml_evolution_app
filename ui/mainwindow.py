from PySide6.QtWidgets import (
    QMainWindow,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QStackedLayout,
    QWidget,
)
from PySide6.QtCore import (
    Qt,
    QSize,
)
from PySide6.QtGui import (
    QPixmap,
    QFont,
    QAction,
    QPainter
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from .widgets import (
    DataEditWidget,
    OrganismGUI,
    EvolutionEnvironmentGUI,
)
import pandas as pd

import keras
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self, size:QSize):
        super().__init__()
        self.setMaximumSize(size)
        self.setMinimumSize(480, 270)
        self.container = QWidget()
        self._layout = QStackedLayout(self.container)
        self.model:keras.models.Sequential = keras.Sequential(layers=[
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(units=8, activation='relu'),
            keras.layers.Dense(units=1, activation='leaky_relu'),
        ])
        self.env, self.map, self.scene, self.view = self.initialize_graphics()
        self._layout.addWidget(self.view)

        self.setCentralWidget(self.container)

        self.dataframes:dict[str, pd.DataFrame] = dict()

        self.initialize_menu()
        self.env.init_sim()

    def initialize_graphics(self) -> tuple[EvolutionEnvironmentGUI, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView]:
        map = QGraphicsPixmapItem()
        map.setPos(0,0)

        env = EvolutionEnvironmentGUI(map, self.maximumSize(), self.model, np.array([0, 0]), np.array([0, 0]), min_org=4)

        view = QGraphicsView()
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setMinimumSize(self.size().width(), self.size().height())
        #view.setViewport(QOpenGLWidget())
        view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        view.setViewportMargins(0,0,0,0)

        scene = QGraphicsScene()
        view.setScene(scene)
        scene.addItem(map)

        return env, map, scene, view

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

        file_menu.setFont(subfont)

        ### config section of menubar ###
        ###############################

        config_menu = menu.addMenu('&Configure')

        ## Data settings ##
        data_settings_menu = config_menu.addMenu('&Data Settings')

        # Normalize data
        normalize_data = data_settings_menu.addMenu('&Normalize Data')
        yes = QAction('Yes', self)
        yes.triggered.connect(lambda: self.set_normalize(True))
        no = QAction('No', self)
        no.triggered.connect(lambda: self.set_normalize(False))
        normalize_data.addAction(yes)
        normalize_data.addSeparator()
        normalize_data.addAction(no)
        normalize_data.setFont(subfont)

        # 



        config_menu.setFont(subfont)

    def new(self) -> None:
        print("creating new instance")

    def load_data(self) -> None:
        try:
            dataEditor = DataEditWidget(self)
            self._layout.addWidget(dataEditor) # type: ignore
            self._layout.setCurrentIndex(1)
        except TypeError:
            pass

    def import_model(self) -> None:
        print("Importing model....")

    def export_model(self) -> None:
        print("exporting model....")

    def set_normalize(self, normalize:bool) -> None:
        print(normalize)

    def defaultScreen(self):
        self._layout.setCurrentIndex(0)