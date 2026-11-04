import keras
import numpy as np
from typing import Any, Union, Literal
from types import NoneType

#from ml_evolution_app.util.env_defaults import EPOCHS_PER_CYCLE, LAYER_ODDS, LEARNING_RATE, MAX_ORGANISMS, MIN_ORGANISMS, NEURON_ODDS

from util import Organism, EvolutionEnvironment, EPOCHS_PER_CYCLE, LAYER_ODDS, LEARNING_RATE, MAX_ORGANISMS, MIN_ORGANISMS, NEURON_ODDS
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsView,
)
from PySide6.QtCore import (
    QSize,
    QRectF,
)
from PySide6.QtGui import (
    QColor,
    QPixmap,
    QPainter,
    QPen,
    Qt,
)
from random import randint

NUM_COLS = 4
RESOLUTION = 1000

def invertColor(color: QColor) -> QColor:
    return QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())

def randomizeColor(color: QColor, lower:int, upper:int, range:int) -> QColor:
    return QColor(
        min(color.red() + randint(-range, range), upper) if color.red() >= lower else lower,
        min(color.green() + randint(-range, range), upper) if color.green() >= lower else lower,
        min(color.blue() + randint(-range, range), upper) if color.blue() >= lower else lower,
        color.alpha()
    )

class OrganismGUI(Organism):
    def __init__(self, parent: Union["OrganismGUI", keras.models.Sequential], env: EvolutionEnvironment, max_size:int=RESOLUTION, clone: bool = False) -> None:
        super().__init__(parent, env, clone=clone)
        self.max_size = max_size
        self.color:QColor
        if isinstance(parent, keras.models.Sequential):
            self.color = QColor(randint(100, 255), randint(100, 255), randint(100, 255))
        else:
            self.color = randomizeColor(parent.color, 100, 255, 30)
        self.pixmap, self.neuron_locs = self.init_pixmap()
        self.draw()

    def init_pixmap(self) -> tuple[QPixmap, list[np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]]]]:
        pixmap = QPixmap(self.max_size, self.max_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        pen = QPen(self.color)
        pen.setWidth(round(self.max_size / 250))
        painter.setPen(pen)
        neuron_locs:list[np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]]] = list()
        padding = int(self.max_size / 10)
        horizontal_internal_padding = int((self.max_size - padding * 2) / (len(self.model.layers)+1))
        neuron_size = int(self.max_size / 40)

        for i, layer in enumerate([self.model.get_weights()[0].shape[0], *self.model.layers]):
            length:int = layer.get_weights()[1].shape[0] if isinstance(layer, keras.layers.Layer) else layer
            neuron_locs.append(np.zeros((length, 2)))

            vertical_internal_padding = int((self.max_size - padding * 2) / length - neuron_size)
            for j in range(length):
                rect = QRectF(padding + (i+.5) * horizontal_internal_padding,padding + round((neuron_size + vertical_internal_padding) * (j + .5)), neuron_size, neuron_size)
                painter.drawEllipse(rect)
                neuron_locs[i][j] = np.array([rect.center().x(), rect.center().y()])

        painter.end()

        return pixmap, neuron_locs

    def draw(self) -> None:
        painter = QPainter(self.pixmap)
        pen = QPen(self.color)
        pen.setWidth(round(self.max_size / 500))
        painter.setPen(pen)
        weights = self.model.get_weights()
        for i in range(1, len(self.neuron_locs)):
            for j, endpoint in enumerate(self.neuron_locs[i]):
                new_endpoint =  np.copy(endpoint)
                new_endpoint[0] -= round(self.max_size / 80) + round(self.max_size / 250) - 1
                for k, startpoint in enumerate(self.neuron_locs[i-1]):
                    new_startpoint = np.copy(startpoint)
                    new_startpoint[0] += round(self.max_size / 80) + round(self.max_size / 520) + 1
                    alpha = weights[(i-1)*2][k][j]
                    if alpha < 0:
                        pen.setColor(invertColor(self.color))
                    else:
                        pen.setColor(self.color)
                    painter.setPen(pen)
                    painter.setOpacity(abs(alpha))
                    painter.drawLine(*new_startpoint, *new_endpoint)
        painter.end()


class EvolutionEnvironmentGUI(EvolutionEnvironment):
    def __init__(self, map:QGraphicsPixmapItem, view:QGraphicsView, size:QSize, start_model: keras.Sequential, X: np.ndarray[tuple[Any, ...], np.dtype[Any]], Y: np.ndarray[tuple[Any, ...], np.dtype[Any]], test_size: float = 0.2, epochs_per_cycle: int = EPOCHS_PER_CYCLE, min_org: int = MIN_ORGANISMS, max_org: int = MAX_ORGANISMS, neuron_odds: float = NEURON_ODDS, layer_odds: float = LAYER_ODDS, learning_rate: float = LEARNING_RATE, loss=keras.losses.MeanSquaredError, kill_count: int = 2) -> None:
        super().__init__(start_model, X, Y, OrganismGUI, test_size, epochs_per_cycle, min_org, max_org, neuron_odds, layer_odds, learning_rate, loss, kill_count)
        self.orgs:list[Union[OrganismGUI, NoneType]]
        self.map = map
        self.view = view
        self.pixmap = QPixmap(size)
        self.pixmap.fill(Qt.GlobalColor.black)
        map.setPixmap(self.pixmap)

    def init_sim(self) -> None:
        super().init_sim()
        self.draw()

    def draw(self) -> None:
        self.pixmap.fill(Qt.GlobalColor.black)
        if len(self.orgs) == 0:
            return
        painter = QPainter(self.pixmap)
        pixmap_size = self.view.size().width()//NUM_COLS
        pixmap_height = self.view.size().height()//2
        for i, model in enumerate(self.orgs):
            if model == None:
                continue
            painter.drawPixmap(pixmap_size * (i % NUM_COLS), pixmap_height * (i // NUM_COLS), pixmap_size, pixmap_height, model.pixmap)
        self.draw_borders(painter)
        painter.end()
        self.map.setPixmap(self.pixmap)

    def draw_borders(self, painter:QPainter) -> None:
        border_pen = QPen(Qt.GlobalColor.gray)
        border_pen.setWidth(3)
        painter.setPen(border_pen)
        spacing = self.view.width()//NUM_COLS
        for i in range(1, NUM_COLS):
            painter.drawLine(i * spacing, 0, i * spacing, self.view.size().height())
        painter.drawLine(0, self.view.size().height()//2, self.view.size().width(), self.view.size().height()//2)