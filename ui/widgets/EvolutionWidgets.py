import keras
import numpy as np
from typing import Any, Union, Literal, TypedDict
from types import NoneType
from collections.abc import Callable

#from ml_evolution_app.util.env_defaults import EPOCHS_PER_CYCLE, LAYER_ODDS, LEARNING_RATE, MAX_ORGANISMS, MIN_ORGANISMS, NEURON_ODDS

from util import Organism, EvolutionEnvironment, EPOCHS_PER_CYCLE, LAYER_ODDS, LEARNING_RATE, MAX_ORGANISMS, MIN_ORGANISMS, NEURON_ODDS
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsView,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QDialogButtonBox,
    QMessageBox,
)
from PySide6.QtCore import (
    QObject,
    QSize,
    QRectF,
    QThread,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QPixmap,
    QPainter,
    QPen,
    Qt,
    QStaticText,
    QFont,
    QIntValidator,
    QDoubleValidator,
    QValidator,
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
            self.color = randomizeColor(parent.color, 100, 255, 15)
        self.pixmap, self.neuron_locs = self.init_pixmap()

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
                    painter.setPen(Qt.GlobalColor.black)
                    painter.drawLine(*new_startpoint, *new_endpoint)
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
    def __init__(self, map:QGraphicsPixmapItem, view:QGraphicsView, size:QSize, start_model: keras.Sequential, X: np.ndarray[tuple[Any, ...], np.dtype[Any]], Y: np.ndarray[tuple[Any, ...], np.dtype[Any]], test_size: float = 0.2, epochs_per_cycle: int = EPOCHS_PER_CYCLE, batch_size:int=24, min_org: int = MIN_ORGANISMS, max_org: int = MAX_ORGANISMS, neuron_odds: float = NEURON_ODDS, layer_odds: float = LAYER_ODDS, learning_rate: float = LEARNING_RATE, loss='mean_squared_error', kill_count: int = 2) -> None:
        super().__init__(start_model, X, Y, OrganismGUI, test_size, epochs_per_cycle, batch_size, min_org, max_org, neuron_odds, layer_odds, learning_rate, loss, kill_count)
        self.orgs:list[Union[OrganismGUI, NoneType]]
        self.map = map
        self.view = view
        self.threads = []
        self.pixmap = QPixmap(size)
        self.pixmap.fill(Qt.GlobalColor.black)
        map.setPixmap(self.pixmap)

    def init_sim(self) -> None:
        super().init_sim()
        for attr in ['parents', 'deaths', 'losses']:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue
        self.draw()

    def cycle(self, *callbacks):
        self.callback = callbacks
        self.losses:np.ndarray = np.zeros(shape=(len(self.orgs),))
        
        # train each model
        for i, org in enumerate(self.orgs):
            if org == None:
                self.losses[i] = -1
                continue
            thread = self.CycleThread(i, self)
            thread.progress.connect(self.update_weights)
            thread.finished.connect(lambda t=thread: self._clean_thread(t))
            thread.finished.connect(thread.deleteLater)
            thread.start()
            self.threads.append(thread)

    def get_viable_orgs(self, losses: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> tuple[np.ndarray[tuple[Any, ...], np.dtype[Any]], np.ndarray[tuple[Any, ...], np.dtype[Any]]]:
        self.parents, self.deaths = super().get_viable_orgs(losses)
        return self.parents, self.deaths

    def update_weights(self, i:int, weights:list[np.ndarray]) -> None:
        if self.orgs[i] == None:
            return
        self.orgs[i].model.set_weights(weights) # type: ignore
        self.losses[i] = self.orgs[i].model.evaluate(self.x_test, self.y_test, batch_size=24)[0] # type: ignore

    def _clean_thread(self, thread:"CycleThread") -> None:
        self.threads.remove(thread)
        if len(self.threads) == 0:
            # get list of models in order of test set, then "kill" bad models and "birth" new models from best models once each training thread is done
            parents, deaths = self.get_viable_orgs(self.losses)
            self.get_new_generation(parents, deaths)
            self.draw()
            for callback in self.callback:
                callback()

    def draw(self) -> None:
        self.pixmap = QPixmap(self.view.size())
        self.pixmap.fill(Qt.GlobalColor.black)
        if len(self.orgs) == 0:
            self.map.setPixmap(self.pixmap)
            return
        painter = QPainter(self.pixmap)
        painter.setPen(Qt.GlobalColor.white)
        pixmap_size = self.view.size().width()//NUM_COLS
        pixmap_height = self.view.size().height() // (round(self.max_org / NUM_COLS + 0.499))
        for i, model in enumerate(self.orgs):
            if model == None:
                continue
            model.draw()
            painter.drawPixmap(pixmap_size * (i % NUM_COLS), pixmap_height * (i // NUM_COLS), pixmap_size, round(pixmap_height*0.95), model.pixmap)
            try:
                if self.losses[i] != -1:
                    text = QStaticText(f"Loss: {self.losses[i]:.5f}")
                    text.prepare(font=QFont("Arial", 25))
                    painter.drawStaticText(pixmap_size * (i % NUM_COLS) + pixmap_size // 10, pixmap_height * (i // NUM_COLS) + round(pixmap_height*0.93), text)
                if self.orgs[i].alive_turns > 0: # type: ignore
                    text = QStaticText(f"Immune Cycles: {self.orgs[i].alive_turns}") # type: ignore
                    text.prepare(font=QFont("Arial", 25))
                    painter.drawStaticText(pixmap_size * (i % NUM_COLS) + pixmap_size // 10, pixmap_height * (i // NUM_COLS) + round(pixmap_height * 0.96), text)
            except AttributeError:
                pass
        self.draw_borders(painter) 
        try:
            painter.setPen(Qt.GlobalColor.green)
            for i in self.parents:
                painter.drawRect(pixmap_size * (i % NUM_COLS)+2, pixmap_height * (i // NUM_COLS)+2, pixmap_size-4, pixmap_height-4)
        except AttributeError:
            pass
        try:
            painter.setPen(Qt.GlobalColor.red)
            for i in self.deaths:
                painter.drawRect(pixmap_size * (i % NUM_COLS)+2, pixmap_height * (i // NUM_COLS)+2, pixmap_size-4, pixmap_height-4)
        except AttributeError:
            pass
        painter.end()
        self.map.setPixmap(self.pixmap)

    def draw_borders(self, painter:QPainter) -> None:
        border_pen = QPen(Qt.GlobalColor.gray)
        border_pen.setWidth(3)
        painter.setPen(border_pen)
        width = self.view.width()//NUM_COLS
        num_rows = round(self.max_org / NUM_COLS + 0.499)
        height = self.view.size().height() // num_rows
        for i in range(1, NUM_COLS):
            painter.drawLine(i * width, 0, i * width, self.view.size().height())
        for i in range(1, num_rows):
            painter.drawLine(0, i * height, self.view.size().width(), i * height)

    def make_config_menu(self, callback:Callable=None, parent=None) -> "EnvConfigMenu": # type: ignore
        return EnvConfigMenu(self, callback=callback, parent=parent)

    class CycleThread(QThread):
        progress = Signal(int, object)

        def __init__(self, i, env:"EvolutionEnvironmentGUI") -> None:
            super().__init__()
            self.i = i
            self.env = env

        def run(self) -> None:
            model = keras.models.Sequential.from_config(self.env.orgs[self.i].model.get_config())
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.env.learning_rate), loss=self.env.loss, metrics=['Accuracy'])
            model.fit(self.env.x_train, self.env.y_train, batch_size=self.env.batch_size, epochs=self.env.epochs_per_cycle)
            self.progress.emit(self.i, model.get_weights())


class EnvConfigMenu(QWidget):
    def __init__(self, env:EvolutionEnvironmentGUI, callback:Callable=None, parent=None): # type: ignore
        super().__init__()
        self._parent = parent
        self.env = env
        self.callbacks:list[Callable] = []
        if callback:
            self.callbacks.append(callback)

        font = QFont("Arial", 12)

        layout = QVBoxLayout()
        title = QLabel("Environment Config")
        title.setFont(font)
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        prob_validator = ProbabilityValidator()

        # Test size 
        self.test_size = self.make_line_edit("Test Size", self.env.test_size, font, prob_validator)
        layout.addWidget(self.add_label(self.test_size, "Test Size", font))

        # Epochs per Cycle
        self.epochs_per_cycle = self.make_line_edit("Epochs/Cycle", self.env.epochs_per_cycle, font, QIntValidator(bottom=1, top=10000))
        layout.addWidget(self.add_label(self.epochs_per_cycle, "Epochs/Cycle", font))

        # Batch Size
        self.batch_size = self.make_line_edit("Batch Size", self.env.batch_size, font, QIntValidator(bottom=1))
        layout.addWidget(self.add_label(self.batch_size, "Batch Size", font))

        # Min Org
        self.min_org = self.make_line_edit("Mininimum Organisms", self.env.min_org, font, QIntValidator(bottom=1))
        layout.addWidget(self.add_label(self.min_org, "Min Org", font))

        # Max Org
        self.max_org = self.make_line_edit("Maximum Organism", self.env.max_org, font, QIntValidator(bottom=1))
        layout.addWidget(self.add_label(self.max_org, "Max Org", font))

        # Kill Count
        self.kill_count = self.make_line_edit("Kill Count", self.env.kill_count, font, QIntValidator(bottom=0))
        layout.addWidget(self.add_label(self.kill_count, "Kill Count", font))

        # New Neuron Odds
        self.neuron_odds = self.make_line_edit("New Neuron Odds", self.env.evolve_odds["neuron"], font, prob_validator)
        layout.addWidget(self.add_label(self.neuron_odds, "New Neuron Odds", font))

        # New Layer Odds
        self.layer_odds = self.make_line_edit("New Layer Odds", self.env.evolve_odds["layer"], font, prob_validator)
        layout.addWidget(self.add_label(self.layer_odds, "New Layer Odds", font))

        # Learning rate
        self.learning_rate = self.make_line_edit("Learning Rate", self.env.learning_rate, font, QDoubleValidator(bottom=0, decimals=7))
        layout.addWidget(self.add_label(self.learning_rate, "Learning Rate", font))

        # Loss Type
        self.loss = self.make_line_edit("Loss Type (str)", self.env.loss, font)
        layout.addWidget(self.add_label(self.loss, "Loss Type", font))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    class Values(TypedDict):
        test_size: float
        epochs_per_cycle: int
        batch_size: int
        min_org: int
        max_org: int
        kill_count: int
        neuron_odds: float
        layer_odds: float
        loss: str

    @staticmethod
    def add_label(widget:QWidget, text:str, font:QFont) -> QWidget:
        line_layout = QHBoxLayout()

        label = QLabel(text)
        label.setFont(font)
        line_layout.addWidget(label)
        line_layout.addWidget(widget)

        new_widget = QWidget()
        new_widget.setLayout(line_layout)
        return new_widget

    @staticmethod
    def make_line_edit(item:str, default, font:QFont, validator=None) -> QLineEdit:
        line_edit = QLineEdit(str(default))
        line_edit.setFont(font)
        line_edit.setPlaceholderText(f"Enter {item}")
        if validator != None:
            line_edit.setValidator(validator)
        return line_edit

    def get_values(self) -> Values:
        values:"EnvConfigMenu.Values" = {
            "test_size": float(self.test_size.text()),
            "epochs_per_cycle": int(self.epochs_per_cycle.text()),
            "batch_size": int(self.batch_size.text()),
            "min_org": int(self.min_org.text()),
            "max_org": int(self.max_org.text()),
            "kill_count": int(self.kill_count.text()),
            "neuron_odds": float(self.neuron_odds.text()),
            "layer_odds": float(self.layer_odds.text()),
            "loss": self.loss.text(),
        }
        for key in values.keys():
            if values[key] == "":
                values[key] = getattr(self.env, key)
        values["loss"] = values["loss"].lower()
        return values

    def validate_values(self, values:"EnvConfigMenu.Values") -> list[str]:
        errors:list[str] = []
        if values["min_org"] > values['max_org']:
            errors.append("min org cannot be greater than max org")
        if values['kill_count'] > values['max_org']:
            errors.append("kill count cannot be greater than max org")
        if not values["test_size"] > 0:
            errors.append("test size cannot be 0")
        try:
            keras.losses.get(values["loss"])
        except ValueError:
            errors.append("invalid loss type")
        return errors

    def accept(self):
        values = self.get_values()
        errors = self.validate_values(values)
        if len(errors) != 0:
            QMessageBox.critical(self._parent, "Invalid Input", str.join("\n", errors), buttons=QMessageBox.StandardButton.Ok)
            return
        for key in values.keys():
            self.env.__setattr__(key, values[key])
        if len(self.callbacks) == 0:
            self.hide()
            return
        for callback in self.callbacks:
            callback()

    def reject(self):
        if len(self.callbacks) == 0:
            self.hide()
            return
        for callback in self.callbacks:
            callback()

    def connect(self, callback:Callable[[], None]) -> None:
        if callback not in self.callbacks:
            self.callbacks.append(callback)

class ProbabilityValidator(QDoubleValidator):
    def __init__(self) -> None:
        super().__init__(bottom=0, top=1, decimals=4)
        self.setNotation(QDoubleValidator.Notation.StandardNotation)

    def validate(self, input: str, pos: int) -> object:
        if input in ["", "."]:
            return QValidator.State.Intermediate, input, pos
        if input == "-":
            return QValidator.State.Invalid, input, pos
        state, input, pos = super().validate(input, pos) # type: ignore

        if state in (QValidator.State.Acceptable, QValidator.State.Intermediate):
            try:
                val = float(input)
                if val < self.bottom() or val > self.top():
                    return QValidator.State.Invalid, input, pos
            except ValueError:
                return QValidator.State.Invalid, input, pos

        return state, input, pos