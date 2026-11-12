from keras.activations import relu
from keras.optimizers import Adam
import keras
import pandas as pd
import numpy as np
from typing import Union, TYPE_CHECKING, overload
from types import NoneType
from collections.abc import Sequence
from random import random

if TYPE_CHECKING:
    from . import EvolutionEnvironment
# from Environment_Defaults import NEURON_ODDS, LAYER_ODDS
from .Environment_Defaults import NEURON_ODDS, LAYER_ODDS

class Organism: 
    def __init__(self, parent:Union["Organism", keras.models.Sequential], env:"EvolutionEnvironment", clone:bool=False) -> None:
        self.env = env
        self.alive_turns = 0
        self.model = self.build_model(parent.model if isinstance(parent, Organism) else parent, clone)
        self.model.compile(optimizer=Adam(learning_rate=env.learning_rate), loss=env.loss, metrics=['Accuracy'])

    def build_model(self, parent:keras.models.Sequential, clone:bool=False) -> keras.models.Sequential:
        if len(parent.layers) <= 1:
            raise ValueError("parent model doesn't have enough layers, need at least 2")
        if clone:
            model = keras.models.Sequential.from_config(parent.get_config())
            model.set_weights(parent.get_weights())
            return model
        model = keras.models.Sequential()
        for i in range(len(parent.layers)):
            layer = self.new_layer(
                parent.layers[i],
                input_shape=model.layers[-1].get_weights()[0].shape[1] if i else parent.layers[i].get_weights()[0].shape[0],
                clone=i>=(len(parent.layers)-1))
            model.add(layer)
            if i < len(parent.layers) -1 and random() <= self.env.evolve_odds['layer']:
                model.add(self.new_layer(
                    model.layers[-1],
                    input_shape=model.layers[-1].get_weights()[0].shape[1],
                    clone=False,
                    mutation=True,
                ))
                self.alive_turns += 3
        model.build(input_shape=(parent.get_weights()[0].shape[0]))
        return model  

    @overload
    def get_weights(self, parent_layer:keras.layers.Dense): ...

    @overload
    def get_weights(self, parent_layer:keras.layers.Dense, input_shape:int): ...

    def get_weights(self, parent_layer:keras.layers.Dense, input_shape:int=None) -> list[np.ndarray]: # type: ignore
        if type(input_shape) == int:
            weights:list[np.ndarray] = list()
            rng = np.random.default_rng()
            weights.append(np.array([rng.random(parent_layer.units) * 2 - 1 for _ in range(input_shape)]))
            weights.append(rng.random(parent_layer.units) * 2 - 1)
            return weights
        else:
            return parent_layer.get_weights()
        
    def new_layer(self, parent_layer:keras.layers.Dense, input_shape:int, clone:bool=False, mutation:bool=False) -> keras.layers.Dense:
        weights = self.get_weights(parent_layer, input_shape=input_shape if mutation else None) # type: ignore
        if not clone:
            if random() <= (self.env.evolve_odds['neuron']):
                weights = self.add_neuron(weights)
                self.alive_turns += 1
        layer = keras.layers.Dense(units=weights[0].shape[1], activation=parent_layer.activation)
        layer.build(input_shape=(None, input_shape))
        while weights[0].shape[0] < input_shape:
            weights = self.add_weight(weights)
        layer.set_weights(weights)
        return layer

    def add_neuron(self, weights:list[np.ndarray]) -> list[np.ndarray]:
        new_weights = list([np.array([np.zeros(weights[0].shape[1]+1) for _ in range(weights[0].shape[0])]), weights[1]])
        for i in range(weights[0].shape[0]):
            new_weights[0][i] = np.append(weights[0][i], 2 * (random()-0.5))
        new_weights[1] = np.append(new_weights[1], new_weights[1][int(random() * len(new_weights[1]))])
        return new_weights

    def add_weight(self, weights:list[np.ndarray]) -> list[np.ndarray]:
        old_shape = weights[0].shape
        weights[0] = np.append(weights[0], (np.array([2 * (random()-0.5) for _ in range(old_shape[1])])))
        weights[0] = weights[0].reshape(old_shape[0]+1, old_shape[1])
        return weights
    