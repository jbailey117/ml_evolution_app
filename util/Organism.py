from keras.activations import relu
from keras.optimizers import Adam
import keras
import pandas as pd
import numpy as np
from typing import Union, TYPE_CHECKING
from types import NoneType
from collections.abc import Sequence
from random import random

if TYPE_CHECKING:
    from . import EvolutionEnvironment
#from env_defaults import NEURON_ODDS, LAYER_ODDS
from .env_defaults import NEURON_ODDS, LAYER_ODDS

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
        model.add(parent.layers[0])
        for i in range(1, len(parent.layers)):
            weights = parent.layers[i].get_weights()
            seed = random()
            if i and i < len(parent.layers)-2 and seed <= (self.env.evolve_odds['neuron'] if type(self.env) != NoneType else NEURON_ODDS):
                weights = self.add_neuron(weights)
            layer = keras.layers.Dense(units=weights[0].shape[1], activation=parent.layers[i].activation)
            input_shape = model.layers[-1].get_weights()[0].shape[1]
            layer.build(input_shape=(None, input_shape))
            if weights[0].shape[0] < input_shape:
                weights = self.add_weight(weights)
            layer.set_weights(weights)
            model.add(layer)
        model.build(input_shape=(parent.get_weights()[0].shape[0]))
        return model  

    def new_layer(self, parent_layer:keras.layers.Dense, input_shape:int, output_shape:int) -> keras.layers.Dense:
        weights:list[np.ndarray] = list()
        rng = np.random.default_rng()
        weights.append(np.array([rng.random(output_shape) for _ in range(input_shape)]))
        weights.append(rng.random(output_shape))
        seed = random()
        if seed <= (self.env.evolve_odds['neuron'] if type(self.env) != NoneType else NEURON_ODDS):
            weights = self.add_neuron(weights)
        layer = keras.layers.Dense(units=output_shape, activation=parent_layer.activation)
        layer.build(input_shape=(None, input_shape))
        if weights[0].shape[0] < input_shape:
            weights = self.add_weight(weights)
        layer.set_weights(weights)
        return layer

    def add_neuron(self, weights:list[np.ndarray]) -> list[np.ndarray]:
        new_weights = list([np.array([np.zeros(weights[0].shape[1]+1) for _ in range(weights[0].shape[0])]), weights[1]])
        for i in range(weights[0].shape[0]):
            new_weights[0][i] = np.append(weights[0][i], random())
            #new_weights[0][i] = np.append(weights[0][i], 0.)
        new_weights[1] = np.append(new_weights[1], new_weights[1][int(random() * len(new_weights[1]))])
        #new_weights[1] = np.append(new_weights[1], 0.)
        return new_weights

    def add_weight(self, weights:list[np.ndarray]) -> list[np.ndarray]:
        old_shape = weights[0].shape
        weights[0] = np.append(weights[0], (np.array([random() for _ in range(old_shape[1])])))
        weights[0] = weights[0].reshape(old_shape[0]+1, old_shape[1])
        return weights
    
    def new_layer_old(self, layers:Sequence[keras.layers.Dense], units:int) -> keras.layers.Dense:
        activation = layers[round(random() * (len(layers)-1))].activation
        layer = keras.layers.Dense(units=units, activation=activation)
        layer.build(input_shape=(None, units))

        return layer

if __name__ == "__main__":
    #test = Organism(keras.models.Sequential([keras.layers.Dense(units=5, activation=relu)]))
    pass