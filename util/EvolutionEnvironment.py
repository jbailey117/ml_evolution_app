from typing import Union, TypedDict
from types import NoneType
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from multiprocessing import Pool
from .Organism import Organism
from .Environment_Defaults import EPOCHS_PER_CYCLE, MIN_ORGANISMS, MAX_ORGANISMS, NEURON_ODDS, LAYER_ODDS, LEARNING_RATE
# from Organism import Organism
# from Environment_Defaults import *

class EvolveOdds(TypedDict):
    neuron: float
    layer: float

class EvolutionEnvironment:
    def __init__(self, start_model:keras.models.Sequential, X:np.ndarray, Y:np.ndarray, organism:type[Organism]=Organism, test_size:float=0.2, epochs_per_cycle:int=EPOCHS_PER_CYCLE, batch_size:int=24, min_org:int=MIN_ORGANISMS, max_org:int=MAX_ORGANISMS, neuron_odds:float=NEURON_ODDS, layer_odds:float=LAYER_ODDS, learning_rate:float=LEARNING_RATE, loss='mean_squared_error', kill_count:int=2) -> None:
        self.x = X
        self.y = Y
        self.test_size = test_size
        self.Organism = organism
        self.epochs_per_cycle = epochs_per_cycle
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.evolve_odds: EvolveOdds = {
            'neuron': neuron_odds,
            'layer': layer_odds,
        }
        self.min_org = min_org
        self.max_org = max_org
        self.kill_count = kill_count

        self.orgs:list[Union[Organism, NoneType]] = list()
        self.start_model = start_model

    def init_sim(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size)
        self.orgs.clear()
        self.orgs.append(self.Organism(self.start_model, self, clone=True))
        for i in range(1, self.min_org):
            self.orgs.append(self.Organism(self.orgs[i-1], self)) # type: ignore
        for _ in range(len(self.orgs), self.max_org):
            self.orgs.append(None)

    def run_sim(self, runs:int) -> None:
        for _ in range(runs):
            self.cycle()

    def cycle(self) -> np.ndarray:
        losses:np.ndarray = np.zeros(shape=(len(self.orgs),))

        # train each model
        for i, org in enumerate(self.orgs):
            if org == None:
                losses[i] = -1
                continue
            org.model.fit(self.x_train, self.y_train, epochs=self.epochs_per_cycle, batch_size=self.batch_size)
            losses[i] = org.model.evaluate(self.x_test, self.y_test, batch_size=24)[0]

        # get list of models in order of test set, then "kill" bad models and "birth" new models from best models
        parents, deaths = self.get_viable_orgs(losses)
        for death in deaths:
            self.orgs[int(death)] = None
        for parent in parents:
            for i in range(len(self.orgs)):
                if self.orgs[i] == None:
                    self.orgs[i] = self.Organism(self.orgs[int(parent)], self) # type: ignore
                    break
        for org in self.orgs:
            if org:
                org.alive_turns -= 1
        return losses
                
    def cycle_parallel(self) -> None:
        tasks = [(self.x_train, self.y_train, self.epochs_per_cycle, org.model.get_config(), org.model.get_weights()) if type(org) != None else None for org in self.orgs] # type: ignore
        with Pool(processes=1) as pool:
            new_weights = pool.map(_parallel_fit, tasks) # type: ignore
        for i, weights in enumerate(new_weights):
            self.orgs[i].model.set_weights(weights) # type: ignore

    def get_viable_orgs(self, losses:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sorted_losses = np.ndarray.copy(losses)[np.where(losses != -1)]
        sorted_losses.sort(descending=True)

        parent_orgs:np.ndarray = np.array([], dtype=int)
        death_orgs:np.ndarray = np.array([], dtype=int)

        if len(sorted_losses) == self.max_org:
            for loss in sorted_losses:
                if len(death_orgs) == self.kill_count:
                    break
                i = np.argmax(losses == loss)
                if i in death_orgs:
                    i = len(losses) - 1 - np.argmax(losses[::-1][1:]==loss)
                if self.orgs[i].alive_turns > 0: # type: ignore
                    continue
                death_orgs= np.append(death_orgs, i)

        parents = sorted_losses[-2:len(sorted_losses)]
        for parent in parents:
            i = np.argmax(losses == parent)
            if i in parent_orgs or i in death_orgs:
                i = len(losses) - 1 - np.argmax(losses[::-1]==parent)
            parent_orgs = np.append(parent_orgs, i)

        return parent_orgs, death_orgs

def _parallel_fit(args:Union[tuple[np.ndarray, np.ndarray, int, dict, np.ndarray], NoneType]) -> np.ndarray | None:
    if type(args) == NoneType:
        return None
    x_train, y_train, epochs, config, weights = args
    model = keras.models.Sequential.from_config(config)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=keras.losses.mean_squared_error, metrics=[keras.metrics.Accuracy])
    model.set_weights(weights)
    model.fit(x_train, y_train, epochs=epochs)
    return model.get_weights() # type: ignore
