import pandas as pd
from collections.abc import Callable
import numpy as np
from sklearn.preprocessing import PowerTransformer
from typing import Literal


def Powertransform(data:np.ndarray, b:float=0., method:Literal['yeo-johnson', 'box-cox']='yeo-johnson') -> np.ndarray:
    if not len(data):
        return np.ndarray([])
    data = data + b
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    transformer = PowerTransformer(method=method, standardize=True)
    transformer = transformer.fit(data)
    new_data = transformer.transform(data)
    return new_data


class DataLoader:
    transformations:dict[str, Callable] = {
        "log(x + b)": lambda x, b: np.log10(x + b),
        "ln(x + b)": lambda x, b: np.log(x + b),
        "x ^ b": lambda x, b: x ** b,
        "1 / (x + b)": lambda x, b: 1 / (x + b),
        "asinh(x)": lambda x, b: np.asinh(x + b),
        "Yeo-Johnson": Powertransform,
        "x + b": lambda x, b: x + b,
    }

    def __init__(self, dataset:pd.DataFrame):
        self.data = dataset
        self.previous_data = None
        self.original_data = dataset.to_dict()
        self.active_transformations = dict.fromkeys(dataset.columns.to_list(), [])
        self.is_normalized = False
        self.target_data = None
        self.training_data = None
        self.exported = False

    def normalize(self, columns:set[str]) -> None:
        self.previous_data = self.data.copy()
        for column in columns:
            if type(self.target_data) == pd.DataFrame and column == self.target_data.name: # type: ignore
                continue
            values = self.data[column].values
            if type(values) != np.ndarray:
                print("values not in numpy array")
                continue
            mean = np.sum(values) / len(values)
            stdev = np.std(values, mean=mean)
            if stdev == 0.:
                continue
            self.data[column] = self.data[column].apply(lambda x: (x - mean) / stdev)
        self.is_normalized = True

    def remove_columns(self, columns:set[str]) -> None:
        self.previous_data = self.data
        self.data = self.data.drop(columns=columns)

    def adjust_columns(self, columns:set[str], adjustment:str, b_value:str) -> None:
        b = np.float64(b_value)
        if adjustment not in self.transformations.keys():
            raise KeyError("Error: invalid adjustment")
        self.previous_data = self.data.copy()
        func = self.transformations[adjustment]
        for column in columns:
            self.data[column] = func(self.data[column].values, b)

    def set_target(self, column:str) -> None:
        self.target_data = self.data[column]

    def set_train(self, columns:set[str]) -> None:
        self.training_data = self.data[columns]

    def reset(self, columns:set[str]) -> None:
        for column in columns:
            self.data[column] = self.original_data[column]

    def undo(self):
        if type(self.previous_data) != pd.DataFrame:
            raise ValueError("Error: No previous data ")
        self.data = self.previous_data
        self.previous_data = None

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        if type(self.training_data) != pd.DataFrame or type(self.target_data) != pd.DataFrame:
            raise KeyError("Error: dataloader needs training and target data")
        try:
            training = self.training_data.to_numpy(dtype=np.float64)
            target = self.target_data.to_numpy(dtype=np.float64)
            self.exported = True
            return training, target
        except:
            raise ValueError("Error: data is not all number values")