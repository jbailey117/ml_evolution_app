import pandas as pd
import keras
from pathlib import Path
from EvolutionEnvironment import EvolutionEnvironment
from Organism import Organism

path = Path("../basic_ml/diabetes.csv")
path = path.resolve()
print(path)

df = pd.read_csv(path)
y = df[df.columns[-1]].values
x = df.drop(columns=[df.columns[-1]]).values

test_model = keras.Sequential(layers=[
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=8, activation='leaky_relu'),
    keras.layers.Dense(units=1, activation='leaky_relu'),
])

layer = keras.layers.Dense(units=16, activation='relu')
layer.build(input_shape=(None, 8))
print(layer.get_weights()[0].shape)

test = EvolutionEnvironment(test_model, x, y, epochs_per_cycle=10, learning_rate=0.005) # type: ignore
test.init_sim()

parent_layers = [layer.shape for layer in test.orgs[0].model.get_weights()]
print(parent_layers)
while True:
    new_organism = Organism(test.orgs[0].model, test, False)
    new_org_layers = [layer.shape for layer in new_organism.model.get_weights()]
    print(f"Equivalent: {parent_layers == new_org_layers}")
    if parent_layers != new_org_layers:
        print(new_org_layers)
        break