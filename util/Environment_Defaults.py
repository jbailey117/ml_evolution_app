# Environment values
MAX_ORGANISMS:int = 8 # max organisms an environment handles
MIN_ORGANISMS:int = 2 # minimum organisms an environment handles
EPOCHS_PER_CYCLE:int = 100 # how many epochs to run on each organism before each cycle


# Organism values
NEURON_ODDS:float = .5 # odds for adding a new neuron to a layer
NEURON_IMMUNITY:int = 1 # rounds an organism is immune to death when born with extra neuron
LAYER_ODDS:float = 0.00 # odds for creating a new layer within a network
LAYER_IMMUNITY:int = 3 # rounds an organism is immune to death when born with extra layer
LEARNING_RATE:float = 0.005 # learning rate for model training
