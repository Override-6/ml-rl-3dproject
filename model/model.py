from keras import Sequential

from data import PlayerState, Input
import numpy as np

def create_model():
    model = Sequential()

    return model

class ModelAgent:
    def __init__(self):
        self.model = create_model()
        self.target_model =

def step(players_states: list[PlayerState]) -> list[int]:
    print("Stepping...")
    random_inputs = [i.value[0] for i in list(Input)]
    n = len(players_states)
    return np.random.choice(random_inputs, size=n)

