import numpy as np

from data import PlayerState, Input


def step(players_states: list[PlayerState]) -> list[int]:
    print("Stepping...")
    random_inputs = [i.value[0] for i in list(Input)]
    n = len(players_states)
    return np.random.choice(random_inputs, size=n)

