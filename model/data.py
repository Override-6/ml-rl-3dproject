from enum import Enum
from typing import List

LASERS_PER_PLAYER = 5
NB_COMPONENT_TYPES = 5

class Input(Enum):
    Noop = 0,
    Forward = 1,
    Backward = 2,
    Left = 4,
    Right = 8,
    Jump = 16,
    TurnRight = 32,
    TurnLeft = 64,


class LaserHit:
    def __init__(self, distance: float, component_type: int):
        self.distance = distance
        self.component_type = component_type

    def __repr__(self):
        return f"LaserHit(distance={self.distance}, type={self.component_type})"

class Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"

class PlayerState:
    def __init__(self, reward: float, done: bool, position: Vec3, ang_velocity: Vec3, lin_velocity: Vec3, rotation: Vec3, lasers: List[LaserHit]):
        self.reward = reward
        self.done = done
        self.position = position
        self.ang_velocity = ang_velocity
        self.lin_velocity = lin_velocity
        self.rotation = rotation
        self.lasers = lasers

    def __repr__(self):
        return f"PlayerState(reward={self.reward}, pos={self.position}, ang_vel={self.ang_velocity}, lin_vel={self.lin_velocity}, rot={self.rotation}, lasers={self.lasers})"
