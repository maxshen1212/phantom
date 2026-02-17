"""Minimal epic_kitchens.hoa.types for pickle compatibility"""

from enum import Enum, unique
from dataclasses import dataclass
import numpy as np


@unique
class HandSide(Enum):
    LEFT = 0
    RIGHT = 1


@unique
class HandState(Enum):
    NO_CONTACT = 0
    SELF_CONTACT = 1
    ANOTHER_PERSON = 2
    PORTABLE_OBJECT = 3
    STATIONARY_OBJECT = 4


@dataclass
class FloatVector:
    x: np.float32
    y: np.float32


@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass
class HandDetection:
    bbox: BBox
    score: np.float32
    state: HandState
    side: HandSide
    object_offset: FloatVector
