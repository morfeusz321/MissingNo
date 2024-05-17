import random
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import List, TypeAlias

FloatType = float

# Constants
C_ERROR = 0.25
C_DELTA = 0.25
DEFAULT_ERROR = 200.0
MIN_ERROR = 1e-12


# HeightVector class implementation
@dataclass
class HeightVector:
    coordinates: List[FloatType]
    height: FloatType

    def __sub__(self, other):
        diff_coords = [a - b for a, b in zip(self.coordinates, other.coordinates)]
        return HeightVector(diff_coords, self.height - other.height)

    def __add__(self, other):
        sum_coords = [a + b for a, b in zip(self.coordinates, other.coordinates)]
        return HeightVector(sum_coords, self.height + other.height)

    def __mul__(self, scalar):
        scaled_coords = [a * scalar for a in self.coordinates]
        return HeightVector(scaled_coords, self.height * scalar)

    def len(self):
        return math.sqrt(sum([a ** 2 for a in self.coordinates]) + self.height ** 2)

    def normalized(self):
        norm_len = self.len()
        if norm_len == 0:
            raise ValueError("Cannot normalize zero-length vector")
        return self * (1.0 / norm_len)

    @staticmethod
    def random(n):
        coordinates = [random.random() for _ in range(n)]
        height = random.random()
        return HeightVector(coordinates, height)

    def is_invalid(self):
        return any(math.isnan(c) or math.isinf(c) for c in self.coordinates + [self.height])


class NetworkCoordinate:

    def __init__(self, dimension: int):
        self.height = HeightVector.random(dimension)
        self.error = MIN_ERROR

    def estimated_rtt(self, rhs: 'NetworkCoordinate') -> timedelta:
        rtt_estimate = (self.height - rhs.height).len() / 1000.0
        return timedelta(seconds=rtt_estimate)

    def update(self, rhs: 'NetworkCoordinate', rtt: timedelta) -> 'NetworkCoordinate':
        rtt_ms = rtt.total_seconds() * 1000
        rtt_estimated_ms = self.estimated_rtt(rhs).total_seconds() * 1000.0
        if rtt_ms < 0.0:
            raise ValueError("RTT cannot be negative")

        w = self.error / (self.error + rhs.error)
        error = rtt_ms - rtt_estimated_ms
        es = abs(error) / rtt_ms
        self.error = max((es * C_ERROR) * w + self.error * C_ERROR * (-w + 1.0), MIN_ERROR)
        delta = C_DELTA * w
        self.height = self.height + (self.height - rhs.height).normalized() * delta * error

        if self.height.is_invalid():
            return NetworkCoordinate(len(self.height.coordinates))
        return self


