import random
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import numpy as np

FloatType = float

# Constants
C_ERROR = 0.25
C_DELTA = 0.25
DEFAULT_ERROR = 200.0
MIN_ERROR = 1e-12
STABLE = 0.3
F_C = 75
F_MAX = 100


class CoordinatesAttackStrategy:
    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        """Perform attack on the network."""
        raise NotImplementedError("This method should be overridden.")


class InflationAttack(CoordinatesAttackStrategy):
    def attack(self, node: 'NetworkNode'):
        """Simulate inflation by setting node's coordinates unusually high."""
        node.coordinates.coordinates = [1000, 1000, 1000]  # Example of inflated coordinates


class DeflationAttack(CoordinatesAttackStrategy):
    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        """Simulate deflation by setting node's coordinates near the origin."""
        node.coordinates.coordinates = [1, 1, 1]  # Example of deflated coordinates
        return node


class NoAttack(CoordinatesAttackStrategy):

    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        return node


class OscillationAttack(CoordinatesAttackStrategy):
    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        """Change coordinates randomly and introduce random delays."""
        node.coordinates.coordinates = [random.randint(-100, 100), random.randint(-100, 100), random.randint(-100, 100)]
        return node


class FrogBoilingAttack(CoordinatesAttackStrategy):
    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        """Gradually change coordinates to move far from the correct position."""
        step = 5
        node.coordinates.coordinates = [x + step for x in node.coordinates.coordinates]
        return node


class NetworkPartitionAttack(CoordinatesAttackStrategy):
    def attack(self, node: 'NetworkNode') -> 'NetworkNode':
        """Group of nodes slowly moves to opposite directions."""
        direction = random.choice([(5, 5, 5), (-5, -5, -5)])
        node.coordinates.coordinates = list(c + d for c, d in zip(node.coordinates.coordinates, direction))
        return node


# HeightVector class implementation
@dataclass
class NetworkCoordinates:
    coordinates: List[FloatType]
    height: FloatType

    def __sub__(self, other):
        diff_coords = [a - b for a, b in zip(self.coordinates, other.coordinates)]
        return NetworkCoordinates(diff_coords, self.height - other.height)

    def __add__(self, other):
        sum_coords = [a + b for a, b in zip(self.coordinates, other.coordinates)]
        return NetworkCoordinates(sum_coords, self.height + other.height)

    def __mul__(self, scalar):
        scaled_coords = [a * scalar for a in self.coordinates]
        return NetworkCoordinates(scaled_coords, self.height * scalar)

    def len(self):
        return math.sqrt(sum([a ** 2 for a in self.coordinates]) + self.height ** 2)

    def normalized(self):
        norm_len = self.len()
        if norm_len == 0:
            raise ValueError("Cannot normalize zero-length vector")
        return self * (1.0 / norm_len)

    @staticmethod
    def random():
        phi = np.random.uniform(0, np.pi)  # Angle from z-axis
        theta = np.random.uniform(0, 2 * np.pi)  # Rotation around z-axis

        # Spherical to Cartesian conversion
        x = 0.001 * np.sin(phi) * np.cos(theta)
        y = 0.001 * np.sin(phi) * np.sin(theta)
        z = 0.001 * np.cos(phi)

        coordinates = [x, y, z]

        height = random.random()
        return NetworkCoordinates(coordinates, height)

    def is_invalid(self):
        return any(math.isnan(c) or math.isinf(c) for c in self.coordinates + [self.height])


class NetworkNode:

    def __init__(self):
        self.coordinates = NetworkCoordinates.random()
        self.error = MIN_ERROR
        self.history = []
        self.peers = []
        self.max_peers = 30
        self.strategy = NoAttack()

    def set_strategy(self, strategy):
        self.strategy = strategy

    def estimated_rtt(self, rhs: 'NetworkNode') -> timedelta:
        rhs_after_strategy = rhs.strategy.attack(rhs)
        rtt_estimate = (self.coordinates - rhs_after_strategy.coordinates).len() / 1000.0
        return timedelta(seconds=rtt_estimate)

    def _update(self, rhs: 'NetworkNode', rtt: timedelta):
        rtt_ms = rtt.total_seconds() * 1000
        rtt_estimated_ms = self.estimated_rtt(rhs).total_seconds() * 1000.0
        if rtt_ms < 0.0:
            raise ValueError("RTT cannot be negative")

        w = self.error / (self.error + rhs.error)
        error = rtt_ms - rtt_estimated_ms
        epsilon = abs(error) / rtt_ms
        self.error = max((epsilon * C_ERROR) * w + self.error * C_ERROR * (-w + 1.0), MIN_ERROR)
        force = C_DELTA * w * error
        if error / rtt_ms <= STABLE:
            force = min(force, F_C)
        median_absolute_deviation = np.median(np.absolute(self.history - np.median(self.history)))
        force_median = np.median(self.history)
        if force > force_median + 8 * median_absolute_deviation:
            return self
        if len(self.history) == 10:
            self.history.pop(0)
        self.history.append(force)
        self.coordinates = self.coordinates + (self.coordinates - rhs.coordinates).normalized() * force

        if self.coordinates.is_invalid():
            return NetworkNode()
        return self

    def update(self, rhs: 'NetworkNode', rtt: timedelta) -> 'NetworkNode':
        if self.check_first_invariant():
            print("Validated one of invariants")
            return self
        print("Normal update")
        return self._update(rhs, rtt)

    def update_without_invariants(self, rhs: 'NetworkNode', rtt: timedelta):
        return self._update(rhs, rtt)

    def check_first_invariant(self):
        random_peers = random.sample(self.peers, min(self.max_peers, len(self.peers)))
        centroid = self.compute_centroid([c.coordinates.coordinates for c in random_peers])
        if np.linalg.norm(centroid - np.array((0, 0, 0))) > 50:
            return True
        return False

    def compute_centroid(self, coordinates):
        n = len(coordinates)
        if n == 0:
            return np.zeros(3)  # assuming 2D coordinates for simplicity
        co = np.mean(coordinates, axis=0)
        return co
