from abc import abstractmethod, ABC
from datetime import timedelta
from random import random
from typing import List


class AttackStrategy(ABC):
    @abstractmethod
    def attack(self, network: List[List[timedelta]], malicious_nodes):
        """Perform attack on the network's latency matrix within the malicious nodes."""
        pass


class InflationAttack(AttackStrategy):
    def attack(self, latency_matrix: List[List[timedelta]], malicious_nodes):
        """Increase latencies dramatically among malicious nodes to simulate inflation."""
        for i in malicious_nodes:
            for j in malicious_nodes:
                if i != j:
                    current_latency = (latency_matrix[i][j]).total_seconds() * 1000
                    latency_matrix[i][j] = timedelta(milliseconds=current_latency * 10)
        return latency_matrix


class NoAttack(AttackStrategy):
    def attack(self, latency_matrix, malicious_nodes):
        return latency_matrix


class DeflationAttack(AttackStrategy):
    def attack(self, latency_matrix: List[List[timedelta]], malicious_nodes: List[int]):
        """Decrease latencies among malicious nodes to near zero to simulate deflation."""
        for i in malicious_nodes:
            for j in malicious_nodes:
                if i != j:
                    latency_matrix[i][j] = timedelta(milliseconds=1)
        return latency_matrix


class OscillationAttack(AttackStrategy):
    def attack(self, latency_matrix, malicious_nodes):
        """Randomly adjust latencies among malicious nodes to create oscillations."""
        for i in malicious_nodes:
            for j in malicious_nodes:
                if i != j:
                    latency_matrix[i][j] = timedelta(miliseconds=random.randint(1, 1000))
        return latency_matrix


class FrogBoilingAttack(AttackStrategy):
    def attack(self, latency_matrix, malicious_nodes):
        """Slowly increase the latency incrementally among malicious nodes."""
        for i in malicious_nodes:
            for j in malicious_nodes:
                if i != j:
                    current_latency = (latency_matrix[i][j]).total_seconds() * 1000
                    latency_matrix[i][j] = timedelta(milliseconds=current_latency + 5)
        return latency_matrix


class NetworkPartitionAttack(AttackStrategy):
    def attack(self, latency_matrix, malicious_nodes):
        """Alter latencies dramatically between groups within malicious nodes."""
        group1 = malicious_nodes[:len(malicious_nodes) // 2]
        group2 = malicious_nodes[len(malicious_nodes) // 2:]
        for i in group1:
            for j in group2:
                current_latency_i_j = (latency_matrix[i][j]).total_seconds() * 1000
                current_latency_j_i = (latency_matrix[j][i]).total_seconds() * 1000
                latency_matrix[i][j] = timedelta(milliseconds=current_latency_i_j + 5)
                latency_matrix[j][i] = timedelta(milliseconds=current_latency_j_i - 5)
        return latency_matrix