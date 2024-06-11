import random
import heapq
import itertools
from copy import copy, deepcopy
from datetime import timedelta

import numpy as np
import scipy.special
import time
import math

from attacks import AttackStrategy, NoAttack


# Compute the times replicas form weighted quorums
# --> Assuming that all replicas are correct
def formQV(n, Mwrite, tProposed, weights, quorumWeight):  # Alg. 1 in AWARE

    received = []
    for i in range(n):
        received.append([])
        for j in range(n):
            heapq.heappush(received[i], (
                tProposed[j] + Mwrite[j][i].total_seconds() * 1000.0, weights[j]))  # use received[] as a priority queue

    nextTimes = [0] * n

    quorumTimes = []
    for i in range(n):
        weight = 0
        while weight < quorumWeight:
            (rcvTime, rcvWeight) = heapq.heappop(received[i])
            weight += rcvWeight
            nextTimes[i] = rcvTime

    return nextTimes


# TODO:
# Cases with asynchrony: 
# f+1 Echo should imply an Echo (Propose) (right now only provoked by the leader's message)
# f+1 Ready (Accept) should imply a Ready (Accept)

# DONE:
# Leader shouldn't send an Echo (Propose) message: OK because latency from leader to itself is 0
# offset (replaced by another mechanism) is useful when the leader is better connected than other replicas:
# the next propose will arrive before they have sent their own Write

def predictLatency(n, f, delta, weights, leaderId, Mpropose, Mwrite, rounds):  # Alg. 2 in AWARE

    quorumWeight = 2 * (f + delta) + 1  # Weight of a quorum

    tProposed = [float(0)] * n  # time at which the latest proposal is received by replicas
    offset = [float(0)] * n  # time at which replicas deliver the latest proposal

    tAccepted = [float(0)] * n

    consensusLatencies = [float(0)] * rounds  # latency of each consensus round

    for r in range(rounds):

        # compute times at which each replica receives the leader's i-th propose
        for i in range(n):
            if i != leaderId:  # not in Alg. 2
                tProposed[i] = max(tProposed[leaderId] + Mpropose[leaderId][i].total_seconds() * 1000.0,
                                   tAccepted[i])  # added tProposed[leaderId]
        # print('sending WRITE times', tProposed)

        tWritten = formQV(n, Mwrite, tProposed, weights, quorumWeight)
        # print('sending ACCEPT times', tWritten)

        tAccepted = formQV(n, Mwrite, tWritten, weights, quorumWeight)
        # print('ACCEPTED times', tAccepted, '\n')

        consensusLatencies[r] = tAccepted[leaderId] - tProposed[leaderId]
        tProposed[leaderId] = tAccepted[leaderId]  # not in Alg. 2

    print(consensusLatencies)
    return sum(consensusLatencies) / len(consensusLatencies)


def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


def exhaustive_search(n, f, delta, Mpropose, Mwrite, r):
    bestConsensusLat = 1000000
    bestLeader = -1
    bestWeights = -1

    numConfigs = scipy.special.comb(n, 2 * f, exact=True) * 2 * f
    print('Num possible configs =', numConfigs)

    weights = []
    for i in range(2 * f):
        weights.append(1 + delta / f)
    for i in range(n - 2 * f):
        weights.append(1)

    start = time.time()
    curConfig = 0
    curLeader = 0
    for vMaxPos in itertools.combinations(range(n), 2 * f):
        curWeights = [1] * n

        for i in vMaxPos:
            curWeights[i] = 1 + delta / f

        for curLeader in vMaxPos:
            tmp = predictLatency(n, f, delta, curWeights, curLeader, Mpropose, Mwrite, r)
            if curConfig == 0 or tmp < bestConsensusLat:
                bestConsensusLat = tmp
                bestLeader = curLeader
                bestWeights = curWeights

            if curConfig % 1000 == 0:
                print(curConfig, '/', numConfigs)
            curConfig += 1

    end = time.time()
    print('Computation time = ', end - start)

    print('best consensus latency:', bestConsensusLat)
    print('best leader:', bestLeader)
    print('best weights:', bestWeights)


def convert_bestweights_to_rmax_rmin(best_weights, vmax):
    replicas = [i for i in range(len(best_weights))]
    r_max = []
    r_min = []
    for rep_id in replicas:
        if best_weights[rep_id] == vmax:
            r_max.append(replicas[rep_id])
        else:
            r_min.append(replicas[rep_id])
    return r_max, r_min


def simulated_annealing(n, f, delta, Mpropose, Mwrite, r, suffix=''):
    start = time.time()

    random.seed(500)
    vmax = 1 + delta / f  # 2f replicas
    vmin = 1  # n-2f replicas
    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2
    r_max = []
    r_min = []

    curWeights = [1] * n
    for i in range(2 * f):
        curWeights[i] = 1 + delta / f
    curLeader = 0

    curLat = predictLatency(n, f, delta, curWeights, curLeader, Mpropose, Mwrite, r)

    bestLat = curLat
    bestLeader = -1
    bestWeights = []
    jumps = 0

    while step < step_max and temp > t_min:
        replicaFrom = -1
        replicaTo = -1
        newLeader = curLeader
        while True:
            replicaFrom = random.randint(0, n - 1)
            if curWeights[replicaFrom] == 1 + delta / f:
                break
        while True:
            replicaTo = random.randint(0, n - 1)
            if replicaTo != replicaFrom:
                break

        if replicaFrom == curLeader:
            newLeader = replicaTo

        newWeights = curWeights.copy()
        newWeights[replicaTo] = curWeights[replicaFrom]
        newWeights[replicaFrom] = curWeights[replicaTo]
        ##    print(newWeights)

        newLat = predictLatency(n, f, delta, newWeights, newLeader, Mpropose, Mwrite, r)

        if newLat < curLat:
            curLeader = newLeader
            curWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLat - curLat) / temp):
                jumps = jumps + 1
                curLeader = newLeader
                curWeights = newWeights

        if newLat < bestLat:
            bestLat = newLat
            bestLeader = newLeader
            bestWeights = newWeights

        temp = temp * (1 - theta)
        step += 1

    end = time.time()
    r_max, r_min = convert_bestweights_to_rmax_rmin(bestWeights, vmax)

    print('--------------------------------')
    print('--------' + suffix + ' Simulated annealing')
    print('--------------------------------')
    print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    print('Final solution latency:', bestLat)
    print('Best Configuration:  R_max: {}  | R_min: {}  with leader {}'.format(r_max, r_min, bestLeader))
    print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))
    return [r_max, r_min, bestLeader]


def run_simulation(n, f, delta, rounds: int, m_propose, m_write, malicious=True, attack: AttackStrategy = NoAttack()):
    times = []

    vmax = 1 + delta / f  # 2f replicas
    vmin = 1  # n-2f replicas
    malicious_nodes = random.sample(range(n), f)
    Mpropose = deepcopy(m_propose)
    Mwrite = deepcopy(m_write)

    weights = []
    for i in range(2 * f):
        weights.append(vmax)
    for i in range(n - 2 * f):
        weights.append(vmin)
    for i in range(rounds):

        r_max, r_min, bestLeader = simulated_annealing(n, f, delta, Mpropose, Mwrite, rounds)
        for replica_with_max in r_max:
            weights[replica_with_max] = vmax
        for replica_with_min in r_min:
            weights[replica_with_min] = vmin
        if malicious:
            print(attack)
            Mpropose = attack.attack(Mpropose, malicious_nodes)
            Mwrite = attack.attack(Mwrite, malicious_nodes)
        times.append(predictLatency(n, f, delta, weights, bestLeader, m_propose, m_write, 1))
    return times


def run_simulation_vivaldi(n, f, delta, rounds: int, coordinates, Mpropose, malicious=True):
    times = []

    vmax = 1 + delta / f  # 2f replicas
    vmin = 1  # n-2f replicas
    malicious_nodes = random.sample(range(n), f)

    weights = []
    for i in range(2 * f):
        weights.append(vmax)
    for i in range(n - 2 * f):
        weights.append(vmin)
    for i in range(rounds):
        m_propose = np.array([[coordinates[i].estimated_rtt(coordinates[j]) for j in range(n)] for i in range(n)])
        Mwrite = copy(m_propose)
        r_max, r_min, bestLeader = simulated_annealing(n, f, delta, m_propose, Mwrite, rounds)
        for replica_with_max in r_max:
            weights[replica_with_max] = vmax
        for replica_with_min in r_min:
            weights[replica_with_min] = vmin
        if malicious:
            for node in malicious_nodes:
                for second_node in range(n):
                    upper_limit = 1e9  # This can be any large number
                    bounded_random_float = random.uniform(0, upper_limit)
                    bounded_random_float = 1e9
                    coordinates[node] = coordinates[node].update(coordinates[second_node],
                                                                 timedelta(milliseconds=bounded_random_float))
        times.append(predictLatency(n, f, delta, weights, bestLeader, copy(Mpropose), copy(Mpropose), 1))
    return times



