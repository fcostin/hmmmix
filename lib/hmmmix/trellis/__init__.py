"""
trellis implements a modified Viterbi algorithm for recovering point estimates
of (state trajectory, observation trajectory) pairs of hidden Markov models.
"""


from . import slowtrellis
from . import libtrellis

import typing
import numpy
import numpy.typing


class WeightedEdge(typing.NamedTuple):
    succ: numpy.int64 # successor state index
    weight: numpy.float64 # edge weight (log prob)


class PackedEdges(typing.NamedTuple):
    start_indices: numpy.typing.NDArray[numpy.int64]
    end_indices: numpy.typing.NDArray[numpy.int64]
    succs: numpy.typing.NDArray[numpy.int64] # successor state indices
    weights: numpy.typing.NDArray[numpy.float64] # edge weights


class WeightedObservation(typing.NamedTuple):
    delta_e: numpy.int64 # emitted observation (count)
    weight: numpy.float64 # edge weight (log prob)


# PackedObservations encodes discrete distribution of possible observation
# counts conditional on the current hidden state s. Index into start_indices,
# end_indices using current state index. That gives a range of 0 or more
# indices addressing elements of delta_es & weights. Former is an event count,
# latter is the associated conditional emission log-probability given state s.
class PackedObservations(typing.NamedTuple):
    start_indices: numpy.typing.NDArray[numpy.int64]
    end_indices: numpy.typing.NDArray[numpy.int64]
    delta_es: numpy.typing.NDArray[numpy.int64] # emitted observation counts
    weights: numpy.typing.NDArray[numpy.float64] # edge weights


class HMM(typing.NamedTuple):
    state_by_statekey: typing.Dict[typing.Hashable, numpy.int64] # state index by state key
    states: numpy.typing.NDArray[numpy.int64] # state indices
    packed_edges: PackedEdges
    packed_observations: PackedObservations # discrete observation model aka sensor model
    prior: numpy.typing.NDArray[numpy.float64] # prior for each state, indexed by state indices.


def search_best_path(hmm: HMM,
                     times: numpy.typing.NDArray[numpy.int64],
                     prizes: numpy.typing.NDArray[numpy.float64], kernel=None):

    if kernel is None:
        kernel = libtrellis._kernel
    elif kernel == 'fast':
        kernel = libtrellis._kernel
    elif kernel == 'slow':
        kernel = slowtrellis._kernel
    if kernel is None:
        raise ValueError("kernel unspecified")

    # TODO also need to also accept constraints banning solutions.

    # bottom up dynamic programming solve for value maximising paths
    objective_star, logprob_star, state_trajectory, obs_trajectory = kernel(
        times, # TODO remove, unnecessary flexibility.
        hmm.states,
        hmm.packed_edges,
        hmm.packed_observations,
        prizes,
        hmm.prior,
    )

    return (objective_star, logprob_star, state_trajectory, obs_trajectory)


def pack_edges(states: numpy.typing.NDArray[numpy.int64], outgoing_edges: typing.Dict[numpy.int64, typing.Sequence[WeightedEdge]]):
    """
    pack edges into C friendly compressed-sparse format that can be iterated
    over without hashing or attribute access

    :param states: list of state indices
    :param outgoing_edges: lists of WeightedEdge keyed by state index
    :return: PackedEdges
    """

    n_states = len(states)
    n = sum(len(outgoing_edges[s]) for s in states)

    start_indices = numpy.zeros(shape=(n_states, ), dtype=int)
    end_indices = numpy.zeros(shape=(n_states, ), dtype=int)

    succs = numpy.zeros(shape=(n, ), dtype=int) # successor state indices
    weights = numpy.zeros(shape=(n, ), dtype=numpy.float64)

    i = 0
    for s in states:
        s = int(s)
        assert 0 <= s and s < n_states
        start_indices[s] = i
        for edge in outgoing_edges[s]:
            s_prime = int(edge.succ)
            assert 0 <= s_prime and s_prime < n_states
            succs[i] = edge.succ
            weights[i] = edge.weight
            i += 1
        end_indices[s] = i

    return PackedEdges(
        start_indices=start_indices,
        end_indices=end_indices,
        succs=succs,
        weights=weights,
    )


def pack_observations(states: numpy.typing.NDArray[numpy.int64],
                      weighted_obs_by_state: typing.Dict[numpy.int64, typing.Sequence[WeightedObservation]]):
    """
    pack weighted observations into C friendly compressed-sparse format that can
    be iterated over without hashing or attribute access

    :param states: list of state indices
    :param outgoing_observations: lists of WeightedObservation keyed by state index
    :return: (PackedObservations
    """

    n_states = len(states)
    n = sum(len(weighted_obs_by_state[s]) for s in states)

    start_indices = numpy.zeros(shape=(n_states, ), dtype=int)
    end_indices = numpy.zeros(shape=(n_states, ), dtype=int)

    delta_es = numpy.zeros(shape=(n, ), dtype=int)
    weights = numpy.zeros(shape=(n, ), dtype=numpy.float64)

    i = 0
    for s in states:
        s = int(s)
        assert 0 <= s and s < n_states
        start_indices[s] = i
        for edge in weighted_obs_by_state[s]:
            weights[i] = edge.weight
            delta_es[i] = edge.delta_e
            i += 1
        end_indices[s] = i

    return PackedObservations(
        start_indices=start_indices,
        end_indices=end_indices,
        delta_es=delta_es,
        weights=weights,
    )