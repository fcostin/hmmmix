import numpy

from . import gentrellis
from . import libtrellis

_STATE_TO_INDEX = {s:i for i, s in enumerate(gentrellis.STATES)}
_STATES = numpy.asarray([_STATE_TO_INDEX[s] for s in gentrellis.STATES], dtype=int)

def _encode_edge(edge):
    return gentrellis.Edge(succ=_STATE_TO_INDEX[edge.succ], weight=edge.weight, delta_e=edge.delta_e)

_OUTGOING_EDGES = {_STATE_TO_INDEX[s]:[_encode_edge(edge) for edge in gentrellis.OUTGOING_EDGES_BY_STATE[s]] for s in gentrellis.STATES}

_LOGPROB_PRIOR = numpy.asarray([gentrellis.LOGPROB_PRIOR_BY_STATE[s] for s in gentrellis.STATES], dtype=numpy.float64)


def _pack_edges(states, outgoing_edges):
    """
    pack edges into C friendly compressed-sparse format that can be iterated
    over without hashing or attribute access

    :param states: list of state indices
    :param outgoing_edges: lists of edge namedtuples keyed by state index
    :return: (start_indices, end_indices, succs, weights, delta_es)
    """

    n_states = len(states)
    n_edges = sum(len(outgoing_edges[s]) for s in states)

    start_indices = numpy.zeros(shape=(n_states, ), dtype=int)
    end_indices = numpy.zeros(shape=(n_states, ), dtype=int)

    succs = numpy.zeros(shape=(n_edges, ), dtype=int) # successor state indices
    weights = numpy.zeros(shape=(n_edges, ), dtype=numpy.float64)
    delta_es = numpy.zeros(shape=(n_edges, ), dtype=int)

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
            delta_es[i] = edge.delta_e
            i += 1
        end_indices[s] = i

    return (start_indices, end_indices, succs, weights, delta_es)


_PACKED_EDGES = _pack_edges(_STATES, _OUTGOING_EDGES)


def search_best_path(times, prizes):
    # TODO also need to also accept constraints banning solutions.

    # times must be array of time indices 0...T-1
    # prizes must be shape (T, ) array of prizes (unit: logprob units per event).

    objective_star, logprob_star, state_trajectory, obs_trajectory = _search_best_path(
        times,
        _STATES,
        _LOGPROB_PRIOR,
        _PACKED_EDGES,
        prizes,
    )
    # translate from state indices back into fancy states
    fancy_state_trajectory = [gentrellis.prettystate(gentrellis.STATES[s]) for s in state_trajectory]

    return (objective_star, logprob_star, fancy_state_trajectory, obs_trajectory)


def _search_best_path(times, states, logprob_prior, packed_edges, prizes):
    # times must be array of time indices 0...T-1
    # states must be array of state indices 0...S-1
    # prior must be shape (S, ) array of logprob giving prior of states

    # bottom up dynamic programming solve for value maximising paths.
    if True:
        kernel = libtrellis._kernel
    else:
        kernel = _kernel

    objective_star, logprob_star, state_trajectory, obs_trajectory = kernel(
        times,
        states,
        packed_edges,
        prizes,
        logprob_prior,
    )

    return (objective_star, logprob_star, state_trajectory, obs_trajectory)


def _kernel(times, states, packed_edges, prizes, logprob_prior):
    """
    :param times: array of integer time indices [0, ..., T-1] shape (T,)
    :param states: array of state indices [s_1, ..., s_K] shape (S,)
    :param packed_edges: see _PACKED_EDGES
    :param prizes: array of float prizes. shape (T, )
    :param logprob_prior: array of float value. shape (S, )

    returns (objective_star, logprob_star, state_trajectory, obs_trajectory)
    """

    # v: shape T * S dense array of float64 objective values, init to -inf
    # p: shape T * S dense array of float64 logprob values, init to -inf
    # parent_s: shape T * S dense array of state values, init to arbitrary state 0
    # parent_obs: shape T * S dense array of integer obs counts, init to 0

    # Phase 1. Setup problem.

    T = len(times)
    S = len(states)

    v = numpy.empty(shape=(T, S), dtype=numpy.float64)
    v[:, :] = -numpy.inf

    logprob = numpy.empty(shape=(T, S), dtype=numpy.float64)
    logprob[:, :] = -numpy.inf

    parent_s = numpy.zeros(shape=(T, S), dtype=int)
    parent_obs = numpy.zeros(shape=(T, S), dtype=int)

    # initialise value and logprob at t=0 using prior
    for s in states:
        v[0, s] = logprob_prior[s]
        logprob[0, s] = logprob_prior[s]

    (start_indices, end_indices, succs, weights, delta_es) = packed_edges

    # Phase 2. Bottom-up DP.

    # FIXME will not correctly handle prizes on last timestep.
    # Handle by padding with extra terminator time?
    for t in times[:-1]:
        for s in states:
            for j in range(start_indices[s], end_indices[s]):
                s_prime = succs[j]
                v_prime = v[t, s] + weights[j] + delta_es[j] * prizes[t]
                logprob_prime = logprob[t, s] + weights[j]
                if v_prime > v[t+1, s_prime]:
                    v[t+1, s_prime] = v_prime
                    logprob[t+1, s_prime] = logprob_prime
                    parent_s[t+1, s_prime] = s
                    parent_obs[t+1, s_prime] = delta_es[j]

    # Phase 3. Recover a value-maximising path (may be nonunique)
    state_trajectory = []
    obs_trajectory = []

    t = T - 1
    s = numpy.argmax(v[t, :])

    objective_star = v[t, s]
    logprob_star = logprob[t, s]

    while True:
        e = parent_obs[t, s]
        state_trajectory.append(s)
        obs_trajectory.append(e)
        if t == 0:
            break
        t_prime = t - 1
        s_prime = parent_s[t, s]
        t = t_prime
        s = s_prime

    state_trajectory = list(reversed(state_trajectory))
    obs_trajectory = list(reversed(obs_trajectory))

    obs_trajectory = obs_trajectory[1:] + [0]  # FIXME aiee

    return (objective_star, logprob_star, state_trajectory, obs_trajectory)