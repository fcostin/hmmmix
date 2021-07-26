import numpy

from . import gentrellis
from . import libtrellis

_STATE_TO_INDEX = {s:i for i, s in enumerate(gentrellis.STATES)}
_STATES = numpy.asarray([_STATE_TO_INDEX[s] for s in gentrellis.STATES], dtype=int)

def _encode_edge(edge):
    return gentrellis.Edge(succ=_STATE_TO_INDEX[edge.succ], weight=edge.weight, delta_e=edge.delta_e)

_OUTGOING_EDGES = {_STATE_TO_INDEX[s]:[_encode_edge(edge) for edge in gentrellis.OUTGOING_EDGES_BY_STATE[s]] for s in gentrellis.STATES}

_LOGPROB_PRIOR = numpy.asarray([gentrellis.LOGPROB_PRIOR_BY_STATE[s] for s in gentrellis.STATES], dtype=numpy.float64)


def search_best_path(times, prizes):
    # TODO also need to also accept constraints banning solutions.

    # times must be array of time indices 0...T-1
    # prizes must be shape (T, ) array of prizes (unit: logprob units per event).

    objective_star, logprob_star, state_trajectory, obs_trajectory = _search_best_path(
        times,
        _STATES,
        _LOGPROB_PRIOR,
        _OUTGOING_EDGES,
        prizes,
    )
    # translate from state indices back into fancy states
    fancy_state_trajectory = [gentrellis.prettystate(gentrellis.STATES[s]) for s in state_trajectory]

    return (objective_star, logprob_star, fancy_state_trajectory, obs_trajectory)


def _search_best_path(times, states, logprob_prior, outgoing_edges, prizes):
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
        outgoing_edges,
        prizes,
        logprob_prior,
    )

    return (objective_star, logprob_star, state_trajectory, obs_trajectory)


def _kernel(times, states, outgoing_edges, prizes, logprob_prior):
    """
    :param times: array of integer time indices [0, ..., T-1] shape (T,)
    :param states: array of state indices [s_1, ..., s_K] shape (S,)
    :param outgoing_edges: dict of state index to list of edge tuples. rework.
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

    # Phase 2. Bottom-up DP.

    # FIXME will not correctly handle prizes on last timestep.
    # Handle by padding with extra terminator time?
    for t in times[:-1]:
        for s in states:
            for edge in outgoing_edges[s]:
                s_prime = edge.succ
                v_prime = v[t, s] + edge.weight + edge.delta_e * prizes[t]
                logprob_prime = logprob[t, s] + edge.weight
                if v_prime > v[t+1, s_prime]:
                    v[t+1, s_prime] = v_prime
                    logprob[t+1, s_prime] = logprob_prime
                    parent_s[t+1, s_prime] = s
                    parent_obs[t+1, s_prime] = edge.delta_e

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