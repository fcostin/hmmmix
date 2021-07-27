import cython
import numpy

ctypedef long index_t
ctypedef double dtype_t

def _kernel(index_t[:] times, \
            index_t[:] states, \
            packed_edges, \
            packed_obs, \
            dtype_t[:] prizes, \
            dtype_t[:] logprob_prior):
    """
    :param times: array of integer time indices [0, ..., T-1] shape (T,)
    :param states: array of state indices [s_1, ..., s_K] shape (S,)
    :param packed_edges: see pack_edges
    :param packed_obs: see pack_observations
    :param prizes: array of float prizes. shape (T, )
    :param logprob_prior: array of float value. shape (S, )

    returns (objective_star, logprob_star, state_trajectory, obs_trajectory)
    """

    cdef index_t t, t_prime, s, s_prime, T, S, sj, j, k
    cdef dtype_t v_0, v_prime, logprob_prime

    # v: shape (T+1) * S dense array of float64 objective values, init to -inf
    # logprog: shape (T+1) * S dense array of float64 logprob values, init to -inf
    # parent_s: shape (T+1) * S dense array of state values, init to arbitrary state 0
    # obs: shape (T+1) * S dense array of integer obs counts, init to 0

    # Phase 1. Setup problem.

    # By convention caller passes in the timesteps when observations occur.
    # But we also need to index over the time before the first observation -
    # this is the time that the prior distribution over states specifies.
    T = len(times)
    S = len(states)

    # Sanity check indices to give more confidence before we turn off
    # bounds checking.

    for si in range(S):
        assert 0 <= states[si] and states[si] < S


    cdef dtype_t[:, :] v = numpy.empty(shape=(T+1, S), dtype=numpy.float64)
    v[:, :] = -numpy.inf

    cdef dtype_t[:, :] logprob = numpy.empty(shape=(T+1, S), dtype=numpy.float64)
    logprob[:, :] = -numpy.inf

    cdef index_t[:, :] parent_s = numpy.zeros(shape=(T+1, S), dtype=int)
    cdef index_t[:, :] obs = numpy.zeros(shape=(T+1, S), dtype=int)

    # initialise value and logprob at t=0 using prior
    for s in states:
        v[0, s] = logprob_prior[s]
        logprob[0, s] = logprob_prior[s]

    # FIXME it'd arguably make better use of cache to pack the three edge
    # attributes into a single array of structs
    (_start_edge_indices, _end_edge_indices, _succs, _edge_weights) = packed_edges
    cdef index_t[:] start_edge_indices = _start_edge_indices
    cdef index_t[:] end_edge_indices = _end_edge_indices
    cdef index_t[:] succs = _succs
    cdef dtype_t[:] edge_weights = _edge_weights

    (_start_obs_indices, _end_obs_indices, _delta_obs, _obs_weights) = packed_obs
    cdef index_t[:] start_obs_indices = _start_obs_indices
    cdef index_t[:] end_obs_indices = _end_obs_indices
    cdef index_t[:] delta_obs = _delta_obs
    cdef dtype_t[:] obs_weights = _obs_weights

    # Phase 2. Bottom-up DP.

    with nogil:
        with cython.wraparound(False):
            with cython.boundscheck(False):
                for t in range(T):
                    for sj in range(S):
                        s = states[sj]
                        for j in range(start_edge_indices[s], end_edge_indices[s]):
                            t_prime = t + 1
                            s_prime = succs[j]
                            logprob_prime = logprob[t, s] + edge_weights[j]
                            v_0 = v[t, s] + edge_weights[j]
                            for k in range(start_obs_indices[s_prime], end_obs_indices[s_prime]):
                                # horror: by prizes[t_prime-1] we mean the prize at t_prime.
                                # prizes uses a different indexing convention to our arrays.
                                v_prime = v_0 + obs_weights[k] + delta_obs[k] * prizes[t_prime-1]
                                if v_prime > v[t+1, s_prime]:
                                    v[t+1, s_prime] = v_prime
                                    logprob[t+1, s_prime] = logprob_prime
                                    parent_s[t+1, s_prime] = s
                                    obs[t+1, s_prime] = delta_obs[k]

    # Phase 3. Recover a value-maximising path (may be nonunique)
    state_trajectory = []
    obs_trajectory = []

    t = T
    s = numpy.argmax(v[t, :])

    objective_star = v[t, s]
    logprob_star = logprob[t, s]

    # Our t = 0 corresponds to time of prior, before first observation,
    # so we do not recover any corresponding state or observation for it.
    while t > 0:
        e = obs[t, s]
        state_trajectory.append(s)
        obs_trajectory.append(e)
        t_prime = t - 1
        s_prime = parent_s[t, s]
        t = t_prime
        s = s_prime

    state_trajectory = list(reversed(state_trajectory))
    obs_trajectory = list(reversed(obs_trajectory))

    obs_trajectory = obs_trajectory

    return (objective_star, logprob_star, state_trajectory, obs_trajectory)