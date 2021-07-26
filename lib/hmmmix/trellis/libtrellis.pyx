ctypedef long index_t
ctypedef double dtype_t

def _kernel(index_t[:] times, \
            index_t[:] states, \
            outgoing_edges, \
            dtype_t[:] prizes, \
            dtype_t[:, :] v, \
            dtype_t[:, :] logprob, \
            index_t[:, :] parent_s, \
            index_t[:, :] parent_obs):
    """
    :param times: array of integer time indices [0, ..., T-1] shape (T,)
    :param states: array of state indices [s_1, ..., s_K] shape (S,)
    :param outgoing_edges: dict of state index to list of edge tuples. rework.
    :param prizes: array of float prizes. shape (T, )
    :param v: array of float value. shape (T, S)
    :param logprob: array of float logprob. shape (T, S)
    :param parent_s: array of state indices. shape (T, S)
    :param parent_obs: array of integer event counts. shape (T, S)

    returns None

    v, logprob, parent_s and parent_obs are mutated in place.
    """

    cdef index_t t, s, s_prime
    cdef dtype_t v_prime

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