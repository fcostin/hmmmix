import numpy
import numpy.typing


def _kernel(times, states, packed_edges, packed_obs, prizes, logprob_prior):
    """
    :param times: array of integer time indices [0, ..., T-1] shape (T,)
    :param states: array of state indices [s_1, ..., s_K] shape (S,)
    :param packed_edges: see pack_edges
    :param packed_obs: see pack_observations
    :param prizes: array of float prizes. shape (T, )
    :param logprob_prior: array of float value. shape (S, )

    returns (objective_star, logprob_star, state_trajectory, obs_trajectory)
    """

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

    v = numpy.empty(shape=(T+1, S), dtype=numpy.float64)
    v[:, :] = -numpy.inf

    logprob = numpy.empty(shape=(T+1, S), dtype=numpy.float64)
    logprob[:, :] = -numpy.inf

    parent_s = numpy.zeros(shape=(T+1, S), dtype=int)
    obs = numpy.zeros(shape=(T+1, S), dtype=int)

    # initialise value and logprob at t=0 using prior
    for s in states:
        v[0, s] = logprob_prior[s]
        logprob[0, s] = logprob_prior[s]

    (start_edge_indices, end_edge_indices, succ_states, edge_weights) = packed_edges

    (start_obs_indices, end_obs_indices, delta_obs, obs_weights) = packed_obs

    # Phase 2. Bottom-up DP.

    for t in range(T): # t is the previous time.
        for s in states:
            for j in range(start_edge_indices[s], end_edge_indices[s]):
                t_prime = t + 1
                s_prime = succ_states[j]
                logprob_prime = logprob[t, s] + edge_weights[j]
                v_0 = v[t, s] + edge_weights[j]
                for k in range(start_obs_indices[s_prime], end_obs_indices[s_prime]):
                    # horror: by prizes[t_prime-1] we mean the prize at t_prime.
                    # prizes uses a different indexing convention to our arrays.
                    v_prime = v_0 + obs_weights[k] + delta_obs[k] * prizes[t_prime-1]
                    if v_prime > v[t_prime, s_prime]:
                        v[t_prime, s_prime] = v_prime
                        logprob[t_prime, s_prime] = logprob_prime
                        parent_s[t_prime, s_prime] = s
                        obs[t_prime, s_prime] = delta_obs[k]

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