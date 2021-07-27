from . import (
    HMM,
    WeightedEdge, pack_edges,
    WeightedObservation, pack_observations,
    search_best_path,
)

import numpy
import pytest


@pytest.fixture(scope="module", params=["slow", "fast"])
def kernel(request):
    return request.param


def make_trivial_biased_coin_model(pr_heads):
    n_states = 1
    # state 0 : the state of being a biased_coin
    states = numpy.arange(n_states, dtype=numpy.int64)

    log_pr_unity = numpy.log(1.0)
    log_pr_heads = numpy.log(pr_heads)
    log_pr_tails = numpy.log(1.0 - pr_heads)

    outgoing_edges_by_state = {
        0: [
            WeightedEdge(
                succ=0,
                weight=log_pr_unity, # continue being a coin
            ),
        ],
    }


    packed_edges = pack_edges(states, outgoing_edges_by_state)

    weighted_obs_by_state = {
        0: [
            WeightedObservation(
                weight=log_pr_heads,
                delta_e=1,
            ),
            WeightedObservation(
                weight=log_pr_tails,
                delta_e=0,
            ),
        ]
    }

    packed_obs = pack_observations(states, weighted_obs_by_state)

    print(repr(packed_obs))

    # uniform prior over states. long-winded way of saying our coin starts in
    # the only possible state with probability 1.
    prior = -numpy.log(1.0 * n_states) * numpy.ones((n_states, ), dtype=numpy.float64)

    return HMM(
        state_by_statekey={}, # refactor. remove?
        states=states,
        packed_edges=packed_edges,
        packed_observations=packed_obs,
        prior=prior,
    )


@pytest.mark.parametrize("observations", [
    ([]),
    ([0]),
    ([1]),
    ([0, 0]),
    ([0, 1]),
    ([1, 0]),
    ([1, 1]),
    ([0, 0, 0]),
    ([0, 0, 1]),
    ([0, 1, 0]),
    ([0, 1, 1]),
    ([1, 0, 0]),
    ([1, 0, 1]),
    ([1, 1, 0]),
    ([1, 1, 1]),
])
def test_trivial_fair_coin_model_can_recover_a_trajectory(kernel, observations):

    observations = numpy.asarray(observations, dtype=numpy.int64)
    T = len(observations)
    times = numpy.arange(T, dtype=numpy.int64)

    # Explain observations, receive prizes.
    # note that we need to give negative prizes (penalties) if anyone
    # wants to emit heads where we saw tails. If we set prize=0 to those
    # entries then that would expresss our indifference to the emitted event.
    prizes = numpy.where(observations > 0, 100.0, -100.0)

    hmm = make_trivial_biased_coin_model(pr_heads=0.5)

    result = search_best_path(hmm, times, prizes, kernel)

    objective_star, logprob_star, state_trajectory, obs_trajectory = result

    # there is a unique state, hence estimate of hidden state trajectory is
    # not very exciting
    expected_state_trajectory = numpy.zeros(shape=(T, ), dtype=numpy.int64)

    assert numpy.allclose(observations, obs_trajectory)
    assert numpy.allclose(expected_state_trajectory, state_trajectory)