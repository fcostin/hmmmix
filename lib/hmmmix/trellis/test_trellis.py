from . import Edge, HMM, pack_edges, search_best_path

import numpy
import pytest


@pytest.fixture(scope="module", params=["slow", "fast"])
def kernel(request):
    return request.param


def make_trivial_fair_coin_model():
    n_states = 2
    # state 0 : "going to flip heads"
    # state 1 : "going to flip tails"
    states = numpy.arange(n_states, dtype=numpy.int64)

    log_pr_heads = -numpy.log(2.0)
    log_pr_tails = -numpy.log(2.0)

    outgoing_edges = {
        0: [
            Edge(
                succ=0,
                weight=log_pr_heads,
                delta_e=1, # emit heads
            ),
            Edge(
                succ=1,
                weight=log_pr_tails,
                delta_e=1, # emit tails
            )
        ],
        1: [
            Edge(
                succ=0,
                weight=-log_pr_heads,
                delta_e=0,  # emit tails
            ),
            Edge(
                succ=1,
                weight=-log_pr_tails,
                delta_e=0,  # emit tails
            )
        ],
    }

    packed_edges = pack_edges(states, outgoing_edges)

    prior = -numpy.log(2.0) * numpy.ones((n_states, ), dtype=numpy.float64)

    return HMM(
        state_by_statekey={}, # refactor. remove?
        states=states,
        packed_edges=packed_edges,
        prior=prior,
    )


@pytest.mark.parametrize("observations", [
#    ([]),  # BROKEN
    ([0]), # OK
#    ([1]),
    ([0, 0]),  # OK
#    ([0, 1]),
    ([1, 0]),  # OK
#    ([1, 1]),
    ([0, 0, 0]),  # OK
#    ([0, 0, 1]),
    ([0, 1, 0]),  # OK
#    ([0, 1, 1]),
    ([1, 0, 0]),  # OK
#    ([1, 0, 1]),
    ([1, 1, 0]),  # OK
#    ([1, 1, 1]),
])
def test_trivial_fair_coin_model_can_recover_a_trajectory(kernel, observations):

    observations = numpy.asarray(observations, dtype=numpy.int64)
    T = len(observations)
    times = numpy.arange(T, dtype=numpy.int64)

    # explain observations, receive prizes.
    prizes = numpy.where(observations, 100.0, 0.0)

    hmm = make_trivial_fair_coin_model()

    result = search_best_path(hmm, times, prizes, kernel)

    objective_star, logprob_star, state_trajectory, obs_trajectory = result

    expected_state_trajectory = observations

    assert numpy.allclose(observations, obs_trajectory)
    assert numpy.allclose(expected_state_trajectory, obs_trajectory)