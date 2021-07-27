import collections
import numpy

from .model import monthly_markov as gentrellis
from . import trellis
from . import markov_event


def make_hmm():
    state_to_index = {s: i for i, s in enumerate(gentrellis.STATES)}
    states = numpy.asarray([state_to_index[s] for s in gentrellis.STATES], dtype=int)

    weighted_edges_by_state_index = collections.defaultdict(list)
    weighted_obs_by_state_index = collections.defaultdict(list)

    # Jank: convert from old monthly_markov aka gentrellis encoding.
    for s in gentrellis.STATES:
        s_i = state_to_index[s]
        for e in gentrellis.OUTGOING_EDGES_BY_STATE[s]:
            weighted_edge = trellis.WeightedEdge(
                succ=state_to_index[e.succ],
                weight=e.weight,
            )
            weighted_edges_by_state_index[s_i].append(weighted_edge)
        for wob in gentrellis.WEIGHTED_OBS_BY_STATE[s]:
            weighted_obs = trellis.WeightedObservation(
                delta_e=wob.delta_e,
                weight=wob.weight,
            )
            weighted_obs_by_state_index[s_i].append(weighted_obs)

    logprob_prior = numpy.asarray(
        [gentrellis.LOGPROB_PRIOR_BY_STATE[s] for s in gentrellis.STATES],
        dtype=numpy.float64)

    packed_edges = trellis.pack_edges(states, weighted_edges_by_state_index)
    packed_obs = trellis.pack_observations(states, weighted_obs_by_state_index)

    return trellis.HMM(
        state_by_statekey=state_to_index,
        states=states,
        packed_edges=packed_edges,
        packed_observations=packed_obs,
        prior=logprob_prior,
    )


class MonthlyMarkovEventAuxiliarySolver(markov_event.MarkovEventAuxiliarySolver):
    def __init__(self):
        super().__init__()
        self.hmm = make_hmm()

    def _searchfunc(self, times, prizes_u):
        # times must be array of time indices 0...T-1
        # prizes must be shape (T, ) array of prizes (unit: logprob units per event).

        objective_star, logprob_star, state_trajectory, obs_trajectory = trellis.search_best_path(
            self.hmm,
            times,
            prizes_u,
        )
        # Translate from state indices back into fancy states
        fancy_state_trajectory = [gentrellis.prettystate(gentrellis.STATES[s]) for s in
                                  state_trajectory]
        return (objective_star, logprob_star, fancy_state_trajectory, obs_trajectory)