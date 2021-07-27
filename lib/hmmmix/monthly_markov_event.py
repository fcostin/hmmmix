import numpy

from .model import monthly_markov as gentrellis
from . import trellis
from . import markov_event


def make_hmm():
    state_to_index = {s: i for i, s in enumerate(gentrellis.STATES)}
    states = numpy.asarray([state_to_index[s] for s in gentrellis.STATES], dtype=int)

    def _encode_edge(edge):
        return trellis.Edge(
            succ=state_to_index[edge.succ],
            weight=edge.weight,
            delta_e=edge.delta_e,
        )

    outgoing_edges = {state_to_index[s]: [_encode_edge(edge) for edge in
                                          gentrellis.OUTGOING_EDGES_BY_STATE[s]] for s
                      in gentrellis.STATES}

    logprob_prior = numpy.asarray(
        [gentrellis.LOGPROB_PRIOR_BY_STATE[s] for s in gentrellis.STATES],
        dtype=numpy.float64)

    packed_edges = trellis.pack_edges(states, outgoing_edges)

    return trellis.HMM(
        state_by_statekey=state_to_index,
        states=states,
        packed_edges=packed_edges,
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