import numpy

from .model import monthly_markov as gentrellis
from . import trellis
from . import markov_event

_STATE_TO_INDEX = {s:i for i, s in enumerate(gentrellis.STATES)}
_STATES = numpy.asarray([_STATE_TO_INDEX[s] for s in gentrellis.STATES], dtype=int)

def _encode_edge(edge):
    return gentrellis.Edge(succ=_STATE_TO_INDEX[edge.succ], weight=edge.weight, delta_e=edge.delta_e)

_OUTGOING_EDGES = {_STATE_TO_INDEX[s]:[_encode_edge(edge) for edge in gentrellis.OUTGOING_EDGES_BY_STATE[s]] for s in gentrellis.STATES}

_LOGPROB_PRIOR = numpy.asarray([gentrellis.LOGPROB_PRIOR_BY_STATE[s] for s in gentrellis.STATES], dtype=numpy.float64)

_PACKED_EDGES = trellis.pack_edges(_STATES, _OUTGOING_EDGES)


def make_hmm():
    return trellis.HMM(
        state_by_statekey=_STATE_TO_INDEX,
        states=_STATES,
        packed_edges=_PACKED_EDGES,
        prior=_LOGPROB_PRIOR,
    )


class MonthlyMarkovEventAuxiliarySolver(markov_event.MarkovEventAuxiliarySolver):
    def __init__(self):
        super().__init__()

    def _searchfunc(self, times, prizes_u):
        # times must be array of time indices 0...T-1
        # prizes must be shape (T, ) array of prizes (unit: logprob units per event).

        hmm = make_hmm()
        objective_star, logprob_star, state_trajectory, obs_trajectory = trellis.search_best_path(
            hmm,
            times,
            prizes_u,
        )
        # Translate from state indices back into fancy states
        fancy_state_trajectory = [gentrellis.prettystate(gentrellis.STATES[s]) for s in
                                  state_trajectory]
        return (objective_star, logprob_star, fancy_state_trajectory, obs_trajectory)