import numpy
import typing

from . import base


class MarkovEventAuxiliarySolver(base.AuxiliarySolver):

    def __init__(self):
        self.banned_ids = set()

    def exclude(self, solution: base.AuxiliarySolution):
        self.banned_ids.add(solution.id)

    def _searchfunc(self, times, prizes_u):
        raise NotImplementedError('subclass me')

    def solve(self, problem: base.AuxiliaryProblem) -> typing.Optional[base.AuxiliarySolution]:
        T = len(problem.times)
        U = len(problem.event_types)

        max_objective = -numpy.inf
        max_logprob = None
        max_description = None
        max_e = None

        logprob_prior_u = -numpy.log(U)

        for u in problem.event_types:

            prizes_u = problem.prizes[:, u]

            (objective, logprob, fancy_state_trajectory, obs_trajectory) = self._searchfunc(problem.times, prizes_u)

            # Account for us not knowing which u to pick
            objective += logprob_prior_u
            logprob += logprob_prior_u

            if objective <= max_objective:
                continue

            candidate_description = 'u=%d;' % (u, ) + ('-'.join(fancy_state_trajectory))

            # TODO it is defective to implement banning solutions like this -
            # a correct implementation needs to implement banned solutions
            # as constraints in the modified Viterbi trellis problem. E.g.
            # if the best 3 solutions are banned, the Viterbi trellis problem
            # should spit out the 4th best.
            if candidate_description in self.banned_ids:
                continue

            # broadcast obs_trajectory for u back into full shape
            e = numpy.zeros(shape=problem.prizes.shape, dtype=numpy.int)
            e[:, u] = obs_trajectory

            max_objective = objective
            max_logprob = logprob
            max_e = e
            max_description = candidate_description


        if max_description is None:
            return None

        return base.AuxiliarySolution(
            id=max_description,
            objective=max_objective,
            logprob=max_logprob,
            e=max_e,
        )