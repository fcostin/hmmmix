import typing

from . import base
from .model import periodic_bernoulli


class PeriodicBernoulliAuxiliarySolver(base.AuxiliarySolver):

    def __init__(self, period, resolution=4):
        self.period = period
        self.resolution = resolution
        self.banned_ids = set()

    def exclude(self, solution: base.AuxiliarySolution):
        self.banned_ids.add(solution.id)

    def solve(self, problem: base.AuxiliaryProblem) -> typing.Optional[base.AuxiliarySolution]:
        result = periodic_bernoulli.solve(
            n_times=len(problem.times),
            n_event_types=len(problem.event_types),
            prizes=problem.prizes,
            decompose=True,
            period=self.period,
            n_R=self.resolution,
            verbose=False,
        )

        if result['id'] in self.banned_ids:
            return None

        return base.AuxiliarySolution(
            id=result['id'],
            objective=result['obj'],
            logprob=result['log_prob'],
            e=result['events'],
        )