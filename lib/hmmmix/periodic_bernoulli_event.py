import numpy
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
        objective_star = 0.0
        soln_star = None
        for raw_soln in periodic_bernoulli.gen_solns(
                n_times=len(problem.times),
                n_event_types=len(problem.event_types),
                prizes=problem.prizes,
                decompose=True,
                period=self.period,
                n_R=self.resolution,
                verbose=False,
            ):
            if raw_soln['obj'] <= objective_star:
                continue
            if raw_soln['id'] in self.banned_ids:
                continue
            objective_star = raw_soln['obj']
            soln_star = base.AuxiliarySolution(
                id=raw_soln['id'],
                objective=raw_soln['obj'],
                logprob=raw_soln['log_prob'],
                e=raw_soln['events'],
            )

        return soln_star

    def gen_solns(self, problem: base.AuxiliaryProblem) -> typing.Optional[base.AuxiliarySolution]:
        obj_lower_bound = 1.0e-4 # roughly zero.
        for raw_soln in periodic_bernoulli.gen_solns(
                n_times=len(problem.times),
                n_event_types=len(problem.event_types),
                prizes=problem.prizes,
                decompose=True,
                period=self.period,
                n_R=self.resolution,
                verbose=False,
            ):
            if raw_soln['obj'] <= obj_lower_bound:
                continue
            if raw_soln['id'] in self.banned_ids:
                continue
            yield base.AuxiliarySolution(
                id=raw_soln['id'],
                objective=raw_soln['obj'],
                logprob=raw_soln['log_prob'],
                e=raw_soln['events'],
            )