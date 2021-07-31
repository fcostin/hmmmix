import numpy
import typing

from . import base


# TODO FIXME speedup this using bottom up dynamic programming over trellis of
# T * S where S is hidden states giving clock of generating process.

def iter_ongoing_patterns(month_duration, T):
    n_phases = month_duration
    log_prior_prob = -numpy.log(n_phases) # for each possible phase
    for phase in range(n_phases):
        pattern_id = 'ongoing;phase=%d' % (phase, )
        months = []
        for month_start in range(-phase, T, month_duration):
            month_start = max(0, month_start)
            month_end = min(month_start + month_duration, T)
            months.append((month_start, month_end))
        yield pattern_id, log_prior_prob, months


def iter_commencing_patterns(month_duration, T):
    log_prior_prob = -numpy.log(T) # for each possible starting date
    for start in range(T):
        pattern_id = 'commencing;start=%d' % (start, )
        months = []
        for month_start in range(start, T, month_duration):
            month_end = min(month_start + month_duration, T)
            months.append((month_start, month_end))
        yield pattern_id, log_prior_prob, months


def iter_ending_patterns(month_duration, T):
    n_phases = month_duration
    log_prior_prob = -numpy.log(n_phases) -numpy.log(T) # for each possible phase and end date
    for phase in range(n_phases):
        for end in range(T+1):
            pattern_id = 'ending;phase=%d;end=%d' % (phase, end)
            months = []
            for month_start in range(-phase, end, month_duration):
                month_start = max(0, month_start)
                month_end = min(month_start + month_duration, T)
                months.append((month_start, month_end))
            yield pattern_id, log_prior_prob, months


def iter_transient_patterns(month_duration, T):
    log_prior_prob = -numpy.log(T*(T+1)*0.5) # for each possible start date and end date
    for start in range(T):
        for end in range(start+1, T+1):
            pattern_id = 'transient;start=%d;end=%d' % (start, end)
            months = []
            for month_start in range(start, end, month_duration):
                month_start = max(0, month_start)
                month_end = min(month_start + month_duration, T)
                months.append((month_start, month_end))
            yield pattern_id, log_prior_prob, months


def iter_all_patterns(month_duration, T):
    iters = (
        iter_ongoing_patterns,
        iter_commencing_patterns,
        # iter_ending_patterns,
        # iter_transient_patterns,
    )
    # Penalise prior again for our uncertainty of not knowing which pattern type to pick!
    log_prior_pattern_type = -numpy.log(len(iters))
    for i in iters:
        for pattern_id, log_prior_prob, months in i(month_duration, T):
            yield pattern_id, log_prior_prob + log_prior_pattern_type, months


class OncePerMonthEventAuxiliarySolver(base.AuxiliarySolver):
    """
    A once per month event generator generators a single
    observable event once every month (on a random day),
    of some type u in U and at some time t in T.

    need to consider:
    *   month duration (30)
    *   phase
    *   random day in each month
    """

    def __init__(self):
        self.banned_ids = set()

    def exclude(self, solution: base.AuxiliarySolution):
        self.banned_ids.add(solution.id)

    def solve(self, problem: base.AuxiliaryProblem) -> typing.Optional[base.AuxiliarySolution]:
        month_duration = 30 # TODO MUST FIX THIS!
        T = len(problem.times)
        U = len(problem.event_types)

        max_objective = -numpy.inf
        max_logprob = None
        max_description = None
        max_e = None

        for u in problem.event_types:
            # case: ongoing process -- no start, no stop
            for pattern_id, log_prior_prob_pattern, months in iter_all_patterns(month_duration, T):
                acc_objective = -numpy.log(U) + log_prior_prob_pattern
                acc_logprob = -numpy.log(U) + log_prior_prob_pattern

                trajectory = []
                # The events generated by our generator
                e = numpy.zeros(shape=problem.prizes.shape, dtype=numpy.int64)

                for (month_start, month_end) in months:
                    month_start = max(0, month_start)
                    month_end = min(month_start + month_duration, T)
                    month_prizes = problem.prizes[month_start:month_end, u]
                    # Since probability of generating an event on any day in
                    # month is uniform, maximise this part of objective func by
                    # simpling generating event on any day of month with maximal prize.
                    i_star = numpy.argmax(month_prizes)
                    t_star = month_start + i_star
                    e[(t_star, u)] += 1

                    prize = month_prizes[i_star]

                    # there was a good chance we generated event on some other day
                    # since we don't assume pattern on any particular day of month.
                    log_prob_emit_on_tstar = -numpy.log(month_duration)

                    acc_objective += prize + log_prob_emit_on_tstar
                    acc_logprob += log_prob_emit_on_tstar

                    trajectory.append(i_star)

                if acc_objective <= max_objective:
                    continue

                candidate_description = 'pattern=%s;u=%d;trajectory=%r' % (
                    pattern_id,
                    u,
                    trajectory,
                )
                if candidate_description in self.banned_ids:
                    continue

                max_objective = acc_objective
                max_logprob = acc_logprob
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