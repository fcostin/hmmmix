import collections
import numpy
import typing

from . import base

from . import once_off_event # TODO master shouldnt depend on concrete aux solvers
from . import once_per_month_event
from . import monthly_markov_event

from . import exact_cover_base
from . import exact_cover_solver_primal
from . import exact_cover_solver_dual_scipy


def make_soln_id(solver_id, soln_id):
    return '%s;%s' % (solver_id, soln_id)


def bootstrap_initial_basis(aux_solver: once_off_event.OnceOffEventAuxiliarySolver, problem: base.MasterProblem):
    # This needs to be positive and have larger magnitude than
    # the log prob of some feasible solution to our (t, u) once-off event
    mega_desperate_prize = 1.0e9

    T = problem.times
    U = problem.event_types

    z_by_i = {}
    ub_by_i = {} # upper bound

    for t in T:
        for u in U:
            # Previously i wasn't generating any possible aux_soln for (t, u) in the
            # case where e_hat[(t, u)] was zero. But maybe we should offer the option.

            prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
            prizes[(t, u)] = mega_desperate_prize
            aux_problem = base.AuxiliaryProblem(
                times=T,
                event_types=U,
                prizes=prizes,
            )
            aux_soln = aux_solver.solve(aux_problem)
            assert aux_soln is not None
            assert base.is_auxiliary_solution_sane(aux_problem, aux_soln)
            assert aux_soln.logprob < 0.0
            soln_id = make_soln_id('once-off', aux_soln.id)
            z_by_i[soln_id] = aux_soln
            ub_by_i[soln_id] = max(problem.e_hat[(t, u)], 1)

    # Also define solutions that allow *negative* observation counts. Consider
    # this equivalent to a relaxation of the problem where we allow a single
    # observed event to be (rather unsatisfyingly) explained as being caused by
    # multiple causes that we expect to have produced two events, but for some
    # reason we only saw a single event, i.e. one "went missing" somehow.
    # There may not be a good justification for allowing this in a given
    # application. Yet it is very enticing to allow this relaxation, as the
    # relaxed problem appears much easier to solve!
    #
    # This trick is inspired by an equivalent relaxation as applied to relax
    # vehicle routing problems, as described in column-generation literature.
    # Ref: Rousseau, Gendreau, Feillet (2007)
    # "Interior point stabilisation for column generation."

    for t in T:
        for u in U:
            prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
            prizes[(t, u)] = mega_desperate_prize
            aux_problem = base.AuxiliaryProblem(
                times=T,
                event_types=U,
                prizes=prizes,
            )
            aux_soln = aux_solver.solve(aux_problem)
            assert aux_soln is not None
            assert base.is_auxiliary_solution_sane(aux_problem, aux_soln)
            assert aux_soln.logprob < 0.0

            e_negative = numpy.copy(aux_soln.e)
            assert e_negative[(t, u)] == 1
            e_negative[(t, u)] = -1

            negative_aux_soln = base.AuxiliarySolution(
                id=(aux_soln.id + ';negative'),
                objective=0.0, # should not matter,
                logprob=aux_soln.logprob - numpy.log(10000.0), # 10,000 times less likely than positive.
                e=e_negative,
            )

            soln_id = make_soln_id('once-off', negative_aux_soln.id)
            z_by_i[soln_id] = negative_aux_soln
            ub_by_i[soln_id] = 1 # allow at most 1 negative observation count

    return z_by_i, ub_by_i


def e_support(aux_solution: base.AuxiliarySolution) -> typing.Set[int]:
    ts, us = numpy.nonzero(aux_solution.e)
    rr = set(zip(ts, us))
    return rr


class RelaxedMasterSolver(base.MasterSolver):

    def __init__(self, obj_cutoff=None):
        self.obj_cutoff = obj_cutoff

    def solve(self, problem: base.MasterProblem) -> typing.Optional[base.MasterSolution]:
        T = problem.times
        U = problem.event_types

        aux_solvers_by_id: typing.Dict[str, base.AuxiliarySolver] = {  # TODO dep inject
            'once-off': once_off_event.OnceOffEventAuxiliarySolver(),
        #    'once-per-month': once_per_month_event.OncePerMonthEventAuxiliarySolver(),
            'monthly': monthly_markov_event.MonthlyMarkovEventAuxiliarySolver(),
        }

        # Shorthand: let z[i] denote i-th Aux solution
        z_by_i: typing.Dict[str, base.AuxiliarySolution] = {}

        # Bootstrap an initial feasible solution
        print('bootstrapping initial feasible soln')
        z_by_i, ub_by_i = bootstrap_initial_basis(aux_solvers_by_id['once-off'], problem)
        for z in z_by_i.values():
            aux_solvers_by_id['once-off'].exclude(z)
        print('ok')

        # Iteratively solve relaxed restricted master problem

        # Maintain cache of which i have nonzero e[(t, u)]
        i_with_support_t_u = collections.defaultdict(set)
        for i, z in z_by_i.items():
            for tu in e_support(z):
                i_with_support_t_u[tu].add(i)


        while True:
            print('iter...')

            use_primal = False
            if use_primal:
                solver = exact_cover_solver_primal.PrimalCoverSolver(
                )
            else:
                solver = exact_cover_solver_dual_scipy.DualCoverSolver()

            cover_problem = exact_cover_base.ExactCoverResourcePricingProblem(
                times=problem.times,
                event_types=problem.event_types,
                e_hat=problem.e_hat,
                z_by_i={i:exact_cover_base.CandidateSet(cost=-z.logprob, e=z.e) for i, z in z_by_i.items()},
                ub_by_i=ub_by_i,
                i_with_support_t_u=i_with_support_t_u,
            )

            cover_solution = solver.solve(cover_problem)
            if cover_solution is None:
                print('error - restricted relaxed exact cover problem infeasible. add more candidate sets!?')
                return None

            obj = cover_solution.objective
            print("restricted relaxed exact cover problem solved, objective=%r" % (obj, ))

            if self.obj_cutoff != None and self.obj_cutoff <= obj:
                print("halting as objective cutoff %g exceeded by objective %g" % (self.obj_cutoff, obj))
                return cover_solution

            prizes = cover_solution.prizes

            aux_problem = base.AuxiliaryProblem(
                times=T,
                event_types=U,
                prizes=prizes,
            )

            min_improvement = 10.e-9

            best_aux_objective = min_improvement # status quo has reduced cost 0. require strictly positive
            best_aux_soln = None
            best_aux_solver_id = None

            for aux_solver_id, aux_solver in aux_solvers_by_id.items():
                aux_soln = aux_solver.solve(aux_problem)
                if aux_soln is None:
                    continue
                assert base.is_auxiliary_solution_sane(aux_problem, aux_soln)
                if aux_soln.objective <= best_aux_objective:
                    continue
                assert not numpy.isnan(aux_soln.objective)
                best_aux_objective = aux_soln.objective
                best_aux_soln = aux_soln
                best_aux_solver_id = aux_solver_id

            if best_aux_soln is None:
                return cover_solution

            print('best aux objective: %r' % (best_aux_objective, ))

            new_i = make_soln_id(best_aux_solver_id, best_aux_soln.id)

            print('adding column z[i]: %s' % (new_i, ))

            # Ban solver from generating the same aux solution again.
            aux_solvers_by_id[best_aux_solver_id].exclude(best_aux_soln)

            assert new_i not in z_by_i
            z_by_i[new_i] = best_aux_soln

            for tu in e_support(best_aux_soln):
                i_with_support_t_u[tu].add(new_i)




