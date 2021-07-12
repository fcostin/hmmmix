import collections
import numpy
import mip
import typing
import sys

from . import base

from . import once_off_event # TODO master shouldnt depend on concrete aux solvers



def make_soln_id(solver_id, soln_id):
    return '%s;%s' % (solver_id, soln_id)


def bootstrap_initial_basis(problem: base.MasterProblem):

    aux_solver = once_off_event.OnceOffEventAuxiliarySolver()

    # This needs to be positive and have larger magnitude than
    # the log prob of some feasible solution to our (t, u) once-off event
    mega_desperate_prize = 1.0e9

    T = problem.times
    U = problem.event_types

    z_by_i = {}
    ub_by_i = {} # upper bound

    for t in T:
        for u in U:
            if problem.e_hat[(t, u)] == 0:
                continue
            ub = problem.e_hat[(t, u)]
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
            ub_by_i[soln_id] = ub
    return z_by_i, ub_by_i


def e_support(aux_solution: base.AuxiliarySolution) -> typing.Set[int]:
    ts, us = numpy.nonzero(aux_solution.e)
    rr = set(zip(ts, us))
    return rr


class RelaxedMasterSolver(base.MasterSolver):

    def solve(self, problem: base.MasterProblem) -> typing.Optional[base.MasterSolution]:
        T = problem.times
        U = problem.event_types

        # Shorthand: let z[i] denote i-th Aux solution
        z_by_i: typing.Dict[str, base.AuxiliarySolution] = {}

        # Bootstrap an initial feasible solution
        print('bootstrapping initial feasible soln')
        z_by_i, ub_by_i = bootstrap_initial_basis(problem)
        print('ok')

        # Iteratively solve relaxed restricted master problem


        # Maintain cache of which i have nonzero e[(t, u)]
        i_with_support_t_u = collections.defaultdict(set)
        for i, z in z_by_i.items():
            for tu in e_support(z):
                i_with_support_t_u[tu].add(i)


        while True:
            print('iter...')
            m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

            # These are hacks, was trying to see if i could stop the constraint dual
            # vars from ending up as nans
            # m.emphasis = mip.SearchEmphasis.OPTIMALITY # please solve the dual problem!
            # m.lp_method = mip.LP_Method.DUAL # please solve the dual problem!!

            # TODO what if we directly model and solve the dual?
            # C.f. https://users.wpi.edu/~msarkis/MA2210/EqualityDual.pdf etc.

            x_by_i = {i: m.add_var(name=i, var_type=mip.CONTINUOUS, lb=0.0, ub=ub_by_i.get(i, 1.0)) for i in z_by_i}

            # maximise log P(H|D)
            m.objective = mip.xsum(z.logprob * x_by_i[i] for (i, z) in z_by_i.items())

            # Constraints -- forall t forall u balance supply and demand:


            con_balance_by_t_u = {}
            for t in T:
                for u in U:
                    rhs = problem.e_hat[(t, u)]
                    linexpr = mip.xsum(z_by_i[i].e[(t, u)] * x_by_i[i] for i in i_with_support_t_u[(t, u)]) == rhs
                    con = m.add_constr(linexpr)
                    con_balance_by_t_u[(t, u)] = con

            status = m.optimize(relax=True)
            assert status == mip.OptimizationStatus.OPTIMAL

            print('objective: %g' % (m.objective_value, ))

            # Recover solution
            soln_x = {}
            for i, x_i in x_by_i.items():
                weight_i = abs(x_i.x)
                if weight_i > 1.0e-6:
                    soln_x[i] = weight_i

            # Recover dual variable value for each constraint
            prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
            for t in T:
                for u in U:
                    pi = con_balance_by_t_u[(t, u)].pi
                    assert not numpy.isnan(pi) # What the.
                    prizes[(t, u)] = - pi # TODO is this antiparallel or parallel


            aux_solvers_by_id: typing.Dict[str, base.AuxiliarySolver] = { # TODO dep inject
                'once-off': once_off_event.OnceOffEventAuxiliarySolver(),
            }

            aux_problem = base.AuxiliaryProblem(
                times=T,
                event_types=U,
                prizes=prizes,
            )

            best_aux_objective = 0.0 # status quo has reduced cost 0. require strictly positive
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
                print('converged')
                return

            print('best aux objective: %r' % (best_aux_objective, ))

            new_i = make_soln_id(best_aux_solver_id, best_aux_soln.id)

            print('adding column z[i]: %s' % (new_i, ))

            assert new_i not in z_by_i
            z_by_i[new_i] = best_aux_soln

            for tu in e_support(best_aux_soln):
                i_with_support_t_u[tu].add(new_i)



def main():
    fn = sys.argv[1]

    e_hat = numpy.load(fn)

    e_hat = e_hat[:, :]

    n_time, n_type = e_hat.shape

    print(repr((n_time, n_type)))

    T = numpy.arange(n_time)
    U = numpy.arange(n_type)

    master = RelaxedMasterSolver()

    problem = base.MasterProblem(
        times=T,
        event_types=U,
        e_hat=e_hat,
    )

    soln = master.solve(problem)


if __name__ == '__main__':
    main()