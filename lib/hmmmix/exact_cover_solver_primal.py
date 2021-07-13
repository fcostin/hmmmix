from . import exact_cover_base as base

import mip
import numpy
import typing


class PrimalExactCoverResourcePricingSolver(base.ExactCoverResourcePricingSolver):

    def __init__(self, regularisation_lambda=0.0):
        self.regularisation_lambda = regularisation_lambda

    def solve(self, problem: base.ExactCoverResourcePricingProblem) -> typing.Optional[base.ExactCoverResourcePricingSolution]:
        T = problem.times
        U = problem.event_types

        ub_by_i = problem.ub_by_i
        z_by_i = problem.z_by_i
        i_with_support_t_u = problem.i_with_support_t_u

        m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

        # These are hacks, was trying to see if i could stop the constraint dual
        # vars from ending up as nans
        # m.emphasis = mip.SearchEmphasis.OPTIMALITY # please solve the dual problem!
        m.lp_method = mip.LP_Method.DUAL  # please solve the dual problem!!

        # TODO what if we directly model and solve the dual?
        # C.f. https://users.wpi.edu/~msarkis/MA2210/EqualityDual.pdf etc.

        x_by_i = {
            i: m.add_var(name=i, var_type=mip.CONTINUOUS, lb=0.0, ub=ub_by_i.get(i, 1.0))
            for i in z_by_i}

        m.objective = mip.xsum(-z.cost * x_by_i[i] for (i, z) in z_by_i.items())

        # Constraints -- forall t forall u balance supply and demand:

        e_hat_l1norm = numpy.sum(numpy.abs(problem.e_hat))
        dual_lambda = self.regularisation_lambda * 0.5 * e_hat_l1norm

        print('|e^| = %g' % (e_hat_l1norm,))
        print('using dual lambda = %g' % (dual_lambda,))

        con_balance_ub_by_t_u = {}
        con_balance_lb_by_t_u = {}
        for t in T:
            for u in U:
                rhs = problem.e_hat[(t, u)]
                ub = mip.xsum(z_by_i[i].e[(t, u)] * x_by_i[i] for i in
                              i_with_support_t_u[(t, u)]) <= rhs + dual_lambda
                lb = mip.xsum(z_by_i[i].e[(t, u)] * x_by_i[i] for i in
                              i_with_support_t_u[(t, u)]) >= rhs - dual_lambda
                con_balance_ub_by_t_u[(t, u)] = m.add_constr(ub)
                con_balance_lb_by_t_u[(t, u)] = m.add_constr(lb)

        status = m.optimize(relax=True)
        if status != mip.OptimizationStatus.OPTIMAL:
            print('failed to find optimal solution: CBC reports status=%r' % (status, ))
            return None

        # Recover primal solution (not yet needed)
        if False:
            soln_x = {}
            for i, x_i in x_by_i.items():
                weight_i = abs(x_i.x)
                if weight_i > 1.0e-6:
                    soln_x[i] = weight_i

        # Recover dual variable value for each constraint
        prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
        # Intuition: I believe when the values of the dual variables are unstable
        # (e.g. very large or even nan) this indicates that the dual problem has
        # too many variables and too few constraints, so the dual solution is
        # not unique and the value of many dual variables y >= 0 is not determined,
        # apart from needing to satisfy A^T y >= c while minimising b^T y .  One way
        # to improve the stability of the dual problem could be to add additional
        # variables (columns) to the restricted master problem -- this would add
        # additional constraints to the dual problem.  Another ad-hoc way to perhaps
        # improve stability is to add a regularisation term to the dual problem
        # objective function. E.g. instead of minimising b^T y we could instead
        # minimising b^T y + lambda R(y) where lambda > 0 is a constant and R(y) is
        # a suitable regularisation penalty. Since we are doing linear programming
        # a natural (read: easy) choice of R(y) would be the L_1 norm of y. Since
        # y >= 0 this is equivalent to adding a + lambda 1^T y term to the dual
        # objective function. Equivalently, back in the primal problem, this is
        # equivalent to replacing the constraints A x <= b with A x <= b + lambda 1 .
        for t in T:
            for u in U:
                pi_ub = con_balance_ub_by_t_u[(t, u)].pi
                pi_lb = con_balance_lb_by_t_u[(t, u)].pi
                if numpy.isnan(pi_ub):
                    print('warning: pi_ub t=%r u=%r is nan' % (t, u))
                if numpy.isnan(pi_lb):
                    print('warning: pi_lb t=%r u=%r is nan' % (t, u))
                prizes[(t, u)] = -pi_ub + pi_lb  # TODO is this antiparallel or parallel

        return base.ExactCoverResourcePricingSolution(
            objective=m.objective_value,
            prizes=prizes,
        )