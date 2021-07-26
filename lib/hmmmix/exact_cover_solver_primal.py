from . import exact_cover_base as base

import mip
import numpy
import typing


class PrimalExactCoverResourcePricingSolver(base.ExactCoverResourcePricingSolver):

    def __init__(self):
        pass

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

        con_balance_ub_by_t_u = {}
        con_balance_lb_by_t_u = {}
        for t in T:
            for u in U:
                rhs = problem.e_hat[(t, u)]
                ub = mip.xsum(z_by_i[i].e[(t, u)] * x_by_i[i] for i in
                              i_with_support_t_u[(t, u)]) <= rhs
                lb = mip.xsum(z_by_i[i].e[(t, u)] * x_by_i[i] for i in
                              i_with_support_t_u[(t, u)]) >= rhs
                con_balance_ub_by_t_u[(t, u)] = m.add_constr(ub)
                con_balance_lb_by_t_u[(t, u)] = m.add_constr(lb)

        status = m.optimize(relax=True)
        if status != mip.OptimizationStatus.OPTIMAL:
            print('failed to find optimal solution: CBC reports status=%r' % (status, ))
            return None

        # Recover dual variable value for each constraint
        prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)

        # BEWARE the recovery of a dual solution obtained through the simplex
        # method in order to define our "prizes" may not be very useful. Adding
        # a new decision variable (column) with negative reduced cost as
        # defined by this dual solution may not increase the objective of the
        # new restricted master problem. A symptom that this is happening is
        # that the auxiliary solver finds a new variable with substantially
        # negative reduced cost, but when it is added to the restricted master
        # problem, it makes no improvement to the objective function.
        #
        # Crude intuition: dual solution from simplex algorithm tells us which
        # direction to take a step in, but not how far we can step. So if we
        # merely search for the steepest direction to step in without
        # considering how far we can step, we may find a very attractive
        # direction to step in turns out to hit a wall with step size zero.
        #
        # See section "Subtleties when using dual solution to select new
        # decision variables". We could probably do a better job of searching
        # for new decision variables to add if we knew the allowable ranges
        # corresponding to each of our dual variables, with respect to the
        # simplex solver's internal state. But alas, we do not.
        #
        # Completely different algorithms for solving LPs such as interior
        # point methods or subgradient methods may produce dual solutions that
        # can
        #
        # A good reference is
        # Jansen, de Jong, Roos, Terlaky (1997)
        # "Sensitivity analysis in linear programming: just be careful!"
        #
        for t in T:
            for u in U:
                # FIXME this nan nonsense may be specific to trying to solve a dual
                # linear program through the layers of python mip & CBC. But we don't
                # necessarily need python mip or CBC (or a primal problem) to do that.
                pi_ub = con_balance_ub_by_t_u[(t, u)].pi
                pi_lb = con_balance_lb_by_t_u[(t, u)].pi
                if numpy.isnan(pi_ub):
                    print('warning: pi_ub t=%r u=%r is nan' % (t, u))
                if numpy.isnan(pi_lb):
                    print('warning: pi_lb t=%r u=%r is nan' % (t, u))

                # + pi_ub - pi_lb seems to do something, but "best aux objective" doesnt seem to be nondecreasing.
                # - pi_ub + pi_lb does not seem to work, it "converges" immediately with no improving aux solution found.
                # - pi_ub - pi_ub seems to do something. "best aux objective" seems to be nondecreasing over successive solves.
                # + pi_ub + pi_ub does not seem to work, it "converges" immediately

                # TODO FIXME figure out _EXACTLY_ what this should be
                prizes[(t, u)] = - pi_ub - pi_lb


        # Recover primal solution
        soln_z = {}
        for i, x_i in x_by_i.items():
            weight_i = abs(x_i.x)
            if weight_i > 1.0e-6:
                soln_z[i] = weight_i

        return base.ExactCoverResourcePricingSolution(
            objective=m.objective_value,
            prizes=prizes,
            z=soln_z,
        )