from . import exact_cover_base as base

import collections
import itertools
import mip
import numpy
import typing


class DualExactCoverResourcePricingSolver(base.ExactCoverResourcePricingSolver):

    def solve(self, problem: base.ExactCoverResourcePricingProblem) -> typing.Optional[base.ExactCoverResourcePricingSolution]:
        T = problem.times
        U = problem.event_types

        ub_by_i = problem.ub_by_i
        z_by_i = problem.z_by_i
        i_with_support_t_u = problem.i_with_support_t_u
        e_hat = problem.e_hat

        """
        Note we regard c as cost hence to minimise cost in primal problem
        we want to maximise using coeffs -c.
        
        Primal formulation:
        
        max     -c^T x
        
        over x >= 0 , x in R^n
        
        subject to
        
            Ax   =  b   # m constraints. note m = |T|*|U|
             x  <=  u   # n constraints
             
        Note that u is usually a vector of 1s, but some entries may have values > 1
        where we allow need multiple copies of candidate sets in initial basis to
        supply enough resources s.t.that A x = b is even feasible.
        
        Dual formulation:
        
        min     b^T y + u^T w
        
        over
            y unrestricted , y in R^m
            w >= 0 , w in R^n
        
        subject to
        
            A^T y + w  >=  -c   # n constraints
        """

        m = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        m.lp_method = mip.LP_Method.PRIMAL # eat what you're given!

        # we need inverse of this to lookup {tu} given i.
        tu_with_support_by_i = collections.defaultdict(set)
        for tu, ii in i_with_support_t_u.items():
            for i in ii:
                tu_with_support_by_i[i].add(tu)

        TU = itertools.product(T, U)

        y_by_tu = {
            (t, u): m.add_var(name='y_%d;%d' % (t, u), var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
            for (t, u) in TU}

        w_by_i = {i: m.add_var(name='w_%s' % (i, ), var_type=mip.CONTINUOUS) for i, z in z_by_i.items()}

        m.objective = (
            mip.xsum(e_hat[tu] * y_tu for tu, y_tu in y_by_tu.items())
            + mip.xsum(ub_by_i.get(i, 1) * w_i for i, w_i in w_by_i.items())
        )

        con_by_i = {}
        for i, zi in z_by_i.items():
            tus = tu_with_support_by_i[i]
            assert tus
            lhs = mip.xsum(zi.e[tu] * y_by_tu[tu] for tu in tus) + ub_by_i[i] * w_by_i[i]
            rhs = -zi.cost
            lb = lhs >= rhs
            con_by_i[i] = m.add_constr(lb)

        status = m.optimize(relax=True)
        if status != mip.OptimizationStatus.OPTIMAL:
            print('failed to find optimal solution: CBC reports status=%r' % (status,))
            return None

        prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
        # Recover solution for y variables in order to define prizes
        for tu, y_tu in y_by_tu.items():
            prizes[tu] = y_by_tu[tu].x # TODO SIGN?


        return base.ExactCoverResourcePricingSolution(
            objective=m.objective_value,
            prizes=prizes,
        )

