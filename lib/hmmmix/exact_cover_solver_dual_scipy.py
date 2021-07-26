from . import exact_cover_base as base

import collections
import itertools
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import numpy
import typing


class DualCoverSolver(base.ExactCoverResourcePricingSolver):

    def __init__(self):
        pass

    def solve(self, problem: base.ExactCoverResourcePricingProblem) -> typing.Optional[base.ExactCoverResourcePricingSolution]:
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

        T = problem.times
        U = problem.event_types

        ub_by_i = problem.ub_by_i
        z_by_i = problem.z_by_i
        i_with_support_t_u = problem.i_with_support_t_u
        e_hat = problem.e_hat

        # we need inverse of this to lookup {tu} given i.
        tu_with_support_by_i = collections.defaultdict(set)
        for tu, ii in i_with_support_t_u.items():
            for i in ii:
                tu_with_support_by_i[i].add(tu)

        TU = list(itertools.product(T, U))

        n_tu = len(TU)
        ii = list(sorted(z_by_i.keys()))
        n_z = len(ii)

        n = n_tu + n_z

        obj_coeffs = numpy.empty(dtype=numpy.float64, shape=(n, ))
        bounds = [None]*n
        j_by_tu = {}
        for j, tu in enumerate(TU):
            j_by_tu[tu] = j
            obj_coeffs[j] = e_hat[tu]
            bounds[j] = (None, None) # each y is unbounded in dual problem
        for j, i in enumerate(ii):
            obj_coeffs[n_tu + j] = ub_by_i.get(i, 1)
            bounds[n_tu + j] = (0.0, None) # each w is nonnegative

        assert numpy.all(numpy.isfinite(obj_coeffs))
        assert numpy.all(obj_coeffs >= 0.0)


        # generate constraints
        # each row of constraint matrix corresponds to some z_i

        con_data = []
        con_col_indices = []
        con_row_indices = []

        for k, i in enumerate(ii):
            # lhs:
            #  - mip.xsum( * y_by_tu[tu] for tu in tus)
            #  - ub_by_i[i]
            #
            # rhs:
            #  zi.cost
            #
            # sense:
            #   lhs <= rhs
            zi = z_by_i[i]
            tus = tu_with_support_by_i[i]
            assert tus

            # there are |tus| + 1 nonzero entries
            n_entries = len(tus) + 1

            data = numpy.empty(shape=(n_entries, ), dtype=numpy.float64)
            col_indices = numpy.empty(shape=(n_entries, ), dtype=numpy.int64)
            row_indices = numpy.empty(shape=(n_entries, ), dtype=numpy.int64)
            row_indices[:] = k

            # Terms corresponding to supply of observation counts by zi over
            # points tu in time-category space.
            for l, tu in enumerate(tus):
                col_indices[l] = j_by_tu[tu]
                data[l] = -1.0 * zi.e[tu]

            # Term corresponding to upper bound on weight of zi
            # Note this is NOT weighted by the upper bound.
            col_indices[n_entries-1] = n_tu + k
            data[n_entries-1] = -1.0

            con_data.append(data)
            con_col_indices.append(col_indices)
            con_row_indices.append(row_indices)

        # assemble sparse matrix
        triplet = (
            numpy.concatenate(con_data),
            (
                numpy.concatenate(con_row_indices),
                numpy.concatenate(con_col_indices),
            )
        )
        # n rows is the number of zi
        # n cols is the number of decision variables
        #
        # Assembly in COO format is easy but decadent (look at those strings
        # of repeated row indices).
        # Internally it looks like linprob will convert this to the compressed
        # sparse matrix format of its choice before attempting to solve.
        a_matrix = coo_matrix(triplet, shape=(len(ii), n), dtype=numpy.float64)

        b = numpy.empty(shape=(len(ii), ), dtype=numpy.float64)
        for k, i in enumerate(ii):
            b[k] = z_by_i[i].cost

        # note that scipy wants upper bounds for inequality constraints

        # We force the use of interior point method in the hope of recovering
        # a solution that is more useful for defining "prizes" for the auxiliary
        # problem.

        res = linprog(
            c=obj_coeffs,
            A_ub=a_matrix,
            b_ub=b,
            bounds=bounds,
            # method='highs-ipm', # FIXME highs solvers output unhelpful solutions.
            method='interior-point',
            options={
                'disp': True,
                'presolve': True,
                'tol': 1.0e-12,
            }
        )

        success = res['success']
        if not success:
            status = res['status']
            message = res['message']
            print('failed to find an optimal solution: status=%r; message=%r' % (status, message))
            return None

        # recover solution
        objective_value = res['fun']

        # Recover solution for y variables in order to define prizes
        prizes = numpy.zeros(shape=(len(T), len(U)), dtype=numpy.float64)
        y = res['x']

        for j, tu in enumerate(TU):
            prizes[tu] = -1.0 * y[j]

        return base.ExactCoverResourcePricingSolution(
            objective=objective_value,
            prizes=prizes,
            z={}, # recovery of primal solution not supported by this solver.
        )

