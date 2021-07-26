import collections
import numpy

import pytest


from . import exact_cover_base as base
from . import exact_cover_solver_primal
from . import exact_cover_solver_dual_scipy


@pytest.fixture(scope="module", params=["primal", "dual_scipy"])
def solver_factory(request):
    backend = request.param
    if backend == "primal":
        return lambda: exact_cover_solver_primal.PrimalCoverSolver()
    if backend == "dual_scipy":
        return lambda: exact_cover_solver_dual_scipy.DualCoverSolver()
    raise KeyError(backend)


def test_exact_cover_trivial_single_set_single_element_problem(solver_factory):
    """
    Consider the following example:

    There is a single time labelled 0. So T = {0}.
    There is a single event type labelled 0. So U = {0}.

    We observe 1 event count for t=0 and u=0, so rhs vector b is:

    b = [b_{t=0, u_0}] = [1]

    We have the following possible generators to explain the event count:

    index   probability     counts supplied at t=1, u=1     upper bound
    a       1/e             1                               1

    cost = - log probability = - log(1/e) = 1

    Let x = [x_a] represent primal decision variable

    Clearly x_a=1 is the only feasible value, hence it is optimal.
    The correspondng optimal objective value of min problem is 1.

    If we formulate relaxed exact cover as a max problem, by flipping the sign
    of the costs, the optimal solution is x_a=1 with optimal objective value
    of -1.


    The dual problem for the max problem is:

    b = [1]
    y = [y_{t=0,u=0}]
    w = [w_a]
    u = [1]
    A = [1]
    c = [1]

    min     b^T y + u^T w
    where
        y in R^L unconstrained ; w in R_{>=0}^n

    subject to
        A^T y + w >= -c

    i.e.

    min     y_{t=0, u=0} + w_a
    where
        y_{t=0, u=0} in R unconstrained
        w_a >= 0
    subject to
        y_{t=0, u=0} + w_a >= -1

    This has optimal value of -1
    As expected it agrees with primal optimal value.
    The solution is nonunique:
    any solution s.t.
        w_1 >= 0
        y_{t=1, u=1} + w_1 = -1
    is optimal.
    """

    times = [0]
    event_types = [0]

    z_by_i = {
        'a':base.CandidateSet(cost=numpy.float64(1.0), e=numpy.ones(shape=(1, 1), dtype=numpy.float64)),
    }

    ub_by_i = {
        'a': 1.0,
    }

    u_with_support_t_u = collections.defaultdict(set)
    u_with_support_t_u[(0, 0)].add('a')

    problem = base.ExactCoverResourcePricingProblem(
        times=times,
        event_types=event_types,
        e_hat = numpy.ones(shape=(1, 1), dtype=numpy.float64),
        z_by_i = z_by_i,
        ub_by_i = ub_by_i,
        i_with_support_t_u=u_with_support_t_u,
    )

    expected_objective = -1.0

    s = solver_factory()
    result = s.solve(problem)
    assert result is not None
    assert numpy.allclose(expected_objective, result.objective)


@pytest.fixture(scope="module", params=[1, 2, 4, 8])
def n_time(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 2, 4])
def n_type(request):
    return request.param


def test_exact_cover_small_problem_feasible_basis(solver_factory, n_time, n_type):
    """
    With T=2, and U=1 the dual problem for this max problem is
    u index omitted for brevity as it ranges over 1 value.

    b = [1, 1]
    y = [y_{t=0}, y_{t=1}]
    w = [w_a, w_b]
    u = [1, 1]
    A = [1 0]
        [0 1]
    c = [log(2), log(2)]

    min     b^T y + u^T w
    where
        y in R^L unconstrained ; w in R_{>=0}^n

    subject to
        A^T y + w >= -c

    i.e.

    min     y_{t=0} + y_{t=1} + w_a + w_b
    where
        y in R^L unconstrained ; w in R_{>=0}^n
    subject to
        y_{t=0} + w_a >= -log(2)
        y_{t=1} + w_b >= -log(2)

    note that both constraints in conjuction trivially give
    a lower bound on the objective function of -2 log(2),
    and that lower bound is attained by any solution where the
    two constraints are tight.
    """
    T = n_time
    U = n_type
    times = numpy.arange(T)
    event_types = numpy.arange(U)

    # Two observations to explain away
    e_hat = numpy.zeros(shape=(T, U), dtype=numpy.float64)
    e_hat[0, 0] = 1.0
    e_hat[T-1, U-1] = 1.0

    z_by_i = {}
    ub_by_i = {}
    u_with_support_t_u = collections.defaultdict(set)
    for t in times:
        for u in event_types:
            i = 'once-off;%d;%d' % (t, u)
            e = numpy.zeros(shape=(T, U), dtype=numpy.float64)
            e[t, u] = 1.0
            zi = base.CandidateSet(cost=numpy.log(T)+numpy.log(U), e=e)
            z_by_i[i] = zi
            ub_by_i[i] = 1.0
            u_with_support_t_u[(t, u)].add(i)

    problem = base.ExactCoverResourcePricingProblem(
        times=times,
        event_types=event_types,
        e_hat=e_hat,
        z_by_i=z_by_i,
        ub_by_i=ub_by_i,
        i_with_support_t_u=u_with_support_t_u,
    )

    expected_objective = -1.0 * numpy.sum(e_hat) * (numpy.log(T) + numpy.log(U))

    s = solver_factory()
    result = s.solve(problem)
    assert result is not None
    assert numpy.allclose(expected_objective, result.objective)


@pytest.fixture(scope="module", params=[2])
def lump(request):
    return request.param


def test_exact_cover_small_lumpy_problem_feasible_basis(solver_factory, lump, n_time, n_type):
    """
    same as test_exact_cover_small_problem_feasible_basis except:
    *   we observe lump>1 counts at t=0, u=0, where lump is an integer.
    *   the upper bound for the singleton basis set at t=0, u=0 raised from
        1 to lump > 1 to ensure a feasible solution exists.
    """
    T = n_time
    U = n_type
    times = numpy.arange(T)
    event_types = numpy.arange(U)

    # Two observations to explain away
    e_hat = numpy.zeros(shape=(T, U), dtype=numpy.float64)
    e_hat[0, 0] = 1.0*lump
    e_hat[T-1, U-1] = 1.0

    z_by_i = {}
    ub_by_i = {}
    u_with_support_t_u = collections.defaultdict(set)
    for t in times:
        for u in event_types:
            i = 'once-off;%d;%d' % (t, u)
            e = numpy.zeros(shape=(T, U), dtype=numpy.float64)
            e[t, u] = 1.0
            zi = base.CandidateSet(cost=numpy.log(T)+numpy.log(U), e=e)
            z_by_i[i] = zi
            if t == 0 and u == 0:
                ub_by_i[i] = 1.0 * lump
            else:
                ub_by_i[i] = 1.0
            u_with_support_t_u[(t, u)].add(i)

    problem = base.ExactCoverResourcePricingProblem(
        times=times,
        event_types=event_types,
        e_hat=e_hat,
        z_by_i=z_by_i,
        ub_by_i=ub_by_i,
        i_with_support_t_u=u_with_support_t_u,
    )

    expected_objective = -1.0 * numpy.sum(e_hat) * (numpy.log(T) + numpy.log(U))

    s = solver_factory()
    result = s.solve(problem)
    assert result is not None
    assert numpy.allclose(expected_objective, result.objective)