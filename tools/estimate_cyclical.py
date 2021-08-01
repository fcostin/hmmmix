import argparse
import numpy
import itertools
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
                   help='filename of input binary npy data file')
    p.add_argument('--profile', '-p', action='store_true')
    return p.parse_args()


def solve(n_times, n_event_types, prizes):
    """
    Prototype auxiliary solver.

    Assumes at most one event per day d and event type u.
    Assumes events generated independently for each (d, u)
    according to Bernoulli distribution with probability q_{d, u}

    Generative model:
    for each week w:
        for each weekday d in w:
            for each event type u:
                emit event
                    with probability q_{d, u} = sum_r y_{d, u, r} q_{d,u,r}
                with
                    t = day d of week w
                    event_type = u

    General scheme of formulation:

        E be observed events
        H be hypotheses

    max log P(E, H) + log P(H) + Prize(E)
    over E & H
    subject to constraints.

    Detailed formulation:

    Index sets:

    *   w in W: indexes over periods ("weeks")
    *   d in D: indexes within periods ("days")
    *   u in U: indexes over event types
    *   r in R: indexes over approximating probabilities

    Input parameters (all constants):

    *   for all {r}:        log q_{d,u,r} in [-inf, 0]
    *   for all {d,u,r}:    log P(q_{d,u}=q_{d,u,r}) in [-inf, 0]
    *   for all {w,d,u}:    prize(t=t(w, d), u=u) in [-inf, inf]

    Maximise

            sum_{w,d,u,r} z^{+}_{w,d,u,r} log q_{d,u,r}
        +   sum_{w,d,u,r} z^{-}_{w,d,u,r} log (1 - q_{d,u,r})
        +   sum_{w,d,u,r} z^{+}_{w,d,u,r} prize(t=t(w, d), u=u)
        +   sum_{d,u,r} y_{d,u,r} log P(q_{d,u}=q_{d,u,r})

    over decision variables

        z^{+}_{w,d,u,r} : W x D x U x R -> {0, 1} relaxed to [0, 1]
        z^{-}_{w,d,u,r} : W x D x U x R -> {0, 1} relaxed to [0, 1]
        y_{d,u,r} : D x U x R -> {0, 1} relaxed to [0, 1]

    subject to constraints

        for all {w,d,u}:             sum_{r} z^{+}_{w,d,u,r}  <=  1
        for all {d,u}:                     sum_{r} y_{d,u,r}  <=  1
        for all {w,d,u,r}:   z^{+}_{w,d,u,r} + z^{-}_{w,d,u,r} =  y_{d,u,r}

    Notes:

    Ideally we would optimize with decision variables q_{d, u} in [0, 1], but
    this introduces nonlinear terms of the form z log(q) into the objective
    function. To avoid this we approximate each q_{d, u} by one of R
    prespecified constant probability values. We break the nonlinearity by
    making R copies of each original decision variable and adding additional
    variables and constraints to encourage a single approximation r in R to
    be chosen for each q_{d, u}.

    The variable z^{-}_{w,d,u,r} is used to encode "0 events of type u were
    observed at time (w, d)". It is necessary to use this instead of
    1 - z^{+}_{w,d,u,r} so we do not penalise the objective when terms for
    values of r that have not been "chosen" via y_{d,u,r} are forced to vanish.
    """

    period = 7

    offcut = n_times % period
    if offcut != 0:
        print("todo fixme: ignoring last %d timestemps")
        n_times -= offcut
        prizes = prizes[:-offcut, :]

    n_periods = int(numpy.ceil(n_times / period))

    assert n_times == period * n_periods

    n_R = 2 ** 2

    W = numpy.arange(n_periods)
    D = numpy.arange(period)
    U = numpy.arange(n_event_types)
    R = numpy.arange(n_R)

    WDUR = list(itertools.product(W, D, U, R))
    DUR = list(itertools.product(D, U, R))
    WDU = list(itertools.product(W, D, U))
    DU = list(itertools.product(D, U))

    n_WDUR = len(WDUR)
    n_DUR = len(DUR)
    n_WDU = len(WDU)
    n_DU = len(DU)

    n = 2*n_WDUR + n_DUR # number of decision variables

    print('n_WDUR=%r, n_DUR=%r, n_WDU=%r, n_DU=%r' % (n_WDUR, n_DUR, n_WDU, n_DU))

    approx_probabilities = (0.5 ** numpy.arange(1, n_R+1))
    print('approx_probabilities: %r' % (approx_probabilities, ))

    log_q = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_q[:, :, :] = numpy.log(approx_probabilities)[numpy.newaxis, numpy.newaxis, :]
    log_one_minus_q = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_one_minus_q[:, :, :] = numpy.log(1.0 - approx_probabilities)[numpy.newaxis, numpy.newaxis, :]

    log_prior_q  = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_prior_q[:, :, :] = numpy.log(1.0 / n_R) # choose one of n_r choices, uniform prior.

    # Assemble decision variables and objective terms

    obj_coeffs = numpy.empty(dtype=numpy.float64, shape=(n,))
    bounds = [None] * n

    zp_i_by_wdur = {}
    zm_i_by_wdur = {}
    y_i_by_dur = {}

    for i, wdur in enumerate(WDUR):  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        t = w * period + d
        obj_coeffs[i] = log_q[d,u,r] + prizes[t, u]
        bounds[i] = (0.0, 1.0)
        zp_i_by_wdur[wdur] = i

    for i0, wdur in enumerate(WDUR):  # z^{-}_{w,d,u,r}
        i = i0 + n_WDUR
        _, d, u, r = wdur
        obj_coeffs[i] = log_one_minus_q[d,u,r]
        bounds[i] = (0.0, 1.0)
        zm_i_by_wdur[wdur] = i

    for i0, dur in enumerate(DUR):
        i = i0 + 2*n_WDUR
        d,u,r = dur
        obj_coeffs[i] = log_prior_q[d,u,r]
        bounds[i] = (0.0, 1.0)
        y_i_by_dur[dur] = i

    assert numpy.all(numpy.isfinite(obj_coeffs))

    # scipy.optimize.linprog minimizes! flip all the signs!
    obj_coeffs *= -1.0

    # Assemble inequality constraints

    n_entries = (n_WDU * n_R) + (n_DU * n_R)
    con_data = numpy.empty(shape=(n_entries,), dtype=numpy.float64)
    col_indices = numpy.empty(shape=(n_entries,), dtype=numpy.int64)
    row_indices = numpy.empty(shape=(n_entries,), dtype=numpy.int64)

    entry = 0
    con = 0

    # constraint: for all {w,d,u}:  sum_{r} z^{+}_{w,d,u,r}  <=  1
    # -- n_WDU constraints, each with n_R LHS entries
    for j, wdu in enumerate(WDU):
        w,d,u = wdu
        con_data[entry:entry+n_R] = 1.0
        for r in range(n_R):
            assert zp_i_by_wdur[(w,d,u,r)] < n
            col_indices[entry+r] = zp_i_by_wdur[(w,d,u,r)]
        row_indices[entry:entry + n_R] = con + j
        entry += n_R
    con += n_WDU

    # constraint: for all {d,u}:  sum_{r} y_{d,u,r}  <=  1
    # -- n_DU constraints, each with n_R LHS entries
    for j, du in enumerate(DU):
        d, u = du
        con_data[entry:entry + n_R] = 1.0
        for r in range(n_R):
            assert y_i_by_dur[(d, u, r)] < n
            col_indices[entry + r] = y_i_by_dur[(d, u, r)]
        row_indices[entry:entry + n_R] = con + j
        entry += n_R
    con += n_DU

    assert entry == n_entries

    a_ub_matrix = coo_matrix(
        (con_data, (row_indices, col_indices)),
        shape=(con, n),
        dtype=numpy.float64,
    )

    b_ub_vector = numpy.ones(shape=(con, ), dtype=numpy.float64)

    # Assemble equality constraints

    n_eq_entries = (n_WDUR * 3)
    con_eq_data = numpy.empty(shape=(n_eq_entries,), dtype=numpy.float64)
    col_eq_indices = numpy.empty(shape=(n_eq_entries,), dtype=numpy.int64)
    row_eq_indices = numpy.empty(shape=(n_eq_entries,), dtype=numpy.int64)

    entry = 0
    con = 0

    # constraint: for all {w,d,u,r}: z^{+}_{w,d,u,r} + z^{-}_{w,d,u,r} =  y_{d,u,r}
    # -- n_WDUR constraints, each with 3 LHS entries
    for j, wdur in enumerate(WDUR):
        w,d,u,r = wdur
        con_eq_data[entry] = 1.0
        con_eq_data[entry+1] = 1.0
        con_eq_data[entry+2] = -1.0
        col_eq_indices[entry] = zp_i_by_wdur[wdur]
        col_eq_indices[entry+1] = zm_i_by_wdur[wdur]
        col_eq_indices[entry+2] = y_i_by_dur[(d,u,r)]
        row_eq_indices[entry:entry+3] = con + j
        entry += 3
    con += n_WDUR

    assert entry == n_eq_entries

    a_eq_matrix = coo_matrix(
        (con_eq_data, (row_eq_indices, col_eq_indices)),
        shape=(con, n),
        dtype=numpy.float64,
    )

    b_eq_vector = numpy.zeros(shape=(con,), dtype=numpy.float64)

    res = linprog(
        c=obj_coeffs,
        A_ub=a_ub_matrix,
        b_ub=b_ub_vector,
        A_eq=a_eq_matrix,
        b_eq=b_eq_vector,
        bounds=bounds,
        method='interior-point',
        options={
            'disp': True,
            'presolve': False, # Presolve gives 12x slowdown. Disable it!
            'tol': 1.0e-8,
            'sparse': True,
        }
    )

    success = res['success']
    if not success:
        status = res['status']
        message = res['message']
        print('failed to find an optimal solution: status=%r; message=%r' % (
        status, message))
        return None

    # recover objective
    objective_value = res['fun']

    print('got objective: %r' % (objective_value), )

    # recover solution
    primal_soln = res['x']

    for dur in DUR:
        y_dur = primal_soln[y_i_by_dur[dur]]
        y_dur = numpy.round(y_dur, decimals=2)
        if y_dur > 0.0:
            print('y[%r] = %r' % (dur, y_dur))

    for wdur in WDUR:
        zp_wdur = primal_soln[zp_i_by_wdur[wdur]]
        zm_wdur = primal_soln[zm_i_by_wdur[wdur]]
        zp_wdur = numpy.round(zp_wdur, decimals=2)
        zm_wdur = numpy.round(zm_wdur, decimals=2)
        if zp_wdur > 0.0:
            print('z(+)[%r] = %r' % (wdur, zp_wdur))
        if zm_wdur > 0.0:
            print('z(-)[%r] = %r' % (wdur, zm_wdur))


    # recover log prob -- equal to objective without prize term
    log_prob = 0.0
    for i, wdur in enumerate(WDUR):  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        log_prob += primal_soln[i] * log_q[d,u,r]

    for i0, wdur in enumerate(WDUR):  # z^{-}_{w,d,u,r}
        i = i0 + n_WDUR
        _, d, u, r = wdur
        log_prob += primal_soln[i] * log_one_minus_q[d, u, r]

    for i0, dur in enumerate(DUR):
        i = i0 + 2*n_WDUR
        d,u,r = dur
        log_prob += primal_soln[i] * log_prior_q[d,u,r]

    print('log prob of solution: %r' % (log_prob, ))


def main():
    args = parse_args()

    with open(args.input_fns[0], 'rb') as f:
        data = numpy.load(f)

    data = data[-7*52*2:, :]

    def do_solve():
        T, U = data.shape
        log_pr_one_off_explanation = -numpy.log(T) -numpy.log(U)
        large_prize = - log_pr_one_off_explanation
        assert large_prize > 0.0
        prizes = numpy.where(data > 0, large_prize, -large_prize)
        solve(T, U, prizes)

    if args.profile:
        import cProfile, pstats
        with cProfile.Profile() as p:
            try:
                do_solve()
            except KeyboardInterrupt:
                pass
        ps = pstats.Stats(p).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(75)
    else:
        do_solve()



if __name__ == '__main__':
    main()
