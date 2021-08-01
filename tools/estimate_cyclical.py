import argparse
import base64
import numpy
import numpy.typing
import itertools
import typing
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
                   help='filename of input binary npy data file')
    p.add_argument('--profile', '-p', action='store_true')
    return p.parse_args()


def b64encode_binvars(binvars: numpy.typing.NDArray[numpy.uint8]) -> str:
    return str(base64.b64encode(numpy.packbits(binvars)), 'utf-8')


def b64decode_binvars(s: str) -> numpy.typing.NDArray[numpy.uint8]:
    return numpy.unpackbits(numpy.asarray((map(ord, base64.b64decode(s))), dtype=numpy.uint8))


def asbit(x, tol=2.0e-2):
    if abs(x) <= tol:
        return 0
    if abs(1-x) <= tol:
        return 1
    raise ValueError(asbit)


def idstring_encode_soln(y_by_dur, zp_by_wdur, W) -> str:
    result = ''
    for dur in sorted(y_by_dur):
        d,u,r = dur
        y = asbit(y_by_dur[dur])
        if not asbit(y):
            continue
        event_bits = numpy.asarray([asbit(zp_by_wdur[(w, d, u, r)]) for w in W], dtype=numpy.uint8)
        # Note that none of the chars "(,):;" are used in base64 encoding
        # so if we really want to, we can decode this id string to recover the parts
        # ref: https://datatracker.ietf.org/doc/html/rfc3548.html
        result += '%r:%s;' % (dur, b64encode_binvars(event_bits))
    return result


def idstring_decode_soln(s: str):
    raise NotImplementedError()


class Problem(typing.NamedTuple):
    period: int

    # indices: [t, u]
    prizes: numpy.typing.NDArray[numpy.float64]

    # indices: [d,u,r]
    log_q: numpy.typing.NDArray[numpy.float64]
    log_one_minus_q: numpy.typing.NDArray[numpy.float64]
    log_prior_q: numpy.typing.NDArray[numpy.float64]

    W: typing.Sequence[typing.Tuple[int]]
    D: typing.Sequence[typing.Tuple[int]]
    U: typing.Sequence[typing.Tuple[int]]
    R: typing.Sequence[typing.Tuple[int]]

    WDUR: typing.Sequence[typing.Tuple[int, int, int, int]]
    DUR: typing.Sequence[typing.Tuple[int, int, int]]
    WDU: typing.Sequence[typing.Tuple[int, int, int]]
    DU: typing.Sequence[typing.Tuple[int, int]]

    n_WDUR: int
    n_DUR: int
    n_WDU: int
    n_DU: int
    n_R: int

    n: int  # number of decision variables


class ProblemSpec(typing.NamedTuple):
    period: int

    # indices: [t, u]
    prizes: numpy.typing.NDArray[numpy.float64]

    # indices: [d,u,r]
    log_q: numpy.typing.NDArray[numpy.float64]
    log_one_minus_q: numpy.typing.NDArray[numpy.float64]
    log_prior_q: numpy.typing.NDArray[numpy.float64]

    W: typing.Sequence[typing.Tuple[int]]
    D: typing.Sequence[typing.Tuple[int]]
    U: typing.Sequence[typing.Tuple[int]]
    R: typing.Sequence[typing.Tuple[int]]

    def restrict(self, D=None, U=None):
        if D is None:
            restricted_D = list(self.D)
        else:
            restricted_D = list(D)
        if U is None:
            restricted_U = list(self.U)
        else:
            restricted_U = list(U)

        assert set(restricted_D) <= set(self.D)
        assert set(restricted_U) <= set(self.U)

        return ProblemSpec(
            period=self.period,
            prizes=self.prizes,
            log_q=self.log_q,
            log_one_minus_q=self.log_one_minus_q,
            log_prior_q=self.log_prior_q,
            W=self.W,
            D=restricted_D,
            U=restricted_U,
            R=self.R,
        )

    def make_problem(self) -> Problem:
        WDUR = list(itertools.product(self.W, self.D, self.U, self.R))
        DUR = list(itertools.product(self.D, self.U, self.R))
        WDU = list(itertools.product(self.W, self.D, self.U))
        DU = list(itertools.product(self.D, self.U))

        n_WDUR = len(WDUR)
        n_DUR = len(DUR)
        n_WDU = len(WDU)
        n_DU = len(DU)
        n_R = len(self.R)

        n = 2*n_WDUR + n_DUR

        return Problem(
            period=self.period,
            prizes=self.prizes,
            log_q=self.log_q,
            log_one_minus_q=self.log_one_minus_q,
            log_prior_q=self.log_prior_q,
            W=self.W,
            D=self.D,
            U=self.U,
            R=self.R,
            WDUR=WDUR,
            DUR=DUR,
            WDU=WDU,
            DU=DU,
            n_WDUR=n_WDUR,
            n_DUR=n_DUR,
            n_WDU=n_WDU,
            n_DU=n_DU,
            n_R=n_R,
            n=n,
        )


def align_to_period(period, n_times, n_event_types, prizes):
    # TODO FIXME this throws away part of the problem.
    offcut = n_times % period
    if offcut != 0:
        n_times -= offcut
        prizes = prizes[:-offcut, :]

    n_periods = int(numpy.ceil(n_times / period))
    assert n_times == period * n_periods

    return period, n_times, n_event_types, prizes


def solve(n_times, n_event_types, prizes, decompose, period: int, n_R: int=4, verbose=False):

    period, n_times, n_event_types, prizes = align_to_period(period, n_times, n_event_types, prizes)

    n_periods = n_times // period

    W = numpy.arange(n_periods)
    D = numpy.arange(period)
    U = numpy.arange(n_event_types)
    R = numpy.arange(n_R)

    approx_probabilities = (0.5 ** numpy.arange(1, n_R+1))

    log_q = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_q[:, :, :] = numpy.log(approx_probabilities)[numpy.newaxis, numpy.newaxis, :]
    log_one_minus_q = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_one_minus_q[:, :, :] = numpy.log(1.0 - approx_probabilities)[numpy.newaxis, numpy.newaxis, :]

    log_prior_q  = numpy.empty(shape=(period, n_event_types, n_R), dtype=numpy.float64)
    log_prior_q[:, :, :] = numpy.log(1.0 / n_R) # choose one of n_r choices, uniform prior.

    spec = ProblemSpec(
        period=period,
        prizes=prizes,
        W=W,
        D=D,
        U=U,
        R=R,
        log_q=log_q,
        log_one_minus_q=log_one_minus_q,
        log_prior_q=log_prior_q,

    )

    agg_solution = {
        'id': '',
        'obj': 0.0,
        'log_prob': 0.0,
    }

    if decompose:
        for d in D:
            for u in U:
                subspec = spec.restrict(D=[d], U=[u])
                subproblem = subspec.make_problem()
                subsolution = _solve(subproblem, verbose=verbose)

                agg_solution['id'] += subsolution['id'] + '|'
                agg_solution['obj'] += subsolution['obj']
                agg_solution['log_prob'] += subsolution['log_prob']
    else:
        problem = spec.make_problem()
        agg_solution = _solve(problem, verbose=verbose)

    return agg_solution


def _solve(p: Problem, verbose: bool):
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

    (a) Ideally we would optimize with decision variables q_{d, u} in [0, 1],
    but this introduces nonlinear terms of the form z log(q) into the objective
    function. To avoid this we approximate each q_{d, u} by one of R
    prespecified constant probability values. We break the nonlinearity by
    making R copies of each original decision variable and adding additional
    variables and constraints to encourage a single approximation r in R to
    be chosen for each q_{d, u}.

    (b) The variable z^{-}_{w,d,u,r} is used to encode "0 events of type u were
    observed at time (w, d)". It is necessary to use this instead of
    1 - z^{+}_{w,d,u,r} so we do not penalise the objective when terms for
    values of r that have not been "chosen" via y_{d,u,r} are forced to vanish.

    (c) The problem as formulated is much simpler: it decomposes into
    completely independent sub-problems for each combination of D and U.
    As formulated, there is no coupling between these sub problems. We are
    fitting |D| x |U| different independent models in parallel.
    """

    # Assemble decision variables and objective terms

    obj_coeffs = numpy.empty(dtype=numpy.float64, shape=(p.n,))
    bounds = [None] * p.n

    zp_i_by_wdur = {}
    zm_i_by_wdur = {}
    y_i_by_dur = {}

    for i, wdur in enumerate(p.WDUR):  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        t = w * p.period + d
        obj_coeffs[i] = p.log_q[d,u,r] + p.prizes[t, u]
        bounds[i] = (0.0, 1.0)
        zp_i_by_wdur[wdur] = i

    for i0, wdur in enumerate(p.WDUR):  # z^{-}_{w,d,u,r}
        i = i0 + p.n_WDUR
        _, d, u, r = wdur
        obj_coeffs[i] = p.log_one_minus_q[d,u,r]
        bounds[i] = (0.0, 1.0)
        zm_i_by_wdur[wdur] = i

    for i0, dur in enumerate(p.DUR):
        i = i0 + 2*p.n_WDUR
        d,u,r = dur
        obj_coeffs[i] = p.log_prior_q[d,u,r]
        bounds[i] = (0.0, 1.0)
        y_i_by_dur[dur] = i

    assert numpy.all(numpy.isfinite(obj_coeffs))

    # scipy.optimize.linprog minimizes! flip all the signs!
    obj_coeffs *= -1.0

    # Assemble inequality constraints

    n_entries = (p.n_WDU * p.n_R) + (p.n_DU * p.n_R)
    con_data = numpy.empty(shape=(n_entries,), dtype=numpy.float64)
    col_indices = numpy.empty(shape=(n_entries,), dtype=numpy.int64)
    row_indices = numpy.empty(shape=(n_entries,), dtype=numpy.int64)

    entry = 0
    con = 0

    # constraint: for all {w,d,u}:  sum_{r} z^{+}_{w,d,u,r}  <=  1
    # -- n_WDU constraints, each with n_R LHS entries
    for j, wdu in enumerate(p.WDU):
        w,d,u = wdu
        con_data[entry:entry+p.n_R] = 1.0
        for r in range(p.n_R):
            assert zp_i_by_wdur[(w,d,u,r)] < p.n
            col_indices[entry+r] = zp_i_by_wdur[(w,d,u,r)]
        row_indices[entry:entry + p.n_R] = con + j
        entry += p.n_R
    con += p.n_WDU

    # constraint: for all {d,u}:  sum_{r} y_{d,u,r}  <=  1
    # -- n_DU constraints, each with n_R LHS entries
    for j, du in enumerate(p.DU):
        d, u = du
        con_data[entry:entry + p.n_R] = 1.0
        for r in range(p.n_R):
            assert y_i_by_dur[(d, u, r)] < p.n
            col_indices[entry + r] = y_i_by_dur[(d, u, r)]
        row_indices[entry:entry + p.n_R] = con + j
        entry += p.n_R
    con += p.n_DU

    assert entry == n_entries

    a_ub_matrix = coo_matrix(
        (con_data, (row_indices, col_indices)),
        shape=(con, p.n),
        dtype=numpy.float64,
    )

    b_ub_vector = numpy.ones(shape=(con, ), dtype=numpy.float64)

    # Assemble equality constraints

    n_eq_entries = (p.n_WDUR * 3)
    con_eq_data = numpy.empty(shape=(n_eq_entries,), dtype=numpy.float64)
    col_eq_indices = numpy.empty(shape=(n_eq_entries,), dtype=numpy.int64)
    row_eq_indices = numpy.empty(shape=(n_eq_entries,), dtype=numpy.int64)

    entry = 0
    con = 0

    # constraint: for all {w,d,u,r}: z^{+}_{w,d,u,r} + z^{-}_{w,d,u,r} =  y_{d,u,r}
    # -- n_WDUR constraints, each with 3 LHS entries
    for j, wdur in enumerate(p.WDUR):
        w,d,u,r = wdur
        con_eq_data[entry] = 1.0
        con_eq_data[entry+1] = 1.0
        con_eq_data[entry+2] = -1.0
        col_eq_indices[entry] = zp_i_by_wdur[wdur]
        col_eq_indices[entry+1] = zm_i_by_wdur[wdur]
        col_eq_indices[entry+2] = y_i_by_dur[(d,u,r)]
        row_eq_indices[entry:entry+3] = con + j
        entry += 3
    con += p.n_WDUR

    assert entry == n_eq_entries

    a_eq_matrix = coo_matrix(
        (con_eq_data, (row_eq_indices, col_eq_indices)),
        shape=(con, p.n),
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
            'disp': verbose,
            'presolve': False, # Presolve gives 12x slowdown. Disable it!
            'rr': False, # note: Presolve rr is main cause of slowdown.
            'tol': 1.0e-7,
            'sparse': True,
        }
    )

    success = res['success']
    if not success:
        status = res['status']
        message = res['message']
        if verbose:
            print('failed to find an optimal solution: status=%r; message=%r' % (
        status, message))
        return None

    # recover objective
    objective_value = res['fun']

    # recover solution
    primal_soln = res['x']

    spam_solution = False
    if spam_solution:
        for dur in p.DUR:
            y_dur = primal_soln[y_i_by_dur[dur]]
            y_dur = numpy.round(y_dur, decimals=2)
            if y_dur > 0.0:
                print('y[%r] = %r' % (dur, y_dur))

        for wdur in p.WDUR:
            zp_wdur = primal_soln[zp_i_by_wdur[wdur]]
            zm_wdur = primal_soln[zm_i_by_wdur[wdur]]
            zp_wdur = numpy.round(zp_wdur, decimals=2)
            zm_wdur = numpy.round(zm_wdur, decimals=2)
            if zp_wdur > 0.0:
                print('z(+)[%r] = %r' % (wdur, zp_wdur))
            if zm_wdur > 0.0:
                print('z(-)[%r] = %r' % (wdur, zm_wdur))

    y_by_dur = {dur:primal_soln[y_i_by_dur[dur]] for dur in p.DUR}
    zp_by_wdur = {wdur: primal_soln[zp_i_by_wdur[wdur]] for wdur in p.WDUR}
    soln_id = idstring_encode_soln(y_by_dur, zp_by_wdur, p.W)

    # recover log prob -- equal to objective without prize term
    log_prob = 0.0
    for i, wdur in enumerate(p.WDUR):  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        log_prob += primal_soln[i] * p.log_q[d,u,r]

    for i0, wdur in enumerate(p.WDUR):  # z^{-}_{w,d,u,r}
        i = i0 + p.n_WDUR
        _, d, u, r = wdur
        log_prob += primal_soln[i] * p.log_one_minus_q[d, u, r]

    for i0, dur in enumerate(p.DUR):
        i = i0 + 2*p.n_WDUR
        d,u,r = dur
        log_prob += primal_soln[i] * p.log_prior_q[d,u,r]

    return {
        'id': soln_id,
        'obj': objective_value,
        'log_prob': log_prob,
    }


def main():
    args = parse_args()

    with open(args.input_fns[0], 'rb') as f:
        data = numpy.load(f)

    def do_solve():
        T, U = data.shape
        log_pr_one_off_explanation = -numpy.log(T) -numpy.log(U)
        large_prize = - log_pr_one_off_explanation
        assert large_prize > 0.0
        prizes = numpy.where(data > 0, large_prize, -large_prize)
        # Sweep over a few different choices for period.
        # BEWARE results will not be comparable until align_to_period &
        # formulation is fixed to handle n_times that is not an integer
        # multiple of the period.
        periods = [6, 7, 8, 14, 21, 28, 30, 31]
        for period in periods:
            soln = solve(
                n_times=T,
                n_event_types=U,
                prizes=prizes,
                decompose=True,
                period=period,
                verbose=False,
            )
            summary = 'period %d\tobj %.1f\tlog_prob %.1f'
            print(summary % (period, soln['obj'], soln['log_prob']))

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
