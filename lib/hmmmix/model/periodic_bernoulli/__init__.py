import base64
import numpy
import numpy.typing
import itertools
import typing
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


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

(d) The solution of the relaxed problem is not always integral. We need an
integer solution. The relaxed solution can be naively rounded, but this does
not necessarily recover an optimal integer solution.

(e) It seems faster and more optimal to not use this LP formulation and just
directly solve each decomposed independent sub-problem. Note that each
subproblem is completely trivial to solve optimally with binary variables in
time linearly proportional to |W||D||U||R|, i.e. linear in the number of
decision variables in the above LP formulation.
"""


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

    missing_times: typing.AbstractSet[int]

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

    # elements: t
    missing_times: typing.AbstractSet[int]

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
            missing_times=self.missing_times,
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
            missing_times=self.missing_times,
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


def align_to_period_with_padding(period, n_times, n_event_types, prizes):
    # The formulation wants the period to cleanly divide into the number of
    # timesteps. So if things are misaligned, pad extra times marked as missing
    # data to fix things up. See usages of missing_times in the formulation.
    extra = n_times % period
    if extra != 0:
        padding = period - extra
        assert 0 < padding and padding < period
        n_times_prime = n_times + padding
        assert n_times_prime % period == 0
        prizes_prime = numpy.zeros(shape=(n_times_prime, n_event_types), dtype=prizes.dtype)
        prizes_prime[:n_times, :] = prizes
        missing_times = {(n_times+i) for i in range(padding)}
        return period, n_times_prime, n_event_types, prizes_prime, missing_times
    else:
        return period, n_times, n_event_types, prizes, set()


def solve(n_times, n_event_types, prizes, decompose, period: int, n_R: int=4, verbose=False):
    solns = list(gen_solns(
        n_times=n_times,
        n_event_types=n_event_types,
        prizes=prizes,
        decompose=decompose,
        period=period,
        n_R=n_R,
        verbose=verbose,
    ))
    if not solns:
        return None
    if len(solns) == 1:
        return solns[0]

    events = numpy.zeros(shape=prizes.shape, dtype=numpy.int64)

    agg_soln = {
        'id': '',
        'obj': 0.0,
        'log_prob': 0.0,
        'events': events,
    }
    for soln in solns:
        agg_soln['id'] += soln['id'] + '|'
        agg_soln['obj'] += soln['obj']
        agg_soln['log_prob'] += soln['log_prob']
        agg_soln['events'] += soln['events']
    return agg_soln


def gen_solns(n_times, n_event_types, prizes, decompose, period: int, n_R: int=4, verbose=False):

    period, n_times, n_event_types, prizes, missing_times = align_to_period_with_padding(period, n_times, n_event_types, prizes)

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
        missing_times=missing_times,
    )

    n_missing = len(missing_times)
    assert missing_times == set(range(n_times-n_missing, n_times))

    if decompose:
        for d in D:
            for u in U:
                subspec = spec.restrict(D=[d], U=[u])
                subproblem = subspec.make_problem()

                compare_methods = False
                if compare_methods:
                    subsolution = _solve_lp(subproblem, verbose=verbose)
                    subsolution_shadow = _bruteforce_solve(subproblem, verbose=verbose)

                    tol = 1.0e-4
                    if abs(subsolution['rounding_error']) > tol:
                        assert subsolution['obj'] <= subsolution_shadow['obj'] + tol, (subsolution, subsolution_shadow)
                        assert subsolution['log_prob'] <= subsolution_shadow['log_prob'] + tol
                    else:
                        assert subsolution['id'] == subsolution_shadow['id']
                        assert numpy.all(subsolution['events']==subsolution_shadow['events'])
                        assert numpy.isclose(subsolution['obj'], subsolution_shadow['obj'], atol=tol), (subsolution, subsolution_shadow)
                        assert numpy.isclose(subsolution['log_prob'], subsolution_shadow['log_prob'], atol=tol)

                subsolution = _bruteforce_solve(subproblem, verbose=verbose)

                if n_missing:  # undo fake timestamp padding
                    subsolution['events'] = subsolution['events'][:-n_missing, :]
                yield subsolution
    else:
        problem = spec.make_problem()
        solution = _bruteforce_solve(problem, verbose=verbose)
        if n_missing:  # undo fake timestamp padding
            solution['events'] = solution['events'][:-n_missing, :]
        yield solution


def _solve_lp(p: Problem, verbose: bool):
    """
    DEPRECATED:
    *   much slower than the direct bruteforce solve
    *   relaxed solution is sometimes non-integer and less optimal than
        real integer bruteforce'd solution.
    """
    # Assemble decision variables and objective terms

    obj_coeffs = numpy.empty(dtype=numpy.float64, shape=(p.n,))
    bounds = [None] * p.n

    zp_i_by_wdur = {}
    zm_i_by_wdur = {}
    y_i_by_dur = {}

    # Support for problems with missing observations at certain times
    # is expressed through the missing_times set. The current implementation
    # just zeros out the corresponding terms in the objective function. We
    # need this to handle problems where the range of time steps is not an
    # integer multiple of the period.
    #
    # Upsides:
    #   * avoids changing indexing convention
    #   * relatively simple to implement
    # Downsides:
    #   * pollutes formulation with useless free variables


    for i, wdur in enumerate(p.WDUR):  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        t = w * p.period + d
        if t in p.missing_times:
            obj_coeffs[i] = 0.0
        else:
            obj_coeffs[i] = p.log_q[d,u,r] + p.prizes[t, u]
        bounds[i] = (0.0, 1.0)
        zp_i_by_wdur[wdur] = i

    for i0, wdur in enumerate(p.WDUR):  # z^{-}_{w,d,u,r}
        i = i0 + p.n_WDUR
        w, d, u, r = wdur
        t = w * p.period + d
        if t in p.missing_times:
            obj_coeffs[i] = 0.0
        else:
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
    # need to flip sign for min problem back to max problem.
    objective_value = -1.0 * res['fun']

    # recover relaxed solution
    primal_soln = res['x']

    y_by_dur = {dur:primal_soln[y_i_by_dur[dur]] for dur in p.DUR}
    zp_by_wdur = {wdur: primal_soln[zp_i_by_wdur[wdur]] for wdur in p.WDUR}
    zm_by_wdur = {wdur: primal_soln[zm_i_by_wdur[wdur]] for wdur in p.WDUR}

    # Coerce relaxed solution into integer solution.
    # Very often (but not always) the relaxed solution is integral anyway.

    acc_rounding_error = 0.0

    for du in p.DU:
        d,u = du
        best_r = p.R[0]
        best = 0.0
        for r in p.R:
            v = y_by_dur[(d,u,r)]
            if v > best:
                best = v
                best_r = r
        for r in p.R:
            acc_rounding_error += abs(1.0 - best)
            y_by_dur[(d, u, r)] = int(r == best_r)

    for wdur in p.WDUR:
        w,d,u,r = wdur
        if not y_by_dur[(d, u, r)]:
            zp_by_wdur[wdur] = 0
        else:
            integer_zp_wdur = int(numpy.round(zp_by_wdur[wdur]))
            rounding_error = integer_zp_wdur - zp_by_wdur[wdur]
            acc_rounding_error += abs(rounding_error)
            zp_by_wdur[wdur] = integer_zp_wdur

    for wdur in p.WDUR:
        w, d, u, r = wdur
        if not y_by_dur[(d, u, r)]:
            zm_by_wdur[wdur] = 0
        else:
            zm_by_wdur[wdur] = 1 - zp_by_wdur[wdur]

    # We don't care about the emitted value at missing timesteps (if any).
    # Zero them out so they don't cause different idstrings.

    for t in p.missing_times:
        w = t // p.period
        d = t % p.period
        for u in p.U:
            for r in p.R:
                wdur = (w,d,u,r)
                zp_by_wdur[wdur] = 0
                zm_by_wdur[wdur] = 0
    soln_id = idstring_encode_soln(y_by_dur, zp_by_wdur, p.W)

    # We need to recompute the objective now that we forced an integer
    # solution. We also recover the log prob -- equal to objective without
    # prize term.

    int_obj = 0.0
    int_log_prob = 0.0

    # TODO FIXME: janky, we're defining the objective in two places.
    for wdur in p.WDUR:  # z^{+}_{w,d,u,r}
        w, d, u, r = wdur
        t = w * p.period + d
        if t in p.missing_times:
            continue
        int_obj += zp_by_wdur[wdur] * (p.log_q[d,u,r] + p.prizes[t, u])
        int_log_prob += zp_by_wdur[wdur] * p.log_q[d, u, r]

    for wdur in p.WDUR:  # z^{-}_{w,d,u,r}
        w, d, u, r = wdur
        t = w * p.period + d
        if t in p.missing_times:
            continue
        int_obj += zm_by_wdur[wdur] * p.log_one_minus_q[d, u, r]
        int_log_prob += zm_by_wdur[wdur] * p.log_one_minus_q[d, u, r]

    for dur in p.DUR:
        int_obj += y_by_dur[dur] * p.log_prior_q[dur]
        int_log_prob += y_by_dur[dur] * p.log_prior_q[dur]


    events = numpy.zeros(shape=p.prizes.shape, dtype=numpy.int64)
    for dur in sorted(y_by_dur):
        d, u, r = dur
        y = asbit(y_by_dur[dur])
        if not asbit(y):
            continue
        for w in p.W:
            t = w*p.period + d
            events[t, u] = asbit(zp_by_wdur[(w, d, u, r)])

    return {
        'id': soln_id,
        'obj': int_obj,
        'log_prob': int_log_prob,
        'events': events,
        'rounding_error': acc_rounding_error,
    }



def _bruteforce_solve(p: Problem, verbose: bool):
    # Brute force solve
    #
    # Fix d in D and u in U
    # For each r in R, the problem is trivial:
    # 1. assume y_{d,u,r}=1
    #       obj_r = log_prior_q[d,u,r]
    # 2. for each w in W:
    #       obj_r += max(log_q[d,u,r] + prizes[t, u], log_one_minus_q[d,u,r])
    #       Set z^{+}_{w,d,u,r} to 1 or 0 based on which side of max is maximal.
    #
    # Then compare all obj_r . The r=r* with maximal obj_r is the best one
    # to pick. Zero all variables y_{d,d,r} and zp_{w,d,u,r} for all r != r*.
    #
    #    If obj_r > 0, keep y_{d,u,r} and z^{+}_{w,d,u,r} as set.
    #    Otherwise, set them all to zero.
    #
    # This can all be done in O(|D|x|U|x|R|).
    #
    # We could do something cleverer than this if |R| was large, e.g. define
    # some thing to solve for continuous probability q in (0, 1) instead of
    # sweeping over finite R. But in practice we see |R|=4 works fairly well.

    y_by_dur = {dur: 0 for dur in p.DUR}
    zp_by_wdur = {wdur: 0 for wdur in p.WDUR}

    acc_obj = 0.0
    acc_log_prob = 0.0
    for du in p.DU:
        d,u = du

        best_obj_r = -numpy.inf
        best_r = None
        best_logprob_r = -numpy.inf

        for r in p.R:
            y_by_dur[(d, u, r)] = 1
            prior_r = p.log_prior_q[d,u,r]
            obj_r = prior_r
            logprob_r = prior_r
            for w in p.W:
                # Support for problems with missing observations at certain
                # times is expressed through the missing_times set. The current
                # implementation just zeros out the corresponding terms in the
                # objective function. We need this to handle problems where the
                # range of time steps is not an integer multiple of the period.
                t = w * p.period + d
                if t in p.missing_times:
                    continue
                logprob_emit = p.log_q[d,u,r]
                logprob_skip = p.log_one_minus_q[d,u,r]
                value_emit = logprob_emit + p.prizes[t, u]
                value_skip = logprob_skip
                if value_emit > value_skip:
                    obj_r += value_emit
                    logprob_r += logprob_emit
                    zp_by_wdur[(w,d,u,r)] = 1
                else:
                    obj_r += value_skip
                    logprob_r += logprob_skip

            if obj_r > best_obj_r:
                best_obj_r = obj_r
                best_r = r
                best_logprob_r = logprob_r

        # select argmax_r and turn other values of r off
        acc_obj += best_obj_r
        acc_log_prob += best_logprob_r

        for r in p.R:
            if r == best_r:
                continue
            # This is an inferior choice of r. zero all decision vars.
            y_by_dur[(d, u, r)] = 0
            for w in p.W:
                zp_by_wdur[(w, d, u, r)] = 0

    events = numpy.zeros(shape=p.prizes.shape, dtype=numpy.int64)
    for dur in sorted(y_by_dur):
        d, u, r = dur
        y = y_by_dur[dur]
        if not asbit(y):
            continue
        for w in p.W:
            t = w*p.period + d
            events[t, u] = asbit(zp_by_wdur[(w, d, u, r)])

    soln_id = idstring_encode_soln(y_by_dur, zp_by_wdur, p.W)

    return {
        'id': soln_id,
        'obj': acc_obj,
        'log_prob': acc_log_prob,
        'events': events,
    }

