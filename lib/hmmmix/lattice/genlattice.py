import numpy
import typing


"""
off

on and scheduled in D days (D = 1, ..., 31)

on and overdue by X days and due in D days
    X = 0, ..., 14
    D = 28...31 .... 28-14 ... 31-14 correlated with X

properties of state

prereq          property
none            active(state): true or false
active(state)   scheduled(state): integer in 1,...,31 days until event scheduled
active(state)   overdue(state): true or false
overdue(state)  delay(state): integer in 0,...,14 days overdue
"""

# arbitrary prior: assume 50% survival rate for an active trajectory after 2 years
DAILY_CHURN = 1.0 - numpy.exp(numpy.log(0.5)/(2*365.0))

PROB_ACTIVE_GIVEN_INACTIVE = 1/365.0 # somewhat arbitrary prior value
PROB_ACTIVE_GIVEN_ACTIVE = 1.0 - DAILY_CHURN


MAX_DELAY = 14 # days

PROB_SHUTDOWN = 1. / 26. # assume everything stops one fortnight per year
PROB_NOT_SHUTDOWN = 1.0 - PROB_SHUTDOWN
PROB_SHUTDOWN_EXACTLY_THIS_LONG = PROB_SHUTDOWN * (1./14.)
PROB_WEEKDAY = 5. / 7.
PROB_SATURDAY = 1. / 7.
PROB_SUNDAY = 1. / 7.
PROB_HOLIDAY = 10. / 260.
PROB_NOT_HOLIDAY = 1.0 - PROB_HOLIDAY

def _normalised(p):
    z = sum(p.values())
    return {k: v/z for (k, v) in p.items()}

# this is my somewhat dodgy prior.
PROB_DAYS_DELAYED = _normalised({
    0: PROB_WEEKDAY * PROB_NOT_HOLIDAY * PROB_NOT_SHUTDOWN,
    1: PROB_SUNDAY * PROB_NOT_SHUTDOWN + PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    2: PROB_SATURDAY * PROB_NOT_SHUTDOWN + PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    3: PROB_SATURDAY * PROB_NOT_SHUTDOWN * PROB_HOLIDAY + PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    4: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    5: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    6: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    7: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    8: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    9: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    10: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    11: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    12: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    13: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
    14: PROB_SHUTDOWN_EXACTLY_THIS_LONG,
})



def _further_delay(current_delay):
    # compare prob of it taking exactly this long
    # vs prob of taking some longer time
    d = current_delay
    if d not in PROB_DAYS_DELAYED:
        return 0.0
    enough = PROB_DAYS_DELAYED[d]
    more = 0.0
    d += 1
    while d in PROB_DAYS_DELAYED:
        more += PROB_DAYS_DELAYED[d]
        d += 1
    return more / (enough + more)

# TODO fix divide by zero in log due to computing impossible event
LOGPROB_FURTHER_DELAY_GIVEN_CURRENT_DELAY = {d:numpy.log(_further_delay(d)) for d in PROB_DAYS_DELAYED}
LOGPROB_EMIT_GIVEN_CURRENT_DELAY = {d:numpy.log(1.0 - _further_delay(d)) for d in PROB_DAYS_DELAYED}

LOGPROB_ACTIVE_GIVEN_INACTIVE = numpy.log(PROB_ACTIVE_GIVEN_INACTIVE)
LOGPROB_INACTIVE_GIVEN_INACTIVE = numpy.log(1.0-PROB_ACTIVE_GIVEN_INACTIVE)
LOGPROB_ACTIVE_GIVEN_ACTIVE = numpy.log(PROB_ACTIVE_GIVEN_ACTIVE)
LOGPROB_INACTIVE_GIVEN_ACTIVE = numpy.log(1.0 - PROB_ACTIVE_GIVEN_ACTIVE)


# Stochastic version of monthly calendar. no, the calendar doesnt need to be
# stochastic, but then we need to thread the actual day of the week through
# all the code, and make everything calendar-aware.
PROB_MONTHLY_SCHEDULE_DAYS_BY_PERIOD = {
    28: (0.75/12.),
    29: (0.25/12.),
    30: (4/12.),
    31: (7/12.),
}

LOGPROB_MONTHLY_SCHEDULE_DAYS_BY_PERIOD = {d: numpy.log(pr) for (d, pr) in PROB_MONTHLY_SCHEDULE_DAYS_BY_PERIOD.items()}


class State(typing.NamedTuple):
    active: bool
    scheduled: typing.Optional[int]
    overdue: typing.Optional[bool]
    delay: typing.Optional[int]


class Edge(typing.NamedTuple):
    succ: State
    weight: float
    delta_e: int


def mkstate(active: bool, scheduled: typing.Optional[int]=None, overdue: typing.Optional[bool]=False, delay: typing.Optional[int]=None):
    return State(active=active, scheduled=scheduled, overdue=overdue, delay=delay)


def mkedge(succ: State, weight: float, delta_e: int):
    return Edge(succ=succ, weight=weight, delta_e=delta_e)


def active(state) -> bool:
    return state.active


def scheduled(state) -> int:
    assert state.active
    if state.scheduled:
        assert not state.overdue
    return state.scheduled


def overdue(state) -> bool:
    assert state.active
    if state.overdue:
        assert not state.scheduled
    return state.overdue


def delay(state) -> int:
    assert state.overdue
    return state.delay


def gen_edges(state):
    if not active(state):
        # either continue being inactive
        yield mkedge(succ=state, weight=LOGPROB_INACTIVE_GIVEN_INACTIVE, delta_e=0)
        # or become active scheduled for next timestep
        yield mkedge(succ=mkstate(active=True, overdue=True, delay=0), weight=LOGPROB_ACTIVE_GIVEN_INACTIVE, delta_e=0)
    else:
        # active state is either not overdue or overdue
        if not overdue(state):
            # scheduled state could suddenly become inactive
            yield mkedge(succ=mkstate(active=False), weight=LOGPROB_INACTIVE_GIVEN_ACTIVE, delta_e=0)
            # but if it doesnt
            d = scheduled(state)
            if d == 1:
                # it becomes due tomorrow
                yield mkedge(succ=mkstate(active=True, overdue=True, delay=0), weight=LOGPROB_ACTIVE_GIVEN_ACTIVE, delta_e=0)
            else:
                # or it keeps counting down
                yield mkedge(succ=mkstate(active=True, scheduled=d-1), weight=LOGPROB_ACTIVE_GIVEN_ACTIVE, delta_e=0)
        else:
            # overdue state could suddenly become inactive
            yield mkedge(succ=mkstate(active=False), weight=LOGPROB_INACTIVE_GIVEN_ACTIVE, delta_e=0)
            # but if it doesnt, and remains active
            # it either gets even more overdue without observably doing anything
            dd = delay(state)
            if dd < MAX_DELAY:
                w = LOGPROB_ACTIVE_GIVEN_ACTIVE + LOGPROB_FURTHER_DELAY_GIVEN_CURRENT_DELAY[dd]
                assert numpy.isfinite(w)
                yield mkedge(succ=mkstate(active=True, overdue=True, delay=dd+1), weight=w, delta_e=0)
            # or it emits right now and sets the clock for next schedule
            for d in sorted(LOGPROB_MONTHLY_SCHEDULE_DAYS_BY_PERIOD):
                w = LOGPROB_ACTIVE_GIVEN_ACTIVE + LOGPROB_EMIT_GIVEN_CURRENT_DELAY[dd] + LOGPROB_MONTHLY_SCHEDULE_DAYS_BY_PERIOD[d]
                # adjust when d is scheduled to compensate for any delay we experienced
                # this time around (if any)
                d_adj = d - dd
                # We also emit 1 count worth of event! Observable behaviour!
                yield mkedge(succ=mkstate(active=True, scheduled=d_adj), weight=w, delta_e=1)


def gen_states():
    yield mkstate(active=False)
    for d in range(1, 31 + 1):
        yield mkstate(active=True, scheduled=d)
    for dd in range(0, MAX_DELAY + 1):
        yield mkstate(active=True, overdue=True, delay=dd)


STATES = list(gen_states())

OUTGOING_EDGES_BY_STATE = {s:list(gen_edges(s)) for s in STATES}




# this prior is low information guesswork

def logprob_prior(state):
    w = 0.0
    if active(state):
        w = numpy.log(1.0 / 2.0)
        if overdue(state):
            w += numpy.log(1.0 / 2.0)
            # assume spread nonuniformly according to delay distribution
            dd = delay(state)
            w += numpy.log(PROB_DAYS_DELAYED[dd])
            return w
        else:
            assert scheduled(state)
            w += numpy.log(1.0 / 2.0)
            # assume spread uniformly over possible values of d
            w += numpy.log(1.0 / 31.0)
            return w
    else:
        w += numpy.log(1.0 / 2.0)
        return w


LOGPROB_PRIOR_BY_STATE = {s:logprob_prior(s) for s in STATES}


def probability_conserved(logprobs):
    acc = numpy.sum(numpy.exp(logprobs))
    return numpy.isclose(acc, 1.0)

# sanity check
# prior should be a valid probability distribution
assert probability_conserved(list(LOGPROB_PRIOR_BY_STATE.values())), "prior fails to conserve probability"


def edges_conserve_probability(edges):
    return probability_conserved([e.weight for e in edges])

# sanity check
# fix a state s. the outgoing edges from s should have probability that sums to unity.
for s in STATES:
    assert edges_conserve_probability(OUTGOING_EDGES_BY_STATE[s]), "edges fail to conserve probability; state: " + repr(s)


def prettystate(state):
    if not active(state):
        return 'ZZZ'
    d = scheduled(state)
    if d:
        return 'S%02d' % (d, )
    dd = delay(state)
    return 'D%02d' % (dd,)


if __name__ == '__main__':
    # Display states and transitions if run as a script for some reason.

    for s in STATES:
        print(repr(prettystate(s)))

    print()

    for s in STATES:
        for edge in OUTGOING_EDGES_BY_STATE[s]:
            print('%r\t->\t%r\t%r\t%r' % (prettystate(s), prettystate(edge.succ), edge.weight, edge.delta_e))

    print()
    print(len(STATES))