## TODO

### Problem Statement

Given an input sequence of event count data

e^[t, u] := number of events of type u at time t

compute the maximum a posteriori probability (MAP) estimate to find a
hypothesis that fits explains the observed data:

h* = argmax_{h in H} P(H=h | E=e^)

The hypothesis space H is the set of factorial hidden Markov models. Each h in
H specifies:

1.  a subset of one or more component hidden Markov models ; and
2.  for each component model, which hidden state it is in at each time t .

The observed event counts e^[t, u] of each type u at each time t is assumed to
be the sum of event counts generated by one or more hidden Markov models.

### Factorial hidden Markov models

[Ghahramani and Jordan (1997)][gj97] discuss factorial hidden Markov models -- a
generalisation of hidden Markov model where the state space is factored.

They adopt the following notation to define a hidden Markov model:

Let {Y_t} denote a set of observations where 1, ..., T.
Let {S_t} denote a set of hidden state random variables, each taking one of K
discrete values S_t in {1, ..., K}.

The joint probability distribution over observations and states, P({Y_t, S_t}),
is factored by two conditional independence assumptions:

1.  Y_t is independent of all other observations and states given S_t ; and
2.  S_t is independent of S_1, ..., S_t-2 given S_t-1 (first-order Markov
    property).

P({Y_t, S_t}) = P(S_1) P(Y_1|S_1) product_{t=2...T} P(S_t|S_t-1) P(Y_t|S_t)

The factorial hidden Markov model makes the additional assumption that each
state is represented as a collection of M state variables:

S_t = S_t^(1) , ... S_t^(m) , ... S_t^(M)

each of which can take on K^(m) values. The transition model is defined in the
natural way from M component transition models, which are assumed to be
uncoupled:

P(S_t|S_t-1) = product_{m=1...M} P(S_t^(m)|S_t-1^(m))

The observation Y_t at time t may depend upon some or all of the M component
state variables at that time step. Ghahramani and Jordan assume an observation
model of the form

P(Y_t|S_t) ~ Normal(mu_t , C)

where mu_t is a size D observation vector defined by

mu_t = sum_{m=1...M} W^(m) S_t^(m)

each W^(m) is a D x K matrix, and the k-th state variable is represented as a
size K vector with 1 in position k and 0s elsewhere.





### Theory

1.  The argument used to justify the derivation in
    `lib/hmmmix/base.py` is unclear, rework with clearer notation.

1.  Standardise notation. Stop using hats for things that are not estimates.

1.  Assess severity of approximation error introduced by
    replacing integration with maximisation. Is there a different approximation
    that could still be computationally feasible but with less error or better
    worst-case bounds?

    1.  Compare with Gharamani and Jordan's variational approximations
    

### Column generation implementation

1.  `lib/hmmix/exact_cover_solver_primal.py`: Definition of prizes from dual
    solution may be incorrect. Figure out exactly what it should be and test
    that implementation is correct. Can sanity test by computing reduced
    costs of variables already in the restricted problem - reduced costs of
    variables that are included in the LP should all be non-positive.

1.  Convergence through successive restricted master problem seems slow.
    Sometimes adding a new column from auxiliary problem does not help improve
    the objective value. Try to understand what is going on and mitigate it.

    1.  Idea: auxiliary problems potentially degenerate, potentially lots of
        symmetry and many solutions (HMM trajectories) with very similar
        objective function values. Is it possible to define cuts using master
        problem to further constraint auxiliary problem search space?
    
    1.  Possible lead: [google/or-tools strawberry fields example][gorsf] mentions
        adding columns that are zero-length steepest pivot moves during simplex.

1.  Defect: fix lattice issue so that events at start and/or end times can be
    properly modelled and recovered by `hmmmix/lattice/slowlattice.py`.

1.  Defect: the model defined in `genlattice.py` does not cleanly correspond to
    a Markov model, as the probability of emitting an event depends on the
    predecessor state, but by the Markov assumption it must only depend on the
    current state. This can be repaired by adjusting how the process is
    modelled: split states that may emit into states "will emit" & "will not
    emit"


### Software Gardening

1. Grow library of test scenarios from different problem domains.

1. Document how to bring up a dev environment.

1. Rig basic CI to build and run test suite.

### Performance

1.  Around two-thirds of running time is inside `slowlattice`.
    Replacing this with naive C or Cython code may give 250x speedup for this
    subroutine -- up to +200% speedup for whole program.

1.  Nearly one-third running time is building the master problem through python
    mip API. This is much slower than the actual linear solve through CBC.
    Rewriting this setup with C or Cython may give 100x speedup for this
    subroutine -- up to +50% speedup for whole program.

1.  Master problem state is discarded and rebuilt from scratch between
    successive iterations. Yet successive problems will be very similar,
    differing only by a single new decision variable, with one new term in the
    objective function and one new column in the constraint matrix. Also, the
    optimal solution of the previous LP is discarded, yet it will be feasible
    and may still be optimal in the next iteration. Reworking this could
    reduce the amount of book-keeping from O((n+1)*m) to O(m) when adding the
    new variable (column) to the existing problem with n variables and m
    constraints.

### Library cleanup

1.  `lib/hmmix/exact_cover_solver_dual.py` doesn't work. Fix it or delete it.

1.  Refactor `lib/hmmmix/master.py` into main app and master problem library.

1.  Rework the initial feasible solution bootstrapper and auxiliary problem
    solvers so they are dependency injected into the library by the main app,
    and not hardcoded.

1.  Idea: what if replace once-off-event with a Poisson process, parametrised by
    rate, that has probability of emitting 0, 1, 2, .. events each timestep.

### See also

1.  [Srihari's lecture notes on HMM extensions][sri] briefly describes
    factorial HMM.

1.  [Schweiger, Erlich & Carmi (2019)][sec2019] published the [factorial_hmm]
    Python package for exact inference on factorial HMMs.

1.  Chiodini has shared an [example on github](lucach_fhmm_bach) that applies
    factorial HMMs to learn and generate Bach sheet music.

1.  Some practical tips about modelling that may be of use are discussed in the
    [Stan User's Guide][stanug]. See discussion of HMMs, finite mixtures and
    clustering.


[gj97]: https://link.springer.com/content/pdf/10.1023/A:1007425814087.pdf
[gorsf]: https://github.com/google/or-tools/blob/b37d9c786b69128f3505f15beca09e89bf078a89/examples/cpp/strawberry_fields_with_column_generation.cc#L401-L409
[sri]: https://cedar.buffalo.edu/~srihari/CSE574/Chap13/Ch13.3-HMMExtensions.pdf
[sec2019]: https://academic.oup.com/bioinformatics/article/35/12/2162/5184283
[factorial_hmm]: https://github.com/regevs/factorial_hmm
[lucach_fhmm_bach]: https://github.com/lucach/fhmm_bach
[stanug]: https://mc-stan.org/docs/2_27/stan-users-guide/