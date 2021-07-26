## TODO

### Theory

#### Problem Statement

See doc/src/note.tex

#### Factorial hidden Markov models

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

#### Decomposition Approaches

[Martins et. al. 2011][dd-admm-2011] "An Augmented Lagrangian Approach to
Constrained MAP Inference" discusses various approaches to approximate MAP
inference, including linear programming relaxations, message-passing, and dual
decomposition.

#### Benchmark against alternatives

1.  Assess impact of using MAP point estimate versus exact/approx posterior.

1.  Assess impact of using hard constraint vs soft constraints.

1.  What about interior point methods instead of simplex for LP solves.

1.  What about branch and price to suppose integer constraints.

1.  Is column generation or a different decomposition approach viable for
    soft constraints.

### Column generation implementation

#### Prize definition

1.  `lib/hmmix/exact_cover_solver_primal.py`: Definition of prizes from dual
    solution may be incorrect. Figure out exactly what it should be and test
    that implementation is correct. Can sanity test by computing reduced
    costs of variables already in the restricted problem - reduced costs of
    variables that are included in the LP should all be non-positive.

#### Subtleties when using dual solution to select new decision variables

Symptom: convergence through successive restricted master problem seems
slow. Sometimes adding a new column from auxiliary problem does not help
improve the objective value.

My understanding of how dual variables, shadow prices and column generation
works was flawed. Care needs to be taken when interpreting the dual solution
associated with primal problem constraints as a means of defining reduced
costs for new variables to add to the restricted master problem.

A clear reference for understanding these subtleties is the following article
[Jansen, de Jong, Roos, Terlaky (1997)][sensitivity-lp-be-careful]
"Sensitivity analysis in linear programming: just be careful!"

The main insight is to consider a function f that gives the optimal objective
value of the LP, then differentiate that function with respect to the RHS b_j
of a particular constraint j. The optimal objective function f(b_j) is
piecewise linear, so the derivative of f with respect to b_j is constant in
parts and undefined whenever f is at a break point.

The value of an optimal dual solution is used to define shadow prices of
constraints which in some situations correspond to these gradients. But:

-   the interpretation of optimal dual solutions as shadow prices only makes
    sence within a particular allowable range, with respect to the simplex
    solver's current basis status. If you don't know what the allowable range is,
    you don't know how far you can move away from the current optimal
    solution before the relationship no longer applies.

-   a naive belief of column generation obtained from reading brief column
    generation tutorials does not understand that even though you may identify
    a new variable to add to the restricted master problem with negative
    reduced cost, the "negative reduced cost" is only valid in some local
    region about the current optimal solution, and perhaps the maximimum
    step size you can take is actually a _zero length step_ that does not
    improve the objective value at all.
    
-   the value of the dual variable corresponding to a constraint at a break
    point may not be well defined. It should be possible to recover a different
    value for the slope to the left and the right of a break point, but your
    simplex may not help you with that.
    
-   your simplex solver may not tell you what the allowable range associated
    with the each dual solution is, so you have no idea how far you can "step"
    when introducing a new variable with negative reduced cost.

It is unclear what the best way to fix this is. Some ideas:

-   ignore the problem that sometimes we will add a column that has negative
    reduced cost that is valid for only a zero length step. Carefully adjust
    the rest of the column generation implementation to compensate. This would
    mean that if we observe no improvement in the objective function of the
    restricted master problem, we cannot terminate, we need to check if there
    are any new variables that could be added with negative reduced costs.

-   integrate with a simplex algorithm that is able to output some information
    about the allowable range associated with each constraint dual value. Use
    those allowable ranges when searching for attractive variables to Add. E.g.
    the current implementation implicitly assumes that all new variables will
    be able to be used with the same step size, therefore picking the variable
    with the minimal reduced cost is best. But what if there is a different
    variable with a less attractive (yet still negative) reduced cost that
    would accept a much larger step size and hence actually improve the
    restricted master problem objective value.
    
-   abandon simplex, investigate if column generation can be implemented atop
    an interior point solver.
    -   scipy integrates a couple of interior point solvers.
    
-   abandon linear programming, reformulate the problem so the hard constraint
    is soft. Would this lead to a quadratic program? Can column generation or
    some other decomposition method be applied? Would that lead to a more
    robust method?

Brief survey of LP solver support for reliable recovery of dual solution along
with corresponding allowable ranges:

-   Excel's LP solver is mentioned in tutorials as supporting a sensitivity
    analysis report.
-   MOSEK's LP solver has high quality documentation about these subtleties
    and [MOSEK's sensitivity analysis][mosek-sensitivity].
-   GLPK's reference manual documents GLPK's sensitivity report
-   Unclear if COIN-OR CLP API exposes enough information.
-   Unclear if google or-tools GLOP exposes enough information.
-   Unclear if gurobi API exposes enough information.

Note that all of these implementations may output sensitivity reports that
are partly defined by the current state of the simplex algorithm (which is
cheap to compute) and are not complete analyses of the actual LP (which might
be much more expensive to recover).

#### Auxiliary modified Viterbi solver

##### Supporting constraints to ban solutions already in master problem.

The solver needs to be extended to support excluding certain
banned solutions from the search space. This is necessary to prevent it
from re-proposing the same solution again and again. In theory it could
be fine to allow multiple copies of an auxiliary solution to be included
in the master problem, as there is no reason why they necessarily need to
be unique. However, given the implementation of column generation is not
correct, it is far more likely that multiple copies of a solution are being
generated because the dual prices are being naively interpreted as being
valid over ranges where they are not in fact valid, which may cause the
column generation algorithm to keep adding infinite copies of the same
column without making progress.

How to implement this constraint?

Each solution to the auxiliary problem can be considered a path - a finite
sequence that interleaves hidden states with emitted outputs. Assume the
outputs belong to some finite set.

Consider the set of paths equipped with the discrete topology: the Viterbi
algorithm can be modified to search for a maximum value path that is distance
1 from any solution in the set of banned solutions.

This could be achieved by taking the product of the existing dynamic
programming state space with {0, 1} to track the distance that each partial
path is from the set of banned solutions. Assuming queries to the set of
banned solutions were O(1) this would at most double the compute and storage
requirements.

We would need a data structure, holding the set of all banned paths, that
supported an efficient way to query a lower bound `D_B` on the distance of a
partial path from the set of banned paths:

    P : set of all paths

    B \subset P : set of banned paths
    
    p, q : paths in P
    
    p' : the current partial path under consideration (a prefix).
    
    p' ~> q : partial path p' can be extended to completed path q
    
    d(p, q) := 0 if p == q, 1 otherwise. discrete distance metric on paths
    
    d(p, B) := min_{q in B} d(p, q)
    
    D_B(p') := min_{q in P s.t. p' ~> q} d(p, B)

The lower bound `G(p')` can be efficiently computed during dynamic
programming by maintaining a trie representation of the set of paths B that
can be traversed from root node by node. Each partial path p' would track
1 bit `D_B(p')` giving the distance from B. This would need to be stored
in the dynamic programming state space. Each partial path would also store
a pointer into a trie node -- this would be stored in a separate array
externally to the state space, indexed by the state space and time step.
Hidden state transitions and emitted observations both correspond to growing
the path by one node. Each would require a trie query of the form
"advance from the current trie node pointer along an edge labelled by the
hidden state (or observation)". The query would return a new trie node pointer
that would either represent a prefix of one or more banned paths, or a
terminal value indicating the current partial path did not exactly match
any partial path of a banned path in B. If a terminal value was returned,
then we know `G(p')=1`. Once `G(p')=1` it stays fixed at that value for
all possible completions of the path.

When recovering the maximum value path for the solution of the auxiliary
problem, we would need to recover a path ending in one of the dynamic
programming states at the final timestep with maximal value subject to the
constraint that `G(p')=1`.

##### Defects

1.  fix trellis issue so that events at start and/or end times can be
    properly modelled and recovered by `hmmmix/trellis/slowtrellis.py`.

1.  the model defined in `gentrellis.py` does not cleanly correspond to
    a Markov model, as the probability of emitting an event depends on the
    predecessor state, but by the Markov assumption it must only depend on the
    current state. This can be repaired by adjusting how the process is
    modelled: split states that may emit into states "will emit" & "will not
    emit"

##### Getting multiple new columns out of each auxiliary solve

Instead of extracting a single max-value-path from the modified Viterbi
algorithm, it is possible to recover the top k highest value paths. This is
more work, and causes larger restricted master LPs to solve, but may speedup
the overall procedure by reducing the number of column generation iterations.
Some industrial applications of column generation use this trick.
Mentioned in [Per Sjögren's masters thesis][airline-scheduling] on industrial
use of column generation for airline crew scheduling.

```
shortest path   O(E + V log V)              general graph   Dijkstra
shortest path   O(E + V)                    trellis-DAG     trellis

K shortest      O(E + K V log V)            general graph?  modified Dijkstra
K shortest      (E + V log V + K + K V)     general graph?  Eppstein
K shortest      O(E + V log V + K + K V)?   general graph?  Hershberger et al.
```

Current state

```
suppose each trellis costs A
suppose each LP costs B
suppose need N columns to be generated
suppose gen 1 column per iter
need N iters

total cost: N * (A + B)
```

Proposed state with K shortest path generation in auxiliary problem.

Bad working assumption: columns generated in batch will be as "effective" as
those generated one at a time between successive LP solves of the restricted
master problem. This assumption will become increasingly wrong as K becomes
larger vs resolving LP and getting a better estimate of what the true
unrestricted LP duals are. Unsure how to quantify this.

```
suppose generate K columns per iter
need N/K iters
each trellis costs K*A now
total cost: (N/K) * (K*A + B) = N*A + N*B/K
```

Without estimate of drop in effectiveness of columns as K increases, unclear
how to choose K. Can do it empirically.

Current cost measurements of A and B (from profiling):

```
phase                           time (s)

- dual exact cover solve            134.0  # cost "B"
- - scipy.optimize.linprog          117.0
- - - scipy.optimize._linprog_ip    107.0
- - - - scipy.sparse.linalg.dsolve   48.0
- trellis search best path            2.2  # cost "A"
- - libtrellis _kernel                1.0
- - <listcomp to render id string>    1.2 # ha!
total                               140.0
```

### Software Gardening


### Performance

1.  (primal python+mip solver only) Nearly one-third running time is building
    the master problem through python mip API. This is much slower than the
    actual linear solve through CBC. Rewriting this setup with C or Cython or
    using scipy.optimize.linprog may give 100x speedup for this subroutine --
    up to +50% speedup for whole program.

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

1.  Cleanup comments and naming.
    1.  The argument used to justify the derivation in
        `lib/hmmmix/base.py` is unclear, rework to follow the cleaner
        derivation in the docs.
    1.  Standardise notation on what is used in the docs derivation.
        Stop using hats for things that are not estimates.

1.  Refactor `lib/hmmmix/master.py` into main app and master problem library.

1.  Rework the initial feasible solution bootstrapper and auxiliary problem
    solvers so they are dependency injected into the library by the main app,
    and not hardcoded.

1.  Idea: what if replace once-off-event with a Poisson process, parametrised by
    rate, that has probability of emitting 0, 1, 2, .. events each timestep.

1. Grow library of test scenarios from different problem domains.
    1.  Roll handfuls of hidden dice, observe sum. Infer dice.
        1.  standard n sided fair dice
        1.  unfair dice
        1.  deck of n die-face-cards: draw set & discard set. cycle.
        1.  deterministic repeating sequence
        1.  dice that never rolls same face twice
        1.  dice with faces that reduce in value after roll
        

1. Document how to bring up a dev environment.

1. Rig basic CI to build and run test suite.


### See also

1.  [Srihari's lecture notes on HMM extensions][sri] briefly describes
    factorial HMM.

1.  [Schweiger, Erlich & Carmi (2019)][sec2019] published the [factorial_hmm]
    Python package for exact inference on factorial HMMs.

1.  Chiodini has shared an [example on github][lucach_fhmm_bach] that applies
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
[pr-sci-or-list]: https://sci.op-research.narkive.com/xPi96STl/handling-degeneracy-during-column-generation
[degenerate2016]: https://www.sciencedirect.com/science/article/pii/S2192437620301011
[dd-admm-2011]: https://www.cs.cmu.edu/~afm/Home_files/icml2011_main.pdf
[sensitivity-lp-be-careful]: https://scholar.google.com/scholar?cluster=13440092608059508637&hl=en&as_sdt=0,5
[mosek-sensitivity]: https://docs.mosek.com/9.2/pythonapi/sensitivity-shared.html
[airline-scheduling]: http://www.math.chalmers.se/Math/Research/Optimization/reports/masters/PerSjogren-final.pdf