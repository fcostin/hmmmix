import abc
import typing
import numpy
import numpy.typing as npt

"""
Inputs:

Let t in T be an index ranging over times.
Let u in U be an index ranging over event types.

For each t in T and each u in U, let \pi_{t, u} be a constant real number. Roughly,
\pi_{t, u} is the "prize" awarded for being able to explain one observed count of an
event of type u in U at time t in T.

Auxiliary problem:

Let G be a generator, a process that can generate events. Regard G as a probability
distribution over a set containing individual generator states g.

The auxiliary problem is to search for a generator g* in G and corresponding vector of
non-negative integer observation counts e*_{t, u} indexed over t in T and u in U such that

g*, e* = argmax_{g in G, e in E} sum_{t, u} \pi_{t, u} e*_{t, u} + log P(G=g, E=e)

The master problem will only consider using a solution of the auxiliary problem if the
value of the objective function is strictly positive, i.e. if the contribution from the
former "prize" term outweighs the unlikeliness of the solution as measured by the latter
term that measures the prior log-probability of the joint distribution over G and E.

We generally assume that once the generator state G=g is fixed, the individual
observations in the observation vector e are independent. That is, we assume

P(E=e | G=g) = product_{t, u} P(E_{t, u}=e_{t, u} | G=g)
"""

class AuxiliaryProblem(typing.NamedTuple):
    times: typing.Sequence[int] # list of time indices T. expected to be contiguous, ordered.
    event_types: typing.Sequence[int] # list of event type indices U.

    # prizes is a shape (|T|, |U|) array of real-valued prizes. prizes[t, u] gives
    # the prize for explaining one count of event type u at time t.
    prizes: npt.ArrayLike


class AuxiliarySolution(typing.NamedTuple):
    id: typing.Hashable # identifier for solution
    objective: numpy.float64 # value of objective function
    logprob: numpy.float64 # value of logprob term of objective function

    # e is a shape (|T|, |U|) array of non-negative integers. e[t][u] gives the
    # count of events of type u in U at time t in T supplied by this solution.
    e: npt.ArrayLike


class AuxiliarySolver(metaclass=abc.ABCMeta):

    def solve(self, problem: AuxiliaryProblem) -> typing.Optional[AuxiliarySolution]:
        pass


def is_auxiliary_solution_sane(problem: AuxiliaryProblem, soln: AuxiliarySolution) -> bool:
    # Check that the difference between the claimed objective and logprob
    # is equal to <prizes, e>.

    total_prize = soln.objective - soln.logprob

    acc = 0.0
    for t in problem.times:
        for u in problem.event_types:
            acc += soln.e[t, u] * problem.prizes[t, u]

    return numpy.isclose(total_prize, acc)


"""
### Inputs:

Let t in T be an index ranging over times.
Let u in U be an index ranging over event types.

For each t in T and each u in U, let e^_{t, u} be a constant non-negative integer
that records the number of events of type u in U observed at time t in T.

### Master problem:

Let H be a generator, a process that can generate events. Regard H as a probability
distribution over a set containing individual generator states h.

The master problem is to compute the maximum a posteriori probability (MAP) estimate:

h* = argmax_{h in H} P(H=h | E=e^)

Equivalently, we can rewrite the objective of the argmax by applying Bayes' theorem and
then taking the log (as log is monotonic):

h*  =   argmax_{h in H} P(E=e^ | H=h) P(H=h) / P(E=e^)                  (Bayes Thm)
    =   argmax_{h in H} log P(E=e^ | H=h) + log P(H=h) - log P(E=e^)    (log monotonic)
    =   argmax_{h in H} log P(E=e^ | H=h) + log P(H=h)

The latter term - log P(E=e^) does not depend on h so it may be ignored.

### Relationship to Auxiliary problem:

Let {G_i}_{i in I} denote a set of generators, indexed by I, that each might explain some
or all of the observed events counted in e^ . Let x = (x_i)_{i in I} be a vector of
weights, with each x_i in {0, 1}.

We define the generator H(x) as a function of the weight vector x by:
*   sample an observation vector e_i from each G_i where x_i = 1
*   sum together all the observation vectors e_i to produce e
That is, by e ~ H(x) we mean e_{t, u} = sum_{i in I} x_i e_i{t, u} where e_i ~ G_i .

If each component generator G_i has a prior P(G_i) then we can define a prior on H(x) by

P(H(x)) := product_{i in I} P(G_i)^{x_i}

that is, P(H) is equal to the product of the priors of the component generators that
are selected by the weight vector x.

Now, we want to evaluate log P(E=e^ | H=h(x))

Recall h(x) generates a sample e by sampling e_i from each component generator G_i then
summing the samples. Let us introduce random variables {E_i}_{i in I} to denote the
sample from each component generator.

Recall for any two sufficiently well behaved random variables A and B, by the law of
total probability we have P(A) = sum_{b in B} P(A|B=b) P(B=b) , i.e. P(A) can be recovered
by conditioning on B then integrating out B again. We condition on all of the random
variables {E_i}_{i in I} then integrate them all out again:

    P(E=e^ | H=h(x))
=   sum_{e_1 in E_1} ... sum_{e_m in E_m} P(E=e^ | H=h(x), E_1=e_1, ..., E_m=e_m )
    P(E_m=e_m | H=h(x)) ... P(E_1=e_1 | H=h(x))

Note we have assumed that the active generators (where x_i=1) can be labelled {1, ..., m},
and that the above expression contains m nested integrals.

Consider P(E=e^ | H=h(x), E_1=e_1, ..., E_m=e_m ) . From our definition of H, H emits
a sample by taking the sum of all samples {e_i}_{i in I} from the component generators
{G_i}_{i in H}. So P(E=e^ | H=h(x), E_1=e_1, ..., E_m=e_m ) is either equal to 1 or 0,
and is 1 iff e^ = sum_{i in I} x_i e_i .

This means we can ignore contributions from all combinations of (e_1, ..., e_m)
that do not satisfy e^ = sum_{i in I} x_i e_i .

Now, we make an approximation: Suppose a_1, ..., a_n are arbitrary real numbers. Then
sum_j a_j can be (perhaps poorly) approximated by max_j a_j .

Similarly log(sum_j a_j) can be (perhaps poorly) approximated by
log (max_j a_j) = max_j log (a_j) using the monotonicity of log.

Similarly, we approximate and replace the nested sums (integrals) over (e_1, ..., e_m)
by instead maximising over (e_1, ..., e_m), so

    P(E=e^ | H=h(x))

is approximated by

    max_{e in E'} product_{i in I} P(E_i=e_i | H=h(x))^{x_i}

subject to the constraint
    
    sum_{i in I} x_i e_i = e^
    
where E' is the product space product_{i in I} {E_i} with one copy E_i of E for each
component generator G_i .

The factors P(E_i=e_i | H=h(x)) simplify further, as each E_i is independent of G_j for
j != i once G_i=g_i is known, so P(E_i=e_i | H=h(x)) = P(E_i=e_i | G_i=g_i)

Therefore, the term log P(E=e^ | H=h(x))

can be approximated by

max_{e in E'} sum_{i in I} x_i log P(E_i=e_i | G_i=g_i)

subject to the constraint
    
    sum_{i in I} x_i e_i = e^

This lets as write the following approximation of our MAP estimate:

        max_{h in H} P(H=h | E=e^)
    =   max_{h in H} log P(E=e^ | H=h) + log P(H=h)
    =   max_{x in X} log P(E=e^ | H=h(x)) + log P(H=h(x))
    =   max_{x in X} log P(E=e^ | H=h(x)) + sum_{i in I} x_i log P(G_i)

which is approximated by

max of  sum_{i in I} x_i log P(E_i=e_i | G_i=g_i) + sum_{i in I} x_i log P(G_i=g_i) -- (*)

over
    x in {0, 1}^I
    e = {e_i}_{i in I} in E^I
    g = {g_i}_{i in I} in states(G_1) * ... * states(G_n)

that is, for each i, x_i in {0, 1}
and for each i, e_i is a vector of non-negative event counts indexed by (t, u) in T x U

subject to:
    sum_{i in I} x_i e_i = e^


!!!!! objective isn't linear in x_i . 

!!!!! i should be maximising over all the g_i too. not a fatal flaw, that needs to
    be considered when maximising over H=h . h is a function of the decision variables
    x_i in {0, 1} and and g_i in states(G_i) for each i in I

Can fix up nonlinearity by running the sum over triples (i, j, k) where i indexes
generator components, j indexes generator states and k indexes possible values of e_j

then need to prevent more than one copy of generator i being active at a time

for each i, sum_j sum_k x_{i, j, k} <= 1 binary

that's a number of constraints proportional to I, which is okay.

it could be acceptable to just drop those constraints, which would be equivalent to
permitting each x_i be any negative integer.
"""


class MasterProblem(typing.NamedTuple):
    times: typing.Sequence[int] # list of time indices T. expected to be contiguous, ordered.
    event_types: typing.Sequence[int] # list of event type indices U.

    # e_hat is a shape (|T|, |U|) array of non-negative integers. e[t][u] gives the
    # count of observed events of type u in U at time t in T.
    e_hat: npt.ArrayLike


class WeightedAuxiliarySolution(typing.NamedTuple):
    weight: float # Coefficient in half-open interval (0, 1]. (Zero terms omitted).
    solution: AuxiliarySolution


# a MasterSolution is a sparse linear combination of AuxiliarySolutions
class MasterSolution(typing.NamedTuple):
    objective: numpy.float64 # value of objective function

    components: typing.Sequence[WeightedAuxiliarySolution]


class MasterSolver(metaclass=abc.ABCMeta):

    def solve(self, problem: MasterProblem) -> typing.Optional[MasterSolution]:
        pass