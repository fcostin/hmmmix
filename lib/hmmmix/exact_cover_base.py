import abc
import typing
import numpy
import numpy.typing as npt


class CandidateSet(typing.NamedTuple):
    # cost of including this candidate set in solution
    cost: numpy.float64

    # e is a shape (|T|, |U|) array of non-negative integers. e[t][u] gives the
    # count of observed events of type u in U at time t in T.
    e: npt.ArrayLike


class ExactCoverResourcePricingProblem(typing.NamedTuple):
    times: typing.Sequence[int] # list of time indices T. expected to be contiguous, ordered.
    event_types: typing.Sequence[int] # list of event type indices U.

    # e_hat is a shape (|T|, |U|) array of non-negative integers. e[t][u] gives the
    # count of observed events of type u in U at time t in T.
    e_hat: npt.ArrayLike

    # z[i]. each has a cost and a description of which resources it supplies e[t][u] at
    # each time t of each time u
    z_by_i: typing.Dict[str, CandidateSet]

    # upper bound on how many copies of set z[i] we are allowed. usually 1
    ub_by_i: typing.Dict[str, float]

    # (t, u) -> set of i such that z[i].e[(t, u)] > 0
    # redundant with information in z_by_i but used to speedup model construction
    i_with_support_t_u: typing.DefaultDict[typing.Tuple[int, int], typing.Set[str]]


class ExactCoverResourcePricingSolution(typing.NamedTuple):
    objective: numpy.float64 # value of objective function

    # Prizes is a shape (|T|, |U|) array of real-valued prizes. prizes[t, u] gives
    # the prize for explaining one count of event type u at time t.
    # Note: each element prizes[(t, u)] is defined as a linear combination of elements
    # of -A^T y* for supply constraints at (t, u), where y* is an optimal solution
    # of the dual to the relaxed exact cover problem.
    prizes: npt.ArrayLike

    # z is sparse vector of weights in cover indexed by i
    z: typing.Dict[str, float]


class ExactCoverResourcePricingSolver(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def solve(self, problem: ExactCoverResourcePricingProblem) -> typing.Optional[ExactCoverResourcePricingSolution]:
        pass