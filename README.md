Ad-hoc Viterbi-like algorithm for recovering approximate MAP estimate of
hidden state of sparse factorial hidden Markov model from search space of
many possible factorial hidden Markov models.

Does not support training.

Bounds on approximation unknown, likely poor.

May be of some use to define sparse subset of component HMMs to form a
factorial HMM if there is an intractably large family of candidate component
HMMs to choose from, and observations are expected to be explained by a
relatively small number of component HMMs.

### Purpose

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

The maximisation is done approximately. It is ad-hoc. Approximation error
is unknown.


### keywords

*   MAP estimate
*   Markov processes, factorial hidden Markov models, HMM
*   modified Viterbi algorithm, dynamic programming
*   combinatorial optimisation, linear programming, column generation


### status

Pre-alpha prototype.

See [TODO](TODO.md) list.


