import argparse
import numpy

from . import master
from . import base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
        help='filename of input event count data (binary npy format)')
    p.add_argument('--profile', '-p', action='store_true')
    return p.parse_args()


def do_solve(args):
    with open(args.input_fns[0], 'rb') as f:
        e_hat = numpy.load(f)

    # mip library goes completely bananas if given unsigned integers
    e_hat = numpy.asarray(e_hat, dtype=numpy.int64)
    e_hat = e_hat[-300:, -3:]

    n_time, n_type = e_hat.shape

    print('note: there are %d total events in e_hat' % (e_hat.sum(), ))

    T = numpy.arange(n_time)
    U = numpy.arange(n_type)

    print('n_time: %d' % (n_time, ))
    print('n_type: %d' % (n_type, ))

    s = master.RelaxedMasterSolver()

    problem = base.MasterProblem(
        times=T,
        event_types=U,
        e_hat=e_hat,
    )

    soln = s.solve(problem)
    if soln is None:
        return

    print('')
    print('solution:')
    for i in sorted(soln.z):
        weight = soln.z[i]
        print('%.4f\t%s' % (weight, i))
    return

def main():
    args = parse_args()

    if args.profile:
        import cProfile, pstats
        with cProfile.Profile() as p:
            try:
                do_solve(args)
            except KeyboardInterrupt:
                pass
        ps = pstats.Stats(p).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(50)
    else:
        do_solve(args)


if __name__ == '__main__':
    main()