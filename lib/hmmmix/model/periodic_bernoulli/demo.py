import argparse
import numpy
from . import solve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
                   help='filename of input binary npy data file')
    p.add_argument('--profile', '-p', action='store_true')
    return p.parse_args()


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