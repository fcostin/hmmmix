import argparse
import collections
import numpy
import os.path
import re

Bucket = collections.namedtuple('Bucket', ['label', 'lower_bound', 'upper_bound'])

_DATESTAMP_DD_MM_YYYY_RE = re.compile("^(\d{2})/(\d{2})/(\d{4})$")

_BUCKETS = [
    Bucket(label='T', lower_bound=0.01, upper_bound=10.00),
    Bucket(label='S', lower_bound=10.0, upper_bound=20.00),
    Bucket(label='M', lower_bound=20.0, upper_bound=75.00),
    Bucket(label='L', lower_bound=75.0, upper_bound=200.00),
    Bucket(label='X', lower_bound=200.0, upper_bound=numpy.inf),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
        help='filename of input CSV data')
    p.add_argument('-c', '--counts', action='store_true',
        help='output event counts grouped by (date, bucket)')
    p.add_argument('-d', '--descr', type=str,
                   help='filter events by description (python regex)')
    p.add_argument('-o', '--out', type=str,
        required=True,
        help='filename of output binary npy data file')
    return p.parse_args()


def parse_date(s):
    # dd/mm/yyyy -> numpy datetime64
    m = _DATESTAMP_DD_MM_YYYY_RE.match(s)
    if m is None:
        raise ValueError(s)
    dd = m.group(1)
    mm = m.group(2)
    yyyy = m.group(3)
    isodate = '%s-%s-%s' % (yyyy, mm, dd)
    return numpy.datetime64(isodate, 'D')


def bucketize_transaction_amount(x):
    assert x < 0.0
    y = -x
    for i, b in enumerate(_BUCKETS):
        if b.lower_bound <= y and y < b.upper_bound:
            return i
    raise ValueError(x)


def parse_bucketized_amount(x):
    return bucketize_transaction_amount(float(x))
    

def load(fn):
    dtype=[('date', 'datetime64[D]'), ('bucket', 'int64'), ('descr', 'U4')]

    converters = {
        0: parse_date,
        1: parse_bucketized_amount,
    }

    return numpy.loadtxt(
        fname=fn,
        dtype=dtype,
        delimiter=',',
        converters=converters,
        usecols=(0, 1, 2),
        skiprows=1,
        encoding=None,
    )

def main():
    args = parse_args()

    with open(args.input_fns[0], 'r') as f:
        data = load(f)

    print('parsed %d record(s) from file "%s"' % (len(data), args.input_fns[0]))

    if args.descr:
        pattern = re.compile(args.descr)
        mask = [pattern.match(x) is not None for x in data['descr']]
        data = data[mask]
        print('after filtering on descr=%r, got %d record(s)' % (args.descr, len(data)))

    # briefly summarise to let user sanity-check
    n_buckets = len(_BUCKETS)
    h = numpy.bincount(data['bucket'], minlength=n_buckets)
    assert len(h) == len(_BUCKETS), len(h)
    for hi, bi in zip(h, _BUCKETS):
        print('bucket "%s"\tlower=%.2f\tupper=%.2f\tsamples=%r' % (bi.label, bi.lower_bound, bi.upper_bound, hi))

    if args.counts:
        print('counting events per (date, bucket) group')
        t_0 = numpy.min(data['date'])
        t_1 = numpy.max(data['date'])

        day = numpy.timedelta64(1, 'D')
        n_days = int(numpy.round(((t_1 - t_0) + day) / day))

        data_out = numpy.zeros(shape=(n_days, n_buckets), dtype='int64')

        for record in data:
            t = int(numpy.round((record['date'] - t_0) / day))
            assert 0 <= t and t < n_days
            u = record['bucket']
            assert 0 <= u and u < n_buckets
            data_out[t, u] += 1
    else:
        print('not counting events')
        data_out = data

    with open(args.out, 'wb') as f_out:
        numpy.save(f_out, data_out, allow_pickle=False)

    print('wrote npy array to "%s"' % (args.out, ))

if __name__ == '__main__':
    main()
