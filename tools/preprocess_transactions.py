import argparse
import numpy
import os.path
import re

_DATESTAMP_DD_MM_YYYY_RE = re.compile("^(\d{2})/(\d{2})/(\d{4})$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
        help='filename of input CSV data')
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
    

def load(fn):
    dtype=[('date', 'datetime64[D]'), ('amount', 'f4')]

    converters = {
        0: parse_date,
    }

    return numpy.loadtxt(
        fname=fn,
        dtype=dtype,
        delimiter=',',
        converters=converters,
        usecols=(0, 1),
        skiprows=1,
        encoding=None,
    )

def main():
    args = parse_args()

    with open(args.input_fns[0], 'r') as f:
        data = load(f)

    print('parsed %d record(s) from file "%s"' % (len(data), args.input_fns[0]))

    with open(args.out, 'wb') as f_out:
        numpy.save(f_out, data, allow_pickle=False)

    print('wrote npy array to "%s"' % (args.out, ))

if __name__ == '__main__':
    main()
