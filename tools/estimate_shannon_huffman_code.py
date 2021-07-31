import argparse
import numpy

"""
Compute an upper bound on minimum description length of given data sample
by assuming samples were taken iid from a categorical distribution.
Decides how many bits to spend to encode categorical distribution weights
via minimum description length.

Provides a dead simple generative model that can be used to benchmark:
*   data compression
*   predictive accuracy

References:
1.  MacKay's book Information Theory, Inference & Learning Algorithms.
2.  Blier, Ollivier (2018) The Description Length of Deep Learning Models
"""

MIN_BINS = 1
MAX_BINS = 8
MIN_BITS_PER_BIN = 2
MAX_BITS_PER_BIN = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_fns', metavar='F', type=str, nargs=1,
                   help='filename of input binary npy data file')
    return p.parse_args()


def quantised(bits_per_bin, probs):
    d = 1.0 * ((2 ** bits_per_bin) - 1)
    # Force at least 1 / d weight for each quantised prob
    # Each bin with observed freq counts has to to have prob > 0.
    q = numpy.minimum(numpy.maximum(1, numpy.round(probs * d)), d)
    return q / d


def main():
    args = parse_args()

    with open(args.input_fns[0], 'rb') as f:
        data = numpy.load(f)

    n_bins_bits = numpy.ceil(numpy.log2(MAX_BINS))

    data_flat = numpy.ravel(data)
    freqs = numpy.bincount(data_flat)
    n_bins = len(freqs)
    assert n_bins <= MAX_BINS

    n = len(data_flat)
    probs = freqs / (1.0 * n)

    best_total_bits = numpy.inf
    best_params = []

    for bits_per_bin in range(MIN_BITS_PER_BIN, MAX_BITS_PER_BIN+1):
        approx_probs = quantised(bits_per_bin, probs)
        z = numpy.sum(approx_probs)
        approx_probs /= z
        log2_approx_probs = numpy.log2(approx_probs)

        # upper bound on Shannon-Huffman code length for sample assuming the
        # model
        sample_bits = numpy.ceil(1 + -numpy.sum(log2_approx_probs[data_flat]))
        # account for bits to describe model parameters
        model_bits = n_bins_bits + n_bins * bits_per_bin
        total_bits = numpy.ceil(sample_bits + model_bits)
        if total_bits < best_total_bits:
            best_total_bits = total_bits
            best_params = [
                ("", "bits/bin", bits_per_bin),
                ("subtotal", "model", model_bits),
                ("subtotal", "sample", sample_bits),
                ("total", "bits", total_bits),
            ]

    for group, key, value in best_params:
        print('%8s\t%8s\t%4d' % (group, key, value))

if __name__ == '__main__':
    main()
