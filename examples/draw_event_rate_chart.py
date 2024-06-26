#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ecf_utils import read_ecf_info, read_ecf_info_fast


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Draw Event Rate Chart sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-hdf-file', dest='input_path', type=str,
                        help="Path to input HDF (ECF) file.", required=True)
    parser.add_argument('--fast', dest='fast', action='store_true', default=False,
                        help="Whether indexes info stored in HDF (ECF) is used for event rate estimation or not")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample demonstrating how to use H5ECFDatReader to load HDF5 (ECF) file.")

    if args.fast:
        info = read_ecf_info_fast(args.input_path)
    else:
        info = read_ecf_info(args.input_path)

    ev_rate = info.get('ev_count_ndarray')
    rate_unit = info.get('ev_count_unit')
    # convert to Ev/s
    if rate_unit == 'ms':
        ev_rate = ev_rate * 1000.0
    ev_rate = ev_rate / (1024.0 * 1024.0)  # to MEv/s
    if ev_rate.ndim == 2:
        plt.plot(ev_rate[:, 0], label='ON', c='#407EC8')
        plt.plot(ev_rate[:, 1], label='OFF', c='#1E2534')
        plt.legend()
    else:
        plt.plot(ev_rate)
    plt.ylabel('MEv/s')
    plt.xlabel(rate_unit)
    plt.show()


if __name__ == "__main__":
    main()
