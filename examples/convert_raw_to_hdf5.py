#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from metavision_core.event_io import EventsIterator
from ecf_io import H5ECFDatWriter


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert RAW to HDF5 sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', dest='input_path', type=str,
                        help="Path to input RAW file.", required=True)
    parser.add_argument('-o', '--output-dir', required=True, help="Path to csv output directory")
    parser.add_argument('-s', '--start-ts', type=int, default=0, help="start time in microsecond")
    parser.add_argument('--delta-t', type=int, default=1000000, help="Duration of served event slice in us.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("Code sample demonstrating how to use H5ECFDatWriter to write HDF5 (ECF) file.")

    if os.path.isfile(args.input_path):
        output_file = os.path.join(args.output_dir, os.path.basename(args.input_path)[:-4] + ".hdf5")
    else:
        raise TypeError(f'Fail to access file: {args.input_path}')

    mv_iterator = EventsIterator(input_path=args.input_path,
                                 delta_t=args.delta_t, start_ts=args.start_ts)
    height, width = mv_iterator.get_size()
    i_hw_identification = mv_iterator.reader.device.get_i_hw_identification()
    system_info = i_hw_identification.get_system_info()

    ev_writer = H5ECFDatWriter(output_file, height, width)
    for evs in mv_iterator:
        if not len(evs):
            continue
        ev_writer.write(evs)

    # end of iteration
    ext_trigger_events = mv_iterator.get_ext_trigger_events()
    # rewrite ext_trigger_events to HDF5 file
    ev_writer.write_exttrigger(ext_trigger_events)
    # write meta information
    ev_writer.update_metadata(system_info)


if __name__ == "__main__":
    main()
