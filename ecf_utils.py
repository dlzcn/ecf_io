# -*- coding: utf-8 -*-
import numpy as np
from metavision_core.event_io.meta_event_producer import MetaEventBufferProducer
from ecf_io import H5ECFDatReader


def read_ecf_info_fast(path, toggle_unit=False, toggle_thresh=1800, do_time_shifting=True):
    """
    Collects information of duration by running H5ECFDatReader

    Args:
        path (str): hdf path
        toggle_unit: set to True if ev counts unit should be changed automatically (default is False).
        toggle_thresh: maximum length of the record to trigger unit toggle (default > 1800 sec).
        do_time_shifting (bool): in case of a file, makes the timestamps start close to 0mus.

    returns:
        dict: 'duration', 'count', 'ev_count_ndarray': col 0 is pos, col 1 is neg, 'ev_count_unit' and 'trigger_count'
    """
    reader = H5ECFDatReader(path, do_time_shifting)
    ev_count, unit = reader.get_event_count(toggle_unit, toggle_thresh)

    info = {'duration': int(reader.duration), 'count': int(reader.counts),
            'ev_count_ndarray': ev_count, 'ev_count_unit': unit,
            'trigger_count': int(reader.trigger_counts)}

    return info


def read_ecf_info(path, toggle_unit=False, toggle_thresh=1800, do_time_shifting=True):
    """
    Collects information of duration by running H5ECFDatReader

    Args:
        path (str): hdf path
        toggle_unit: set to True if ev counts unit should be changed automatically (default is False).
        toggle_thresh: maximum length of the record to trigger unit toggle (default > 1800 sec).
        do_time_shifting (bool): in case of a file, makes the timestamps start close to 0mus.

    returns:
        dict: 'duration', 'count', 'ev_count_ndarray': col 0 is pos, col 1 is neg, 'ev_count_unit' and 'trigger_count'
    """
    delta_t = 1000
    cd_producer = MetaEventBufferProducer(H5ECFDatReader(path, do_time_shifting), delta_t=delta_t)

    duration, counts = 0, 0
    first_batch = True
    count_per_sec_list = []  # list of numpy ndarray
    count_buf = []
    for evs in cd_producer:
        n = len(evs)
        if n:
            counts += n
            if first_batch:
                first_batch = False
                if cd_producer.current_time != delta_t:
                    # delta_t less interval between first timestamp and first event in raw file
                    count_buf += [(0, 0)] * max(0, cd_producer.current_time // delta_t - 1)
            n_pos = np.count_nonzero(evs['p'] == 0)
            n_neg = n - n_pos
            duration = evs['t'][-1]
            count_buf.append((n_pos, n_neg))
        else:
            count_buf.append((0, 0))

        if len(count_buf) == 1000:
            count_per_sec_list.append(np.array(count_buf))
            count_buf.clear()

    if toggle_unit and len(count_per_sec_list) > toggle_thresh:
        unit = 's'
        ev_count_per_s = [np.sum(v, axis=0) for v in count_per_sec_list]
        if count_buf:
            # avg_count_per_s = np.sum(np.array(count_buf), axis=0) / len(count_buf) * 1000
            avg_count_per_s = np.sum(np.array(count_buf), axis=0)  # use real counts
            ev_count_per_s.append(np.round(avg_count_per_s).astype(ev_count_per_s[0].dtype))
        ev_count = np.array(ev_count_per_s)
    else:
        unit = 'ms'
        if count_buf:
            count_per_sec_list.append(count_buf)
        ev_count = np.concatenate(count_per_sec_list)

    del count_per_sec_list

    info = {'duration': int(duration), 'count': int(counts),
            'ev_count_ndarray': ev_count, 'ev_count_unit': unit,
            'trigger_count': int(cd_producer.event_producer.trigger_counts)}

    return info
