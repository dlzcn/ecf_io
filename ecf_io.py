# -*- coding: utf-8 -*-
"""
Experimental impl. for Prophesee ECF HDF5 events file format
"""

import h5py
import numpy as np
from typing import Union, Tuple, Optional, Dict
from metavision_sdk_base import EventCD, EventExtTrigger


H5Z_FILTER_ECF = 0x8ecf
_kIndexesPeriodUs = 2000  # 2ms
_kChunkSize = 16384


_EventCD_DTYPE = {'names': ['x', 'y', 'p', 't'],
                  'formats': ['<u2', '<u2', '<i2', '<i8'],
                  'offsets': [0, 2, 4, 8],
                  'itemsize': 16}

_EventExtTrigger_DTYPE = {'names': ['p', 't', 'id'],
                          'formats': ['<i2', '<i8', '<i2'],
                          'offsets': [0, 8, 16], 'itemsize': 24}

_Event_IDX_DTYPE = [('id', '<u8'), ('ts', '<i8')]


class H5ECFDataset(object):
    """
        ECF Dataset (writer)
    """

    def __init__(self, file: h5py.File, ev_trigger: bool = False):
        self._dataset_size_increment = 10 * _kChunkSize
        if not ev_trigger:
            grp = file.create_group('CD', track_order=True)
            self._dat_ds = grp.create_dataset(
                'events', shape=(self._dataset_size_increment,),
                maxshape=(None,), chunks=(_kChunkSize, ), compression=H5Z_FILTER_ECF,
                dtype=np.dtype(_EventCD_DTYPE))
            self._dat_idx_ds = grp.create_dataset(
                'indexes', shape=(self._dataset_size_increment,),
                maxshape=(None,), chunks=(_kChunkSize, ),
                dtype=np.dtype(_Event_IDX_DTYPE))
        else:
            grp = file.create_group('EXT_TRIGGER', track_order=True)
            self._dat_ds = grp.create_dataset(
                'events', shape=(self._dataset_size_increment,),
                maxshape=(None,), chunks=(_kChunkSize, ),
                dtype=np.dtype(_EventExtTrigger_DTYPE))
            self._dat_idx_ds = grp.create_dataset(
                'indexes', shape=(self._dataset_size_increment,),
                maxshape=(None,), chunks=(_kChunkSize, ),
                dtype=np.dtype(_Event_IDX_DTYPE))

        self._dat: Union[None, np.ndarray] = None
        self._dat_index: int = 0
        self._dat_idx = []
        self._dat_idx_next_ts = 0
        self._dat_idx_index: int = 0
        self._first_ts: int = 0
        self._last_ts: int = 0

    @property
    def data_index(self) -> int:
        return self._dat_index

    @property
    def last_ts(self) -> int:
        return self._last_ts

    def write(self, data):
        """
        Writes event buffer into a compressed packet

        Args:
            data (ndarray): events of type EventCD or events of type ExtTrigger
        """
        if not len(data):
            return

        if self._dat is None:
            self._dat = np.array(data, copy=True)
        else:
            self._dat = np.concatenate([self._dat, np.array(data)])

        offset = 0
        while self._dat.size - offset > _kChunkSize:
            data = self._dat[offset: offset + _kChunkSize]
            self._write_data(data)
            offset += _kChunkSize

        self._dat = self._dat[offset:]

    def _write_data(self, data):
        n_dat = len(data)
        assert n_dat
        self._update_data_idx(data)

        if self._dat_ds.shape[0] < self._dat_index + n_dat:
            self._dat_ds.resize((self._dat_ds.shape[0] + self._dataset_size_increment,))

        self._dat_ds[self._dat_index: self._dat_index + n_dat, ] = data.copy()
        self._dat_index += n_dat
        self._last_ts = data['t'][-1]

    def _update_data_idx(self, data):
        if self._dat_idx_next_ts == 0:
            self._first_ts = data['t'][0]
            self._dat_idx.append((0, -1))  # be careful!
            self._dat_idx.append((0, 0))  # update the first valid index
            self._dat_idx_ds.attrs['offset'] = f'{-self._first_ts}'.encode('ascii')
            self._dat_idx_next_ts = _kIndexesPeriodUs + self._first_ts

        last_ev_ts, start_pos, dat_ts = data['t'][-1], 0, data['t']
        while self._dat_idx_next_ts <= last_ev_ts:
            rst = np.where(dat_ts[start_pos:] >= self._dat_idx_next_ts)
            assert rst[0].size
            start_pos += rst[0][0]
            start_pos_ts = dat_ts[start_pos]
            # calculate duplicated counts, yes this is the design of current indexes -_-!
            _dat_idx_next_ts = int((start_pos_ts - self._first_ts) // _kIndexesPeriodUs + 1) * _kIndexesPeriodUs + self._first_ts
            _duplicated_counts = max((_dat_idx_next_ts - self._dat_idx_next_ts) // _kIndexesPeriodUs, 1)
            self._dat_idx.extend(
                [(start_pos + self._dat_index, start_pos_ts - self._first_ts)] * _duplicated_counts
            )
            self._dat_idx_next_ts = _dat_idx_next_ts

        offset = 0
        while len(self._dat_idx) - offset >= _kChunkSize:
            events_idx = self._dat_idx[offset: offset + _kChunkSize]
            self._write_data_idx(events_idx)
            offset += _kChunkSize

        del self._dat_idx[:offset]

    def _write_data_idx(self, events_idx):
        if not events_idx:
            return

        events_idx_arr = np.array(events_idx, dtype=np.dtype(_Event_IDX_DTYPE))
        if self._dat_idx_ds.shape[0] < self._dat_idx_index + events_idx_arr.size:
            self._dat_idx_ds.resize(
                (self._dat_idx_ds.shape[0] + self._dataset_size_increment,)
            )

        self._dat_idx_ds[self._dat_idx_index: self._dat_idx_index + events_idx_arr.size] = events_idx_arr
        self._dat_idx_index += events_idx_arr.size

    def close(self):
        """ Dump remaining data and resize the datasets """
        if self._dat is not None:
            self._write_data(self._dat)
            self._dat = None
        if self._dat_idx:
            self._write_data_idx(self._dat_idx)
            self._dat_idx = []

        self._dat_ds.resize((self._dat_index,))
        self._dat_idx_ds.resize((self._dat_idx_index,))


class H5ECFDatWriter(object):
    """
    Writes Event Packets via H5 ECF filter plugin

    Args:
        hdf_file (str): destination path
        height (int): height of recording
        width (int): width of recording
    """

    def __init__(self, hdf_file: str, height: int, width: int, metadata: Optional[Dict[str, str]] = None):
        self._dataset_size_increment = 10 * _kChunkSize

        self._path = hdf_file
        self._f = h5py.File(hdf_file, 'w')
        self._is_close = False

        self._events_datasets = H5ECFDataset(self._f, False)
        self._trigger_datasets = H5ECFDataset(self._f, True)

        self.update_metadata(metadata, True)
        self._f.attrs['version'] = '1.0'.encode('ascii')
        self._f.attrs['geometry'] = f'{width}x{height}'.encode('ascii')  # rewrite the info
        self._f.attrs['generation'] = '4.0'.encode('ascii')

    def update_metadata(self, metadata: Optional[Dict[str, str]] = None, clean=False):
        """ Write metadata as attributes of the HDF5 file
        set clean for removing 'evt' and 'plugin_name' in systeminfo from RAW file or Camera (the Hal Device)
        """
        if metadata is None:
            return
        drop_keys = ('Connection', 'Data Encoding Format')
        _metadata = {'format': 'ECF'}
        for key, val in metadata.items():
            test_result = [True for v in drop_keys if v in key]
            if sum(test_result):
                continue
            if key == 'Sensor Info':
                _metadata['generation'] = val
            elif key == 'SystemID':
                _metadata['system_ID'] = val
            elif key == 'Serial':
                _metadata['serial_number'] = val
            else:
                _metadata[key.lower()] = val
        if clean:
            _metadata.pop('evt')
            _metadata.pop('plugin_name')
        for key, val in _metadata.items():
            self._f.attrs[key] = val.encode('ascii')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        """String representation of a `H5EventsWriter` object.

        Returns:
            string describing the H5EventsWriter state and attributes
        """
        wh_str = self._f.attrs.get('geometry').split('x')
        width, height = int(wh_str[0]), int(wh_str[1])
        wrd = ''
        wrd += 'H5ECFDatWriter: path {} \n'.format(self._path)
        wrd += 'Width {}, Height  {}\n'.format(width, height)
        wrd += 'events written : {}, last timestamp {}\n'.format(
            self._events_datasets.data_index, self._events_datasets.last_ts)
        wrd += 'triggers written : {}, last timestamp {}\n'.format(
            self._trigger_datasets.data_index, self._trigger_datasets.last_ts)
        return wrd

    def write(self, dat):
        """
        Writes CD Events or Ext Trigger Events into HDF5 file

        Args:
            dat (ndarray): events of type EventCD or events of type EvetExtTrigger
        """
        if dat.dtype == _EventCD_DTYPE:
            self._events_datasets.write(dat)
        elif dat.dtype == _EventExtTrigger_DTYPE:
            self._trigger_datasets.write(dat)
        else:
            raise ValueError('Unknown dtype, only EventCD and EventExtTrigger are supported!')

    def write_cdevents(self, events):
        """
        Writes event buffer into HDF5 file (ECF format, chunked)

        Args:
            events (ndarray): events of type EventCD
        """
        self._events_datasets.write(events)

    def write_exttrigger(self, trigger):
        """
        Writes trigger buffer into HDF5 file (chunked)

        Args:
            trigger (ndarray): events of type EventExtTrigger
        """
        self._trigger_datasets.write(trigger)

    def close(self):
        if self._is_close:
            return

        self._events_datasets.close()
        self._trigger_datasets.close()
        self._f.close()
        self._is_close = True

    def __del__(self):
        self.close()


class H5ECFDatReader(object):
    """
    Reads & Seeks into a hdf5 file of compressed (ECF format) events (with indexes).

    Args:
        hdf_file (str): input path of the HDF file
    """

    def __init__(self, hdf_file, do_time_shifting=True):
        self._path = hdf_file
        self._f = h5py.File(hdf_file, 'r')
        assert self._f.attrs['version'] == '1.0'
        geometry = self._f.attrs['geometry'].split('x')
        self._width = int(geometry[0])
        self._height = int(geometry[1])

        cd_grp = self._f['CD']
        self._cd_events_ds = cd_grp['events']
        self._cd_events_indexes_ds = cd_grp['indexes']
        self._cd_events_indexes_offset: int = int(self._cd_events_indexes_ds.attrs.get('offset', 0))

        ext_trigger_grp = self._f['EXT_TRIGGER']
        self._trigger_events_ds = ext_trigger_grp['events']
        self._trigger_events_indexes_ds = ext_trigger_grp['indexes']
        self._trigger_events_indexes_offset: int = int(self._cd_events_indexes_ds.attrs.get('offset', 0))

        self.counts = len(self._cd_events_ds)
        self.start_ts = int(self._cd_events_ds[0]['t'])
        self.last_ts = int(self._cd_events_ds[-1]['t'])
        self.trigger_counts = len(self._trigger_events_ds)
        if self.trigger_counts:
            self._first_ext_trigger_ts = self._trigger_events_ds[0]['t']
            self._last_ext_trigger_ts = self._trigger_events_ds[-1]['t']
        else:
            self._first_ext_trigger_ts, self._last_ext_trigger_ts = 0, 0

        self._start_index: int = 0
        self.current_time: int = 0
        self.do_time_shifting = do_time_shifting

    def __len__(self):
        return len(self._cd_events_ds)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__()

    def __del__(self):
        if not hasattr(self, '_f'):
            return
        self._f.close()

    def __repr__(self):
        """String representation of a `H5ECFDatReader` object.

        Returns:
            string describing the H5ECFDatReader attributes and reading state
        """
        wrd = 'H5ECFDatReader: {}\n'.format(self._path)
        wrd += 'Width {}, Height  {}\n'.format(self._width, self._height)
        wrd += 'event count: {}\n'.format(self.counts)
        wrd += 'duration: {} μs \n'.format(self.duration)
        wrd += 'trigger count: {}\n'.format(self.trigger_counts)
        wrd += 'current time: {} μs \n'.format(self.current_time)
        return wrd

    def get_metadata(self) -> Dict[str, str]:
        """ Read metadata stored as attributes of the HDF5 file """
        metadata = {
            key: val for key, val in self._f.attrs.items()
        }
        return metadata

    def reset(self):
        """Resets at beginning of file."""
        if not self._f:
            self._f = h5py.File(self._path, 'r')
            # restore data set attributes
            cd_grp = self._f['CD']
            self._cd_events_ds = cd_grp['events']
            self._cd_events_indexes_ds = cd_grp['indexes']

            ext_trigger_grp = self._f['EXT_TRIGGER']
            self._trigger_events_ds = ext_trigger_grp['events']
            self._trigger_events_indexes_ds = ext_trigger_grp['indexes']
            # double check
            assert self.counts == len(self._cd_events_ds)
            assert self.start_ts == int(self._cd_events_ds[0]['t'])
            assert self.last_ts == int(self._cd_events_ds[-1]['t'])
            assert self.trigger_counts == len(self._trigger_events_ds)

        self.current_time = 0
        self._start_index = 0

    def seek_time(self, ts: int):
        """
        Move the position to the event whose timestamp is before and closest to ts.
        """
        _ts = ts + self.start_ts if self.do_time_shifting else ts
        assert _ts < self.last_ts, "the seek timestamp is even beyond the max timestamp of the events!"

        _ts_offset = _ts + self._cd_events_indexes_offset
        indexes_idx = _ts_offset // _kIndexesPeriodUs + 1  # stupid 1st value (0, -1), index + 1
        _start_index = self._cd_events_indexes_ds[indexes_idx][0]
        next_indexes_idx = indexes_idx + 1
        if next_indexes_idx < len(self._cd_events_indexes_ds):
            _end_index = self._cd_events_indexes_ds[next_indexes_idx][0]
        else:
            _end_index = -1

        cd_event_t = self._cd_events_ds[_start_index: _end_index]['t']
        _sub_start_index = np.searchsorted(cd_event_t, _ts)
        self._start_index = int(_start_index + max(0, _sub_start_index))
        self.current_time = ts

    def get_size(self):
        """ Returns geometry or size (height, width) of the sensor """
        return self._height, self._width

    @property
    def path(self) -> str:
        """ Path of the opened HDF5 file """
        return self._path

    @property
    def duration(self) -> int:
        """ Duration of the events stored in the HDF5 file """
        if self.do_time_shifting:
            return self.last_ts - self.start_ts
        else:
            return self.last_ts

    def __iter__(self):
        while self._start_index < self.counts:
            _start_index = self._start_index
            _end_index = int(self._start_index // _kChunkSize + 1) * _kChunkSize

            if _end_index >= self.counts:
                _end_index = -1
                self._start_index = self.counts
            else:
                self._start_index = _end_index

            events = self._cd_events_ds[_start_index: _end_index]
            if self.do_time_shifting:
                events['t'] = events['t'] - self.start_ts

            self.current_time = events['t'][-1]

            yield events

    def get_event_count(self, toggle_unit=False, toggle_thresh=1800) -> Tuple[np.ndarray, str]:
        """ Get event count per millisecond or second of CD Events in the HDF file
            The count is calculated via interpolation on data from Indexes of the CD Events

        Args:
            toggle_unit: set to True if ev counts unit should be changed automatically (default is False).
            toggle_thresh: maximum length of the record to trigger unit toggle (default > 1800 sec).

        Returns:
            ev_count (np.ndarray): event count array per millisecond or second
            ev_count_unit (str): the unit, valid output is 'ms' or 's'
        """
        if (self.start_ts == 0 and self.last_ts == 0) or (self.last_ts - self.start_ts <= 0):
            raise RuntimeError('Duration of event buffers is invalid')

        cd_events_indexes = self._cd_events_indexes_ds[:]
        c, t = cd_events_indexes['id'], cd_events_indexes['ts']
        rst = np.where(t >= 0)
        if rst[0].size:
            n_pos = rst[0][0]
            c, t = c[n_pos:], t[n_pos:]
        # using interpolation to calculate ev counts
        ts_val = np.arange(0, self.duration + 1000, 1000)
        c_val = np.interp(ts_val, t, c)
        ev_count = np.diff(c_val).astype('int64')

        if toggle_unit and len(ev_count) > toggle_thresh * 1000:
            unit = 's'
            last_pos = len(ev_count) // 1000 * 1000
            ev_count_per_s = np.sum(ev_count[:last_pos].reshape(-1, 1000), axis=1)
            last_counts = np.sum(ev_count[last_pos:])
            ev_count = np.concatenate([ev_count_per_s, [last_counts]])
        else:
            unit = 'ms'

        return ev_count, unit

    def info(self, toggle_unit=False, toggle_thresh=1800):
        """
        Collects basic information of the CD Events in the HDF5 file

        Args:
            toggle_unit: set to True if ev counts unit should be changed automatically (default is False).
            toggle_thresh: maximum length of the record to trigger unit toggle (default > 1800 sec).

        returns:
            dict: 'duration', 'count', 'ev_count_ndarray': col 0 is pos, col 1 is neg,
            'ev_count_unit' and 'trigger_count'
        """
        ev_count, unit = self.get_event_count(toggle_unit, toggle_thresh)

        info = {'duration': int(self.duration), 'count': int(self.counts),
                'ev_count_ndarray': ev_count, 'ev_count_unit': unit,
                'trigger_count': int(self.trigger_counts)}

        return info

    def get_ext_trigger_events(self, ignore_current_time=False):
        """
        Load external trigger events whose timestamp are less than the current time.

        Args:
            ignore_current_time: set to True to get all external trigger events

        returns:
            ndarray: dtype is EventExtTrigger
        """
        _current_ts = self.current_time + self.start_ts if self.do_time_shifting else self.current_time

        if ignore_current_time:
            trigger_events = self._trigger_events_ds[:]
            if self.do_time_shifting:
                trigger_events['t'] = trigger_events['t'] - self.start_ts
            return trigger_events

        if _current_ts < self._first_ext_trigger_ts:
            return np.empty((0,), dtype=_EventExtTrigger_DTYPE)
        elif _current_ts >= self._last_ext_trigger_ts:
            trigger_events = self._trigger_events_ds[:]
            if self.do_time_shifting:
                trigger_events['t'] = trigger_events['t'] - self.start_ts
            return trigger_events
        else:
            _ts_offset = _current_ts + self._trigger_events_indexes_offset
            indexes_idx = _ts_offset // _kIndexesPeriodUs + 1  # stupid 1st value (0, -1), index + 1
            if indexes_idx < len(self._trigger_events_indexes_ds):
                _curr_index = self._trigger_events_indexes_ds[indexes_idx]
                trigger_events = self._trigger_events_ds[: _curr_index + 1]
            else:
                trigger_events = self._trigger_events_ds[:]
            if self.do_time_shifting:
                trigger_events['t'] = trigger_events['t'] - self.start_ts
            return trigger_events
