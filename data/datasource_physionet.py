
import os
import sys
import random
import re

import numpy as np
import pandas as pd
from scipy import signal

import wfdb
import wfdb.processing as wfdb_processing

import scipy.io as sio
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from lib import signal_process


HOME = "/home/XXX"
DB_BASE_PATH = f"{HOME}/data"
DB_NAMES = [
    'mitdb', 'incartdb', 'qtdb', 'edb', 'stdb', 'twadb', 'nstdb', 'ltstdb',
    'svdb', 'fantasia', 'cpsc19'
]
DB_HZ = {
    DB_NAMES[0]: 360,
    DB_NAMES[1]: 257,
    DB_NAMES[2]: 250,
    DB_NAMES[4]: 360,
    DB_NAMES[5]: 500,
    DB_NAMES[6]: 360,
    DB_NAMES[8]: 128,
    DB_NAMES[-1]: 500,
}


def read_cpsc_signal(
    index
):
    r"""Read CPSC data."""
    if index.endswith('.mat'):
        raise ValueError("Expected index, found .mat.")
    db_name = DB_NAMES[-1]
    data_path = f"{DB_BASE_PATH}/{db_name}/data"
    rpos_path = f"{DB_BASE_PATH}/{db_name}/ref"

    # index = re.split('[_.]', rpos_file.split('/')[-1])[1]
    ecg_file = f"data_{index}.mat"
    rpos_file = f"R_{index}.mat"
    ecg_path = os.path.join(data_path, ecg_file)
    ref_path = os.path.join(rpos_path, rpos_file)
    ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
    # ecg_data = np.expand_dims(ecg_data, axis=0)
    r_ref = sio.loadmat(ref_path)['R_peak'].flatten()
    return ecg_data, r_ref


def read_signal(
    record_name, ann_ext='atr', sampfrom=0, sampto=None
):
    r"""Load data file.

    Returns
    - signal: signals
    - annot: annotations
    """
    file_wo_ext = record_name
    if file_wo_ext.endswith(".hea"):
        file_wo_ext = file_wo_ext.replace('.hea', '')

    signal, fields = wfdb.rdsamp(
        f"{file_wo_ext}", sampfrom=sampfrom, sampto=sampto)
    annot = wfdb.rdann(
        f"{file_wo_ext}", ann_ext, sampfrom=sampfrom, sampto=sampto)
    return signal, annot


def resample_ann(
    ann_sample, fs, fs_target
):
    """
    Compute the new annotation indices.

    Parameters
    ----------
    ann_sample : ndarray
        Array of annotation locations.
    fs : int
        The starting sampling frequency.
    fs_target : int
        The desired sampling frequency.

    Returns
    -------
    ndarray
        Array of resampled annotation locations.

    """
    ratio = fs_target/fs
    return (ratio * ann_sample).astype(np.int64)


def _resample_sig(
    x, fs, fs_target
):
    """
    Resample a signal to a different frequency.

    Parameters
    ----------
    x : ndarray
        Array containing the signal.
    fs : int, float
        The original sampling frequency.
    fs_target : int, float
        The target frequency.

    Returns
    -------
    resampled_x : ndarray
        Array of the resampled signal values.
    resampled_t : ndarray
        Array of the resampled signal locations.

    """
    t = np.arange(x.shape[0]).astype('float64')

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0]*fs_target/fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = pd.Series(x.reshape((-1,))).interpolate().values
    resampled_x, resampled_t = signal.resample(x, num=new_length, t=t)
    # print(f"x:{x.shape}, resamp_x:{resampled_x.shape}, resamp_t:{resampled_t.shape}")
    # assert resampled_x.shape == resampled_t.shape and resampled_x.shape[0] == new_length
    assert np.all(np.diff(resampled_t) > 0)

    # return resampled_x, resampled_t
    return resampled_x


def resample_signal(
    signal, fs, fs_target
):
    # print(f"[resample_signal] sig:{signal.shape}, {fs} -> {fs_target}")
    resampled_x = _resample_sig(
        signal, fs, fs_target)
    return resampled_x


class QrsDataset(Dataset):
    r"""Physionet ECG dataset."""

    def __init__(
        self, dbname, hz=None, seg_sec=1, seg_slide_sec=1,
        valid_beats='NLRBAaJSVrFejn/fQ', point_beat_annot=False, log=print,
        q_offset_sec=0.05
        # label_map=None,
    ):
        r"""Instance."""
        super(QrsDataset, self).__init__()
        self.log = log
        self.hz = hz if hz else DB_HZ[dbname]
        self.dbname = dbname
        self.seg_sec = seg_sec
        self.seg_slide_sec = seg_slide_sec
        self.q_offset_sec = q_offset_sec
        self.valid_beats = valid_beats
        self.point_beat_annot = point_beat_annot
        self.seg_sz = self.hz * seg_sec
        # self.label_map = label_map
        self.input_dir = os.path.join(DB_BASE_PATH, self.dbname)

        self.record_names = []
        self.record_wise_segments = {}
        self.signals, self.annotations, self.labels = {}, {}, {}

        self._initialize()

        r"Shuffle segment global indexe."
        self.indexes = [i for i in range(len(self.record_wise_segments))]
        np.random.shuffle(self.indexes)

        # self.record_names.sort()
        random.shuffle(self.record_names)

    def _initialize(self):
        r"""Initialise dataset."""
        self.header_files = []
        if self.dbname == DB_NAMES[-1]:
            r"CPSC-19 dataset."
            # data_path = f"{self.input_dir}/data"
            rpos_path = f"{self.input_dir}/ref"
            self.log(f"rpos_path: {rpos_path}")
            for rpos_file in os.listdir(rpos_path):
                # g = os.path.join(self.input_dir, rpos_file)
                index = re.split('[_.]', rpos_file)[1]
                # ecg_file = f"data_{index}.mat"
                self.header_files.append(index)
        else:
            for f in os.listdir(self.input_dir):
                g = os.path.join(self.input_dir, f)
                if f.endswith(".hea"):
                    self.header_files.append(g)
        self.header_files.sort()
        self.log(
            f"Input:{self.input_dir}, headers: {len(self.header_files)}, "
            f"seg_size:{self.seg_sz}")

        self.log("Loading data...")

        r"Read record and annotations."
        beat_count = 0
        for i_hea in range(len(self.header_files)):
            hea_file = self.header_files[i_hea]
            r"Extract recordname from header file path and remove .hea"
            if self.dbname == DB_NAMES[-1]:
                record_name = hea_file
            else:
                record_name = (lambda x: x[:x.find('.hea')])(
                    hea_file.split('/')[-1])
            self.record_names.append(record_name)
            r"initialise record-wise segment index storage"
            self.record_wise_segments[record_name] = []

            try:
                if self.dbname == DB_NAMES[-1]:
                    signal, annot = read_cpsc_signal(
                        hea_file)
                else:
                    signal, annot = read_signal(
                        hea_file, ann_ext='qrs' if self.dbname == 'twadb' else 'atr'
                    )
            except FileNotFoundError:
                del_rec = self.record_names.pop()
                self.log(f"Failed to include record:{del_rec}")
                continue

            # r"Subtract mean from whole signal."
            # signal = signal_process.sig_sub_mean(signal)
            # annotated R-locations.
            ann_samples = annot
            if self.dbname != DB_NAMES[-1]:
                r"Physionet data annotation extracted differently."
                ann_samples = annot.sample

            r"At this stage, resample signal/ann if self.hz is different."
            if DB_HZ[self.dbname] != self.hz:
                self.log(
                    f"{self.dbname} is {DB_HZ[self.dbname]}Hz, "
                    f"target:{self.hz}Hz")
                signal = resample_signal(
                    signal, DB_HZ[self.dbname], self.hz)
                ann_samples = resample_ann(
                    ann_samples, DB_HZ[self.dbname], self.hz)
                self.log(
                    f"resample sig:{signal.shape}, label:{ann_samples.shape}")

            r"Zero append signal to be divisible to seg_sz."
            # self.log(f"Residue signal:{len(signal.shape)}")
            if len(signal.shape) == 1:
                signal = signal.reshape((-1, 1))
            sig_residue = signal.shape[0] % self.seg_sz
            zero_feed = np.zeros((
                sig_residue, signal.shape[1]
            ))
            signal = np.vstack((signal, zero_feed))

            self.signals[record_name] = signal
            # self.annotations[record_name] = ann_samples

            r"Prepare signal length annotation, mark 1 to R-loc, 0 elsewhere."
            labels = np.zeros(signal.shape[0])
            lbl_filter_count = 0
            lbl_filter_types = set()
            valid_ann_samples = []
            for i_label, r_loc in enumerate(ann_samples):
                r"Filter beat by type, if not CPSC dataset."
                if self.dbname != DB_NAMES[-1] and \
                        annot.symbol[i_label] not in self.valid_beats:
                    # self.log(
                    #     f"Annot:{i_label}, sample:{r_loc}, "
                    #     f"invalid-beat:{annot.symbol[i_label]}")
                    lbl_filter_count += 1
                    lbl_filter_types.add(annot.symbol[i_label])
                    continue
                # r"Use label_map to decide target label from beat-type."
                # if self.label_map:
                #     labels[r_loc] = self.label_map[annot.symbol[i_label]]
                # else:
                #     labels[r_loc] = 1
                valid_ann_samples.append(r_loc)
                n_samp_q = int(self.q_offset_sec * self.hz)
                right_offset = n_samp_q // 3
                # right_offset = 0
                lm = r_loc
                i_q_start = lm - n_samp_q
                i_q_end = lm + n_samp_q + 1 + right_offset

                '''Ensure boundary is not exceeded.'''
                i_q_start = i_q_start if i_q_start >= 0 else 0
                i_q_end = i_q_end if i_q_end <= labels.shape[0] else labels.shape[0]
                for i_q in range(i_q_start, i_q_end):
                    labels[i_q] = 1
            self.labels[record_name] = labels
            self.annotations[record_name] = valid_ann_samples
            beat_count += ann_samples.shape[0]
            self.log(
                f"[{self.record_names[-1]}] signal:{signal.shape}, "
                f"R-loc:{ann_samples.shape}, "
                f"filtered-R-loc:{lbl_filter_count}, filtered-types:{lbl_filter_types}")

        r"Do segmentation by index only."
        self.log(f"#beats:{beat_count}, Segmentating data...")
        self._segmentation()

    def _segmentation(self):
        r"""Segment records."""
        for rec_name in self.record_names:
            signal = self.signals[rec_name]
            n_samples = signal.shape[0]
            # n_samples = signal.shape[-1]
            n_window = 1 + (n_samples - self.seg_sz)\
                // (self.hz*self.seg_slide_sec)
            for i_window in range(n_window):
                seg_start = i_window * self.hz * self.seg_slide_sec
                self.record_wise_segments[rec_name].append(seg_start)
            # self.log(f"[{rec_name}] {n_window} segments calculated.")
        self.log(
            f"Segmentation done, records:{len(self.record_wise_segments)}, ")

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.record_wise_segments)

    def __getitem__(self, idx):
        return None, None

    def get_record(self, rec_name):
        r"""Return a tuple of signal, label, and annotation for the record."""
        return self.signals.get(rec_name), self.labels.get(rec_name), \
            self.annotations.get(rec_name)


class PartialQrsDataset(QrsDataset):
    r"""Segment the parent database records."""

    def __init__(
        self, dataset, record_names=None, i_seg_norm=-1, i_chan=-1, log=print,
        as_np=False
    ):
        r"""Construct PartialDataset object."""
        self.memory_ds = dataset
        # self.from_first = from_first
        self.seg_sz = dataset.seg_sz
        self.i_seg_norm = i_seg_norm
        self.i_chan = i_chan
        self.log = log
        self.as_np = as_np

        # assert record_names is not None
        if not record_names:
            log(
                f"No record_names, taking all {len(dataset.record_names)} "
                f"records of {dataset.dbname}.")
            self.record_names = self.memory_ds.record_names
        else:
            self.record_names = record_names
        # self.log(f'[{self.__class__.__name__}] training records choosen using provided record names.')
        self.segments = []
        self.labels = []

        self.initialise()

    def on_epoch_end(self):
        r"""Perform epoch-end activites, for now, shuffle segments."""
        np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Calculate the number of segments."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Return segment of specified index."""
        ID = self.indexes[idx]
        trainX = np.array(self.segments[ID])
        if self.i_chan > -1:
            trainX = trainX[:, self.i_chan]
            trainX = np.expand_dims(trainX, axis=0).T
        trainY = self.labels[ID]

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.view(1, -1)
        # X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = torch.from_numpy(trainY).type(torch.LongTensor)
        # Y_tensor = trainY

        r"Remove any nan or inf."
        if torch.any(torch.isnan(X_tensor)):
            X_tensor = torch.nan_to_num(X_tensor)
        return X_tensor, Y_tensor

    def initialise(self):
        r"""Segment the records."""
        for i_rec, rec_name in enumerate(self.record_names):
            rec, labels, _ = self.memory_ds.get_record(rec_name)
            # assert len(rec) == len(labels)
            assert rec.shape[0] == labels.shape[0]
            for i_seg_start in self.memory_ds.record_wise_segments[rec_name]:
                # seg = rec[i_seg_start:i_seg_start+self.seg_sz, :self.n_chan]
                seg = rec[i_seg_start:i_seg_start+self.seg_sz, :]

                '''Check if the last segment size equals to seg_sz.'''
                if (seg.shape[0] != self.seg_sz):
                    self.log(
                        f"++ Segment size mismatch, expecting {self.seg_sz}, "
                        f"found {seg.shape[0]}")
                    continue

                # seg = signal_process.sig_sub_mean(seg)
                seg = signal_process.zscore_hz_wise(
                    seg,
                    # hz=self.memory_ds.hz
                    hz=self.seg_sz
                    )

                r"Segment add to database. Normalise as required."
                self.segments.append(
                    seg
                )
                self.labels.append(labels[i_seg_start:i_seg_start+self.seg_sz])

        self.indexes = [i for i in range(len(self.segments))]


if __name__ == '__main__':
    dataset = QrsDataset(
        dbname=DB_NAMES[-1], hz=100)
