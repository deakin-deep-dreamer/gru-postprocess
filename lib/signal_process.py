
import numpy as np

from scipy.stats import zscore


def zscore_signal(
    signal
):
    r"""zscore given signal."""
    return zscore(signal, axis=0)


def sig_sub_mean(
    sig
):
    r"""Signal normalisation."""
    if len(sig.shape) == 1:
        return sig - np.mean(sig)
    n_samp, n_chan = sig.shape
    sig_new = np.zeros(sig.shape)
    for i_chan in range(n_chan):
        tmp_sig = sig[:, i_chan]
        sig_new[:, i_chan] = tmp_sig - np.mean(tmp_sig)
    return sig_new


def zscore_hz_wise(
    sig, hz=None
):
    r"""Scale raw signal. Second-wise mean difference"""
    if hz is None:
        raise ValueError("Hz expected, received None.")
    i_window = 0
    is_done = False

    if len(sig.shape) == 1:
        return zscore_hz_wise_single_chan(sig, hz)

    n_samp, n_chan = sig.shape
    sig_new = np.zeros(sig.shape)
    while True:
        i_start = hz*i_window
        if n_samp - hz*i_window < hz:
            i_end_seg = i_start + (n_samp - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        for i_chan in range(n_chan):
            sig_new[i_start:i_end_seg, i_chan] = zscore(
                sig[i_start:i_end_seg, i_chan])
        i_window += 1
        if is_done:
            break
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == sig_new.shape[0]
    return sig_new


def zscore_hz_wise_single_chan(
    sig, hz=None
):
    r"""Scale raw signal. Second-wise mean difference"""
    new_sig = []
    i_window = 0
    is_done = False
    while True:
        i_start = hz*i_window
        if len(sig) - hz*i_window < hz:
            i_end_seg = i_start + (len(sig) - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        sec_seg = sig[i_start:i_end_seg]
        new_sig.extend(
            # (np.array(sec_seg) - np.mean(sec_seg))
            zscore(sec_seg)
        )
        i_window += 1
        if is_done:
            break
    new_sig = np.array(new_sig)
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == new_sig.shape[0]
    return new_sig


def sub_mean_hz_wise(
    sig, hz=None
):
    r"""Scale raw signal. Second-wise mean difference"""
    if hz is None:
        raise ValueError("Hz expected, received None.")
    i_window = 0
    is_done = False

    if len(sig.shape) == 1:
        return sub_mean_hz_wise_single_chan(sig, hz)

    n_samp, n_chan = sig.shape
    sig_new = np.zeros(sig.shape)
    while True:
        i_start = hz*i_window
        if n_samp - hz*i_window < hz:
            i_end_seg = i_start + (n_samp - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        for i_chan in range(n_chan):
            tmp_seg = sig[i_start:i_end_seg, i_chan]
            sig_new[i_start:i_end_seg, i_chan] = tmp_seg - np.mean(tmp_seg)
        i_window += 1
        if is_done:
            break
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == sig_new.shape[0]
    return sig_new


def sub_mean_hz_wise_single_chan(
    sig, hz=None
):
    r"""Scale raw signal. Second-wise mean difference"""
    new_sig = []
    # sig = sig.flatten()
    # for i_window in range(1 + len(sig)//hz):
    i_window = 0
    is_done = False
    while True:
        i_start = hz*i_window
        if len(sig) - hz*i_window < hz:
            i_end_seg = i_start + (len(sig) - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        sec_seg = sig[i_start:i_end_seg]
        new_sig.extend(
            # (np.array(sec_seg) - np.mean(sec_seg))
            # zscore(sec_seg)
            sec_seg - np.mean(sec_seg)
        )
        i_window += 1
        if is_done:
            break
    new_sig = np.array(new_sig)
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == new_sig.shape[0]
    return new_sig
