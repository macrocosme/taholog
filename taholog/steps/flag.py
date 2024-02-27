"""
Flagging functions.
"""

import numpy as np

from aoflagger import sumthreshold

def extend_flags(flags, time_percent=0.5, freq_percent=0.5):
    """
    Given a table of flags in the frequency time domain it flags all times 
    and channels which have a flag percentage larger than time_percent and 
    freq_percent, respectively.
    """

    time_flags = np.copy(flags)
    freq_flags = np.copy(flags)

    time_mean = flags.mean(axis=0)

    mask = (time_mean >= time_percent)

    # Mask channels between completely flagged channels.
    for i,m in enumerate(mask):
        if i == len(mask) - 2:
            break
        if m == True and mask[i+2] == True and mask[i+1] == False:
            mask[i+1] = True

    time_flags[:,:] = np.array([mask]*flags.shape[0]).reshape(flags.shape)

    freq_mean = flags.mean(axis=1)

    mask = (freq_mean >= freq_percent)

    freq_flags[:,:] = np.array([mask]*flags.shape[1]).T.reshape(flags.shape)

    extended_flags = time_flags | freq_flags | flags

    return extended_flags

def rfi_flag(frame_data, flagging_threshold, flag_window_lengths, threshold_shrink_power):
    """
    Uses a sumthreshold algorithm to flag RFI in frame_data.
    """

    # Convert data to a Z score.
    frame_data = np.ma.asarray(frame_data, dtype=np.float32)
    frame_data -= frame_data.mean()
    frame_data_std = frame_data.std()
    if frame_data_std != 0.0:
        frame_data /= frame_data_std

    # Look for flags in the 2D data.
    flags = sumthreshold.sum_threshold_2d(frame_data,
                                          np.copy(frame_data.mask),
                                          flagging_threshold,
                                          window_lengths=flag_window_lengths,
                                          threshold_shrink_power=threshold_shrink_power)

    return flags
