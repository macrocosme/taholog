import os
import sys
import glob
import h5py
import numpy as np
from utils import check_slash, check_folder_exists_or_create
from datetime import datetime

import h5py

try:
    from numba import (
        jit,
        cuda,
        prange,
        typeof,
        float32,
        float64,
        complex64,
        int64,
        boolean,
    )

    has_numba_cuda = True
except:
    has_numba_cuda = False

try:
    from mpi4py import MPI

    has_mpi = True
except:
    has_mpi = False

import multiprocessing as mp

from ..steps.xcorr import (
    cross_correlate,
    cross_correlate_jit,
    fast_average,
    estimate_error,
    determine_averaging_factor,
)

from .to_freq import *

import cupy as cp

has_cupy = True
ncpus = 16
n_gpu_devices = 2


# Let's override these for cross-correlation
def call_func(func, dt, dr, i, j, k, ch0, chf, parallel, use_numba):
    return func(dt, dr, i, j, k, ch0, chf, parallel, use_numba)


def time_func(func, dt, dr, i, j, k, ch0, chf, parallel, use_numba):
    start_time = datetime.now()
    call_func(func, dt, dr, i, j, k, ch0, chf, parallel, use_numba)
    return datetime.now() - start_time


# Beginning
os.chdir(output_dir)

print("Checking that there are enough output files.")
for ref in reference_ids:
    for beam in ref_beams:
        # all_files = glob.glob(f'{output_dir}{ref}/{cs_str}/*spw*.h5')
        all_files = glob.glob(f"{output_dir}{ref}/{beam}/*spw*.h5")
        continuing = len(all_files) == spws

if continuing:
    edges = 0.25
    xcorr_output_dir = f"{output_dir}{target_id}_xcorr"

    # Needs output_dir, cs_str, reference_ids, target_id, xcorr_dt, params, parallel
    target = (
        lambda ibm, spw: f"{output_dir}{target_id}/{target_id}_SAP000_B{ibm:03d}_P000_bf_spw{spw}.h5"
    )
    refers = (
        lambda ref, refbm, spw: f"{output_dir}{ref}/{refbm}/{ref}_SAP000_B000_P000_bf_spw{spw}.h5"
    )
    output = (
        lambda ref, refbm, ibm, spw: f"{xcorr_output_dir}/{ref}/{refbm}/SAP000_B{ibm:03d}_P000_spw{spw}_avg{xcorr_dt}.h5"
    )
    rfi_output = (
        lambda ref, refbm, ibm, spw: f"{xcorr_output_dir}/{ref}/{refbm}/SAP000_B{ibm:03d}_P000_spw{spw}_rfiflags.h5"
    )

    rfi_kwargs = {
        "flagging_threshold": xcorr_flagging_threshold,
        "threshold_shrink_power": xcorr_threshold_shrink_power,
        "ext_time_percent": 0.5,
        "ext_freq_percent": 0.5,
        "n_rfi_max": 1,
    }
    kwargs = {
        "target_time_res": xcorr_dt,
        "rfiflag": xcorr_rfiflag,
        "edges": xcorr_edges,
        "rfi_kwargs": rfi_kwargs,
    }

    def initialize(ibm, spw, refid, ref_beam):
        tgt = target(ibm, spw)
        ref = refers(refid, ref_beam, spw)
        out = output(refid, ref_beam, ibm, spw)
        rfi = rfi_output(refid, ref_beam, ibm, spw)

        kwargs["rfi_output"] = rfi

        # xcorr.main(tgt, ref, out, **kwargs)

        ft = h5py.File(tgt, "r")
        ht = ft.attrs
        beams = list(ft.keys())

        # Open the reference station file.
        fr = h5py.File(ref, "r")
        hr = fr.attrs

        # Load reference data.
        dr = np.array(
            [fr[f"/{b}/DATA"].get("data") for b in fr.keys()], dtype=np.complex64
        )

        ntimes = dr.shape[1]
        nchans = dr.shape[3]
        ch0 = int(nchans * edges)
        chf = int(nchans * (1.0 - edges))

        # Load frequency axes and compare.
        tgt_freq = np.array([ft[f"/{b}/FREQ"].get("freq") for b in fr.keys()])
        ref_freq = np.array([fr[f"/{b}/FREQ"].get("freq") for b in fr.keys()])

        if np.nansum(tgt_freq - ref_freq) != 0:
            print("Target and reference station frequencies do not match.")
            print("Will now exit.")
            return 0

        # Number of time slots, channels, jones matrix.
        xcorr = np.zeros((ntimes,) + (chf - ch0,) + (2, 2), dtype=np.complex64)
        radec = np.zeros((2))
        print(f"Data has a shape: {xcorr.shape}")

        # Cross-correlate the data.
        print("Will cross-correlate the data.")

        # Load target data. Only one beam per file.
        # dt = ft['/{0}/DATA'.format(beams[0])].get('data').value
        dt = ft[f"/{beams[0]}/DATA"].get("data")

        # Where are we looking at?
        beam_info = ft[f"/{beams[0]}/POINTING"]

        return (
            fr,
            ft,
            xcorr,
            radec,
            tgt_freq,
            ref_freq,
            dr,
            dt,
            ntimes,
            nchans,
            ch0,
            chf,
            beam_info,
        )

    # Overall loop:
    # for refid in reference_ids:
    #     for ref_beam in ref_beams:
    #         for spw in xcorr_spws:
    #             for ibm in xcorr_beams:

    ibm, spw, refid, ref_beam = 0, 0, reference_ids[0], ref_beams[0]
    (
        fr,
        ft,
        xcorr,
        radec,
        tgt_freq,
        ref_freq,
        dr,
        dt,
        ntimes,
        nchans,
        ch0,
        chf,
        beam_info,
    ) = initialize(ibm, spw, refid, ref_beam)
    #                 xcorr, radec, tgt_freq, ref_freq, dr, dt, ntimes, nchans, ch0, chf, beam_info = initialize(ibm, spw, refid, ref_beam)

    print("cross_correlate")
    print(time_func(cross_correlate, dt[:], dr, 0, 0, 0, ch0, chf, True, False))

    print("cross_correlate_jit")
    print(time_func(cross_correlate, dt[:], dr, 0, 0, 0, ch0, chf, False, True))

    print("cross_correlate parallel jit")
    print(time_func(cross_correlate, dt[:], dr, 0, 0, 0, ch0, chf, True, True))

    print("cross_correlate")
    print(time_func(cross_correlate, dt[:], dr, 0, 0, 0, ch0, chf, False, False))

    # fr.close()
    # ft.close()
else:
    print("not enough files, stopping.")
