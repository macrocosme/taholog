import os
import sys
import h5py
import numpy as np
from utils import check_slash, check_folder_exists_or_create
from datetime import datetime

target_id='L2036952' 
reference_ids=['L2036944'] 
input_dir='/data/bassa/taholog/'
output_dir='/data/vohl/output/'
trunk_dir = check_slash(input_dir)

target_beams = 169
spws = 10

cs_str = 'cs'

to_freq_num_chan = 64  # Number of channels per spectra.
to_freq_num_pol = 2    # Number of polarizations recorded by the backend.
to_freq_num_files = 4  # Number of files across which the complex valued polarizations are spread, two for X two for Y.
to_freq_cpus = 16
to_freq_beams = range(0,target_beams)

# Check if beams are included in a single reference id or not (not ideal)
f = h5py.File(f'{trunk_dir}{reference_ids[0]}/{cs_str}/{reference_ids[0]}_SAP000_B000_S0_P000_bf.h5', "r")
ref_beams = [i for i in range(f.attrs['OBSERVATION_STATIONS_LIST'].size)]
f.close()

xcorr_spws = range(0,spws)
xcorr_beams = range(0,target_beams)
xcorr_edges = 0.125 # Flag this percentage of edge channels.
xcorr_dt = 0.4 # Time resolution in seconds after correlating.
xcorr_rfiflag = True
xcorr_flagging_threshold = 2.0      # Data is flagged if above this time the local rms.
xcorr_threshold_shrink_power = 0.05 # How big are the regions where the local rms is computed.
xcorr_cpus = 12

# Loop 1 (0, ..., n_ref)
ref = reference_ids[0]

current_dir = f"{trunk_dir}{ref}/{cs_str}/"
os.chdir(current_dir)

_outdir = check_folder_exists_or_create(f"{output_dir}{ref}", return_folder=True)

# Loop 1  (0...n_beam)
beam = 0
input_file = f'{current_dir}{ref}_SAP000_B{beam:03d}_S0_P000_bf.h5'
outdir = check_folder_exists_or_create(f"{_outdir}{beam}", return_folder=True)
output_base = f'{outdir}{ref}_SAP000_B{0:03d}_P000_bf'

nchan = to_freq_num_chan
npols = npol = num_pol = to_freq_num_pol
nfiles = to_freq_num_files

mfp = '0,1,2,3'
mfpl = mfp.split(',')
polmap = np.array(mfpl, dtype=int).reshape(num_pol, len(mfpl)//num_pol)


# to_freq   a.k.a   FFT

import h5py
try:
    from numba import (
        jit, cuda, prange, typeof,
        float32, float64, complex64, int64, boolean
    )
    has_numba_cuda = True
except: 
    has_numba_cuda = False

from ..steps.to_freq import (
    set_nspec, 
    setup_containers, 
    join_pols, 
    parse_head,
    to_freq,
    fft_cupy, fft_numpy, fft_numpy_original, fft_pyfftw, fft_pyfftw_original
)

nspw, nspec, ntime, smplr = set_nspec(input_file, nchan=nchan)
data, flag, beam, head = setup_containers(nspec, nchan, npols)
beam_data = join_pols(input_file, ntime, nspw, nfiles)

do_spws = range(nspw)

# Loop 2  (0...n_spws)
spw = 0
parse_head(input_file, beam, head, spw)

import cupy as cp
has_cupy = True


def timing():
    timing = {
        'pyfftw_original': [], 
        'pyfftw': [], 
        'numpy_original': [], 
        'numpy': [], 
        'cupy': [],
    }
    
    for m in list(timing.keys()):
        for i in range(3):
            print (m, i)
            start_time = datetime.now()
            if m == 'cupy':
                _ = fft_cupy(beam_data[:,0,:], nspec, npol, nchan, polmap)
            elif m == 'pyfftw':
                _ = fft_pyfftw(beam_data[:,0,:], nspec, npol, nchan, polmap)
            elif m == 'pyfftw_original':
                _ = fft_pyfftw_original(beam_data[:,0,:], nspec, npol, nchan, polmap)
            elif m == 'numpy':
                _ = fft_numpy(beam_data[:,0,:], nspec, npol, nchan, polmap)
            elif m == 'numpy_original':
                _ = fft_numpy_original(beam_data[:,0,:], nspec, npol, nchan, polmap)
            elapsed = datetime.now() - start_time
            timing[m].append(elapsed)
            
    return timing 