import re
from datetime import datetime
import h5py
import logging
import numpy as np
import multiprocessing as mp

try:
    import pyfftw
    has_pyfftw = True
    # has_pyfftw = False
except ImportError:
    has_pyfftw = False

try:
    from numba import (
        jit, cuda, prange, typeof,
        float32, float64, complex64, int64, boolean
    )
    has_numba_cuda = True
except ImportError: 
    has_numba_cuda = False

try:
    import cupy as cp
    has_cupy = True
    cp.fft.config.use_multi_gpus = True
    cp.fft.config.set_cufft_gpus([0, 1]) # Devices fixed for now; to be adjusted later.

except ImportError: 
    has_cupy = False



def get_props_from_filename(filename, props=['SAP', 'S', 'B']):
    """
    Extracts numbers and codes relevant to load the corresponding h5 tables given a filename.
    """

    fnprops = dict.fromkeys(props)

    for i,prop in enumerate(props):
        fnprops[prop] = re.findall('{0}\d+'.format(prop), filename)[0][len(prop):]

    return fnprops

def set_nspec(filename, nchan=256):
    r"""
    Determines how many spectra will a time series contain 
    given the desired number of channels in the spectra.
    """

    props = get_props_from_filename(filename)
    table = f"SUB_ARRAY_POINTING_{props['SAP']}/BEAM_{props['B']}"

    f = h5py.File(filename, 'r')
    datah5 = f[table+f"/STOKES_{props['S']}"]
    head3 = f[table].attrs
    nspw = datah5.shape[1]  # n spectral window(s), a.k.a. number of subbands
    nspec = datah5.shape[0]//nchan
    ntime = datah5.shape[0]
    sampl = head3['SAMPLING_RATE']

    return nspw, nspec, ntime, sampl

def fft_pyfftw_original(data, nspec, npol, nchan, polmap, threads=1):
    # pyFFTW aligned arrays to store input and output of DFT.
    infft = pyfftw.empty_aligned(nchan, dtype=np.complex64)
    outfft = pyfftw.empty_aligned(nchan, dtype=np.complex64)
    # pyFFTW FFTW object to perform the actual DFT.
    fft_object = pyfftw.FFTW(infft, outfft, threads=threads)
    # Data.
    fftdata = pyfftw.zeros_aligned((nspec,npol,nchan), dtype=np.complex64)

    for t in range(nspec):
        for p in range(npol):
            if len(polmap[p]) == 2:
                i0, i1 = polmap[p]
                infft[:] = data[t*nchan:(t+1)*nchan, i0] + 1j*data[t*nchan:(t+1)*nchan, i1]
                fft_object()
                fftdata[t:t+1, p, :] = np.fft.fftshift(outfft)
            else:
                i0 = polmap[p]
                infft = data[t*nchan:(t+1)*nchan, i0]
                fft_object()
                fftdata[t:t+1,p,:] = np.fft.fftshift(outfft)

    return fftdata

def fft_pyfftw(data, nspec, npol, nchan, polmap, threads=1):
    # Reshape array
    d = data.reshape(nspec, nchan, polmap.size)
    # pyFFTW aligned arrays to store input and output of DFT.
    infft = pyfftw.empty_aligned((nspec, nchan), dtype=np.complex64)
    outfft = pyfftw.empty_aligned((nspec, nchan), dtype=np.complex64)
    # pyFFTW FFTW object to perform the actual DFT.
    fft_object = pyfftw.FFTW(infft, outfft, axes=(1,), threads=threads)
    # Data.
    fftdata = pyfftw.zeros_aligned((nspec,npol,nchan), dtype=np.complex64)

    for p in range(npol):
        if len(polmap[p]) == 2:
            i0, i1 = polmap[p]
            infft[:, :] = d[:, :, i0] + 1j*d[:, :, i1]
            fft_object()
            fftdata[:, p, :] = np.fft.fftshift(outfft, axes=(1,))
        else:
            i0 = polmap[p]
            infft = d[:, :, i0]
            fft_object()
            fftdata[:,p,:] = np.fft.fftshift(outfft, axes=(1,))

    return fftdata

def fft_cupy(cpu_data, nspec, npol, nchan, polmap, device=0):
    with cp.cuda.Device(device):
        gpu_fftdata = cp.zeros((nspec,npol,nchan), dtype=np.complex64)
        # CPU to GPU
        gpu_data = cp.array(cpu_data.reshape(nspec, nchan, polmap.size), dtype=np.float64)    

        for p in range(npol):  # X Y
            if len(polmap[p]) == 2:
                i0,i1 = polmap[p]
                gpu_fftdata[:, p, :] = cp.fft.fftshift(
                                        cp.fft.fft(gpu_data[:, :, i0] + 1j*gpu_data[:, :, i1], 
                                                    axis=1),
                                        axes=(1,)
                                    )
            elif len(polmap[p]) == 1:
                i0 = polmap[p]
                gpu_fftdata[:, p,:] = cp.fft.fftshift(cp.fft.fft(data[:, :, i0], axis=1), axes=(1,))

        # GPU back to CPU 
        cpu_fftdata = cp.asnumpy(gpu_fftdata)
    return cpu_fftdata

def fft_numpy(data, nspec, npol, nchan, polmap):
    fftdata = np.zeros((nspec,npol,nchan), dtype=np.complex64)
    d = data.reshape(nspec, nchan, polmap.size)
    for p in range(npol):
        if len(polmap[p]) == 2:
            i0,i1 = polmap[p]
            fftdata[:, p, :] = np.fft.fftshift(np.fft.fft(d[:, :, i0] + 1j*d[:, :, i1], axis=1), axes=(1,))
        elif len(polmap[p]) == 1:
            i0 = polmap[p]
            fftdata[:,p,:] = np.fft.fftshift(np.fft.fft(d[:, :, i0], axis=1), axes=(1,))
    return fftdata

def fft_numpy_all_spws(data, nspw, nspec, npol, nchan, polmap):
    fftdata = np.zeros((nspec, npol, nspw, nchan), dtype=np.complex64)
    d = data.reshape(nspec, nchan, nspw, polmap.size)
    for p in range(npol):
        if len(polmap[p]) == 2:
            i0,i1 = polmap[p]
            fftdata[:, p, :] = np.fft.fftshift(np.fft.fft(d[:, :, i0] + 1j*d[:, :, i1], axis=1), axes=(1,))
        elif len(polmap[p]) == 1:
            i0 = polmap[p]
            fftdata[:,p,:] = np.fft.fftshift(np.fft.fft(d[:, :, i0], axis=1), axes=(1,))
    return fftdata

def fft_numpy_original(data, nspec, npol, nchan, polmap):
    fftdata = np.zeros((nspec,npol,nchan), dtype=np.complex64)
    for t in range(nspec):
        for p in range(npol):
            if len(polmap[p]) == 2:
                i0,i1 = polmap[p]
                fftdata[t:t+1, p, :] = np.fft.fftshift(
                                            np.fft.fft(
                                                #    real
                                                data[t*nchan:(t+1)*nchan, i0] + \
                                                #       imaginary
                                                1j*data[t*nchan:(t+1)*nchan, i1]
                                            )
                                        )
            elif len(polmap[p]) == 1:
                i0 = polmap[p]
                fftdata[t:t+1,p,:] = np.fft.fftshift(np.fft.fft(data[t*nchan:(t+1)*nchan, i0]))
    return fftdata

def to_freq(data, nchan, nspec, npol, polmap, sample_rate, center_freq, use_cupy, device, threads=1):
    """
    Computes the FFT of a time series to obtain a spectrum.
    """     
    
    # FFT.
    if has_cupy and use_cupy:
        fftdata = fft_cupy(data, nspec, npol, nchan, polmap, device)
    elif has_pyfftw:
        fftdata = fft_pyfftw(data, nspec, npol, nchan, polmap)
    else:
        fftdata = fft_numpy(data, nspec, npol, nchan, polmap)

    # Frequency.
    fftfreq = np.zeros((nchan), dtype=np.float32)
    freqres = 1./sample_rate
    fftfreq = np.fft.fftshift(np.fft.fftfreq(nchan, freqres)) + center_freq

    # Flags.
    flags = np.zeros((nspec,npol,nchan), dtype=bool)

    return fftfreq, fftdata, flags

def setup_containers(nspec, nchan, ncorr):
    """
    Defines various arrays that will store the processed data based on the properties of the input data.
    """

    data = np.zeros((nspec, ncorr, nchan), dtype=np.complex)
    flag = np.zeros((nspec, ncorr, nchan), dtype=np.bool)

    bpos = np.zeros((2))
    boff = np.zeros((2))
    bref = np.zeros((2))
    bnam = np.zeros((1)) #np.zeros((nspw))
    beam = {'name': bnam, 'reference': bref, 'position': bpos, 'offset': boff}

    mjds = np.zeros((1)) # Start time in Major Julian Date 
    mjde = np.zeros((1)) # End time in Major Julian Date
    utcs = np.zeros((1), dtype='S128') # Start time in UTC
    utce = np.zeros((1), dtype='S128') # End time in UTC
    tant = np.zeros((1), dtype=object) # Target antennas
    rant = np.zeros((1), dtype=object) # Reference antennas
    freq = np.zeros((1)) # Reference frequency
    targ = np.zeros((1), dtype=object) # Source name
    head = {'mjd_start': mjds, 'mjd_end': mjde,
            'utc_start': utcs, 'utc_end': utce,
            'target_antennas': tant,
            'reference_antenna': rant,
            'frequency_hz': freq,
            'target_source_name': targ,
            'time_samples': 0}

    return data, flag, beam, head

def join_pols(filename, ntime, nspw, nfiles):
    """
    Joins the real and imaginary data from different polarizations (X and Y) into one array.
    """

    logger = logging.getLogger(__name__)

    """ 
    data[:,:,0] will store the real part of X, 
    data[:,:,1] the imaginary part of X, 
    data[:,:,2] the real part of Y and 
    data[:,:,3] the imaginary part of Y
    """
    data = np.zeros((ntime, nspw, nfiles), dtype=np.float64)
    indx = filename.find('S0')
    fn = lambda s: '{0}S{1}{2}'.format(filename[:indx], s, filename[indx+2:])

    props = get_props_from_filename(filename)

    for i in range(nfiles):
        fnm = fn(i)
        # logger.info('Reading file: {0}'.format(fnm))
        f = h5py.File(fnm, 'r')
        table = f"SUB_ARRAY_POINTING_{props['SAP']}/BEAM_{props['B']}/STOKES_{i}"
        data[:,:,i] = f[table]

    return data

def parse_head(filename, beam, head):
    """
    Reads an hdf5 file called `filename` and fills the 
    contents of `beam` and `head` for the spectral window 
    with index `spw_idx`.
    """

    f = h5py.File(filename, 'r')
    props = get_props_from_filename(filename)

    head1 = f['/'].attrs
    head2 = f['/SUB_ARRAY_POINTING_{0}'.format(props['SAP'])].attrs
    head3 = f['/SUB_ARRAY_POINTING_{0}/BEAM_{1}'.format(props['SAP'], props['B'])].attrs
    axis2 = f['/SUB_ARRAY_POINTING_{0}/BEAM_{1}/COORDINATES/COORDINATE_1'.format(props['SAP'], props['B'])].attrs

    # head['frequency_hz'] = axis2['AXIS_VALUES_WORLD'][spw]
    head['AXIS_VALUES_WORLD'] = axis2['AXIS_VALUES_WORLD']
    head['target_source_name'] = head1['TARGETS'][0] # Assumes only one target per file.
    head['mjd_start'] = head1['OBSERVATION_START_MJD']
    head['mjd_end'] = head1['OBSERVATION_END_MJD']
    head['utc_start'] = head1['OBSERVATION_START_UTC']
    head['utc_end'] = head1['OBSERVATION_END_UTC']
    head['target_antennas'] = head3['STATIONS_LIST']
    head['reference_antenna'] = head3['STATIONS_LIST']
    head['sample_rate_hz'] = head3['SAMPLING_RATE']

    beam['name'] = int(props['B'])
    beam['reference'] = [head2['POINT_RA'], head2['POINT_DEC']]
    beam['position'] = [head3['POINT_RA'], head3['POINT_DEC']]
    beam['offset'] = [head3['POINT_OFFSET_RA'], head3['POINT_OFFSET_DEC']]

def save_hdf5(output, freq, data, flag, beam, head):
    """
    """

    # Create file.
    f = h5py.File(output, "w")

    # Create groups where the data will be stored.
    f0 = f.create_group('0')
    a = f0.create_group('POINTING')
    a['beam_ref_radec'] = beam['reference']
    a['beam_pos_radec'] = beam['position']
    a['beam_off_radec'] = beam['offset']
    b = f0.create_group('DATA')
    b['data'] = data
    c = f0.create_group('FLAGS')
    c.create_dataset('flags', data=flag)
    d = f0.create_group('FREQ')
    d.create_dataset('freq', data=freq)

    # Common attributes.
    f.attrs['target_source_name'] = head['target_source_name']
    f.attrs['frequency_hz'] = head['frequency_hz']
    f.attrs['reference_antennas'] = head['reference_antenna']
    f.attrs['target_antennas'] = head['target_antennas']
    f.attrs['mjd_start'] = head['mjd_start']
    f.attrs['mjd_end'] = head['mjd_end']
    f.attrs['utc_start'] = head['utc_start']
    f.attrs['utc_end'] = head['utc_end']
    f.attrs['time_samples'] = head['time_samples']
    f.attrs['sample_rate_hz'] = head['sample_rate_hz']
    f.attrs['integration_time_s'] = head['integration_time_s']

    # Close file. 
    f.close()

def call_fft(ncpus, n_gpu_devices, input_file, output_base, beam, head, beam_data, spw, nchan, nspec, npols, polmap, ntime, smplr):
    ctime = datetime.now()
    logger = logging.getLogger(__name__)

    head['frequency_hz'] = head['AXIS_VALUES_WORLD'][spw]

    # Check if process should be sent to CPU or GPU
    if ncpus > 1:
        p_info = mp.current_process()
        pid = p_info.pid
        use_cupy = pid % ncpus < n_gpu_devices
        device = pid % ncpus if use_cupy else None
        # device = None
        # use_cupy = False
    else:
        use_cupy = True
        device = 0

    # logger.info(f'Use cuda: {use_cupy}')
    # logger.info(f"Will process it on the {'cpu' if not use_cupy else 'gpu'} {device if use_cupy else ''}")

    # Read the header and extract relevant information.
    

    freq, beam_data_nu, beam_flag_nu = to_freq(beam_data[:,spw,:],
                                               nchan,
                                               nspec,
                                               npols,
                                               polmap,
                                               smplr,
                                               head['frequency_hz'],
                                               use_cupy, 
                                               device)

    data = beam_data_nu
    flag = beam_flag_nu

    # Last header update before writing.
    head['time_samples'] = ntime
    head['integration_time_s'] = nchan*1./smplr

    # Write output.
    output = output_base + '_spw{0}.h5'.format(spw)
    # logger.info('Saving file: {0}'.format(output))
    save_hdf5(output, freq, data, flag, beam, head)

    # logger.info(f'Processed spectral window: {spw}')
    logger.info(f'{use_cupy} {device}: Processed spectral window {spw} in: {datetime.now() - ctime} -- {input_file.split("/")[-1]}')

def main(input_file, output_base, nchan=64, npols=2, nfiles=4, polmap=[[0,1],[2,3]], parallel=False, ncpus=1, n_gpu_devices=2, this_spw=-1):
    r"""
    Main body of script.
    
    All files should have the same shape for this to work properly.
    More precisely, all files should correspond to different beams 
    of the same observation.
    """

    start_time = datetime.now()

    logger = logging.getLogger(__name__)

    # logger.info('Working on file: {0}'.format(input_file))

    nspw, nspec, ntime, smplr = set_nspec(input_file, nchan=nchan)

    data, flag, beam, head = setup_containers(nspec, nchan, npols)

    beam_data = join_pols(input_file, ntime, nspw, nfiles)

    # Add integration time to header.
    #head['integration_time_s'] = nchan*1./smplr

    # Select a particular spectral window/subband if specified
    if this_spw == -1:
        do_spws = range(nspw)
    else:
        do_spws = this_spw

    # logger.info('do_spws {0}'.format(this_spw))

    parse_head(input_file, beam, head)

    # logger.info('Will process spectral windows: {0} {1}'.format(do_spws, input_file.split('/')[-1]))
    if not parallel:
        for spw in do_spws:
            call_fft(ncpus, n_gpu_devices, input_file, output_base, beam, head, beam_data, spw, nchan, nspec, npols, polmap, ntime, smplr)
    else:
        logger.info(f"Multiprocessing: {ncpus} processes.")
        pool = mp.Pool(processes=ncpus)
        for spw in do_spws:
            pool.apply_async(call_fft, 
                             args=(ncpus, n_gpu_devices, input_file, output_base, beam, head, beam_data, spw, nchan, nspec, npols, polmap, ntime, smplr))
        pool.close()
        pool.join()

    logger.info('Processed file {0} in {1}'.format(input_file.split('/')[-1], datetime.now() - start_time))

