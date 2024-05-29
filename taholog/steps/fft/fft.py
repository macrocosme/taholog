import numpy as np
import logging

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
    logger = logging.getLogger(__name__)

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    with cp.cuda.Device(device):
        gpu_fftdata = cp.zeros((nspec,npol,nchan), dtype=np.complex64)
        # CPU to GPU
        gpu_data = cp.array(cpu_data.reshape(nspec, nchan, polmap.size), dtype=np.float64)    

        logger.info(f'CPU/GPU allocation')
        logger.info(gpu_data.nbytes, gpu_fftdata.nbytes)
        logger.info(mempool.used_bytes())               
        logger.info(mempool.total_bytes())              
        logger.info(pinned_mempool.n_free_blocks())     

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
                gpu_fftdata[:, p,:] = cp.fft.fftshift(cp.fft.fft(gpu_data[:, :, i0], axis=1), axes=(1,))

        # GPU back to CPU 
        cpu_fftdata = cp.asnumpy(gpu_fftdata)

        logger.info(f'CPU/GPU final usage')
        logger.info(gpu_data.nbytes, gpu_fftdata.nbytes) 
        logger.info(mempool.used_bytes())                
        logger.info(mempool.total_bytes())               
        logger.info(pinned_mempool.n_free_blocks())      

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