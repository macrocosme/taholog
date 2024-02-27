"""
Find the matrix inverse of the central beam's Jones matrix.
"""

import h5py
import logging
import numpy as np

from numpy.lib.scimath import sqrt as csqrt

from astropy.convolution import convolve, Gaussian1DKernel

from datetime import datetime

def main(target, output, smooth=False, cond_threshold=5.):
    """
    """

    startTime = datetime.now()

    logger = logging.getLogger(__name__)

    # Read the central beam file.
    logger.info('Reading file: {0} .'.format(target))
    fth5 = h5py.File(target, 'r')
    ht = fth5.attrs
    beams = list(fth5.keys())

    # Load the central beam data.
    logger.info('Loading data.')
    # dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
    # st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
    # ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag').value, dtype=np.bool)
    dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data'), dtype=np.complex64)
    st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma'), dtype=np.complex64)
    ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag'), dtype=np.bool)

    # Apply flags to the data.
    dt = np.ma.masked_where(ft, dt)
    st = np.ma.masked_where(ft, st)

    # Load frequency axis.
    # tgt_freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq').value for b in beams])
    tgt_freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq') for b in beams])

    # Obtain calibration.
    logger.info('Generating calibration.')

    calibration = np.ma.ones(dt.shape, dtype=np.complex64)
    sigma_cal = np.ma.empty(dt.shape, dtype=np.complex64)
    cond = np.empty(dt.shape[0], dtype=np.float64)

    # Do each time individually to avoid errors due to a singular time slot.
    for t in range(dt.shape[0]):

        dtt = dt[t,:,:]
        stt = st[t,:,:]

        try:

            calibration[t,:,:] = np.linalg.inv(dtt)
            cond[t] = np.linalg.cond(dtt)

            # Get the error on each entry.
            sign_det, logdet_dt = np.linalg.slogdet(dtt)
            det_dt = sign_det*np.ma.exp(logdet_dt, dtype=np.complex128)
            det_dt_2 = np.power(det_dt, -2., dtype=np.complex128)
            sigma_cal[t,0,0] = det_dt_2*csqrt(np.power(dtt[1,1]*dtt[1,1]*stt[0,0], 2., dtype=np.complex128) +
                                              np.power(dtt[1,1]*dtt[1,0]*stt[0,1], 2., dtype=np.complex128) +
                                              np.power(dtt[1,1]*dtt[0,1]*stt[1,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,1]*dtt[1,0]*stt[1,1], 2., dtype=np.complex128))
            sigma_cal[t,0,1] = det_dt_2*csqrt(np.power(dtt[1,1]*dtt[0,1]*stt[0,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[1,1]*stt[0,1], 2., dtype=np.complex128) +
                                              np.power(dtt[0,1]*dtt[0,1]*stt[1,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[0,1]*stt[1,1], 2., dtype=np.complex128))
            sigma_cal[t,1,0] = det_dt_2*csqrt(np.power(dtt[1,0]*dtt[1,1]*stt[0,0], 2., dtype=np.complex128) +
                                              np.power(dtt[1,0]*dtt[1,0]*stt[0,1], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[1,1]*stt[1,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[1,0]*stt[1,1], 2., dtype=np.complex128))
            sigma_cal[t,1,1] = det_dt_2*csqrt(np.power(dtt[0,1]*dtt[1,0]*stt[0,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[1,0]*stt[0,1], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[0,1]*stt[1,0], 2., dtype=np.complex128) +
                                              np.power(dtt[0,0]*dtt[0,0]*stt[1,1], 2., dtype=np.complex128))

        except np.linalg.linalg.LinAlgError:

            logger.info('Singular matrix.')
            calibration[t,:,:].fill(np.nan)
            sigma_cal[t,:,:].fill(np.nan)

    calibration = np.ma.masked_invalid(calibration)
    sigma_cal = np.ma.masked_invalid(sigma_cal)

    # Mask based on cond.
    mask_ = (cond > cond_threshold)
    mask = np.array([[mask_,mask_],[mask_,mask_]]).T
    calibration = np.ma.masked_where(mask, calibration)

    # Smooth.
    if smooth:

        calibration = calibration.filled(np.nan)

        smooth_cal = np.empty(dt.shape, dtype=np.complex64)

        logger.info('Convolving the solutions.')

        kernel = Gaussian1DKernel(stddev=3.)

        # Use lists to avoid issues with astropy and python2.
        xxre = convolve(list(calibration[:,0,0].real.astype(np.float)), kernel)
        xxim = convolve(list(calibration[:,0,0].imag.astype(np.float)), kernel)

        yyre = convolve(list(calibration[:,1,1].real.astype(np.float)), kernel)
        yyim = convolve(list(calibration[:,1,1].imag.astype(np.float)), kernel)

        xyre = convolve(list(calibration[:,0,1].real.astype(np.float)), kernel)
        xyim = convolve(list(calibration[:,0,1].imag.astype(np.float)), kernel)

        yxre = convolve(list(calibration[:,1,0].real.astype(np.float)), kernel)
        yxim = convolve(list(calibration[:,1,0].imag.astype(np.float)), kernel)

        smooth_cal[:,0,0] = xxre + 1.j*xxim
        smooth_cal[:,1,1] = yyre + 1.j*yyim
        smooth_cal[:,1,0] = yxre + 1.j*yxim
        smooth_cal[:,0,1] = xyre + 1.j*xyim

        calibration = np.ma.masked_invalid(smooth_cal)

    # Save.
    logger.info('Saving calibration to: {0} .'.format(output))
    save_calibration_hdf5(output, calibration.filled(np.nan), sigma_cal.filled(np.nan), tgt_freq, ht)

    logger.info('Step run time: {0}'.format(datetime.now() - startTime))

def save_calibration_hdf5(output, data, sigma, freq, target_head):
    """
    """

    f = h5py.File(output, "w")

    # Create groups where the data will be stored.
    f0 = f.create_group('0')
    # Calibration solutions.
    a = f0.create_group('CAL')
    a['cal'] = data
    # Errors on the calibration solutions.
    b = f0.create_group('SIGMACAL')
    b['sigmacal'] = sigma
    # Frequency axis.
    c = f0.create_group('FREQ')
    c.create_dataset('freq', data=freq)

    # Common attributes.
    f.attrs['target_source_name'] = target_head['target_source_name']
    f.attrs['frequency_hz'] = target_head['frequency_hz']
    f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']
    f.attrs['reference_antennas'] = target_head['reference_antennas']
    f.attrs['target_antennas'] = target_head['target_antennas']
    f.attrs['mjd_start'] = target_head['mjd_start']
    f.attrs['mjd_end'] = target_head['mjd_end']
    f.attrs['utc_start'] = target_head['utc_start']
    f.attrs['utc_end'] = target_head['utc_end']
    f.attrs['time_samples'] = target_head['time_samples']
    f.attrs['integration_time_sample'] = target_head['integration_time_sample']

    f.close()
