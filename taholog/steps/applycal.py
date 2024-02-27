"""
"""

import sys
import h5py
import logging
import numpy as np

from numpy.lib.scimath import sqrt as csqrt

from datetime import datetime

def main(cal_sols, target, output):
    """
    """

    startTime = datetime.now()

    logger = logging.getLogger(__name__)

    # Read the file with the calibration solutions.
    logger.info('Reading file with calibration: {0} .'.format(cal_sols))
    sols_file = h5py.File(cal_sols, 'r')
    sols_head = sols_file.attrs
    sols_keys = list(sols_file.keys())

    # Load the calibration solutions.
    # sd = sols_file['/{0}/CAL'.format(sols_keys[0])].get('cal').value
    # ss = sols_file['/{0}/SIGMACAL'.format(sols_keys[0])].get('sigmacal').value
    sd = sols_file['/{0}/CAL'.format(sols_keys[0])].get('cal')
    ss = sols_file['/{0}/SIGMACAL'.format(sols_keys[0])].get('sigmacal')

    # Read the file to be calibrated.
    logger.info('Reading file with data: {0} .'.format(target))
    fth5 = h5py.File(target, 'r')
    ht = fth5.attrs
    beams = list(fth5.keys())

    # Load the data to be calibrated,
    # dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
    dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data'), dtype=np.complex64)
    # its error
    # st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
    st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma'), dtype=np.complex64)
    # and its flags.
    # ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag').value, dtype=np.bool)
    ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag'), dtype=np.bool)

    # Apply flags.
    dt = np.ma.masked_where(ft, dt)
    st = np.ma.masked_where(ft, st)

    # Load frequency axes and compare.
    # tgt_freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq').value for b in beams])
    # sol_freq = np.array([sols_file['/{0}/FREQ'.format(b)].get('freq').value for b in sols_keys])
    tgt_freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq') for b in beams])
    sol_freq = np.array([sols_file['/{0}/FREQ'.format(b)].get('freq') for b in sols_keys])

    if np.nansum(tgt_freq - sol_freq) != 0:
        logger.info('Target and calibration solution frequencies do not match.')
        logger.info('Will now exit.')
        sys.exit(1)

    # Create calibrated data array.
    calibrated = np.ma.zeros(sd.shape, dtype=np.complex64)
    sigmacal_re = np.ma.zeros(sd.shape, dtype=np.complex64)
    sigmacal_im = np.ma.zeros(sd.shape, dtype=np.complex64)

    # Calibrate the target data.
    logger.info('Calibrating.')
    calibrated[:,0,0] = dt[:,0,0]*sd[:,0,0] + dt[:,0,1]*sd[:,1,0]
    calibrated[:,1,0] = dt[:,1,0]*sd[:,0,0] + dt[:,1,1]*sd[:,1,0]
    calibrated[:,0,1] = dt[:,0,0]*sd[:,0,1] + dt[:,0,1]*sd[:,1,1]
    calibrated[:,1,1] = dt[:,1,0]*sd[:,0,1] + dt[:,1,1]*sd[:,1,1]

    # Propagate the error.
    logger.info('Propagating the errors.')
    mult = lambda a, da, b, db, c, dc, d, dd: np.power(b*da, 2., dtype=np.complex128) + \
                                              np.power(a*db, 2., dtype=np.complex128) + \
                                              np.power(d*dc, 2., dtype=np.complex128) + \
                                              np.power(c*dd, 2., dtype=np.complex128)
    sigmacal_re[:,0,0] = csqrt(mult(dt[:,0,0].real, st[:,0,0].real, sd[:,0,0].real, ss[:,0,0].real,
                                    dt[:,0,1].real, st[:,0,1].real, sd[:,1,0].real, ss[:,1,0].real))
    sigmacal_re[:,1,0] = csqrt(mult(dt[:,1,0].real, st[:,1,0].real, sd[:,0,0].real, ss[:,0,0].real,
                                    dt[:,1,1].real, st[:,1,1].real, sd[:,1,0].real, ss[:,1,0].real))
    sigmacal_re[:,0,1] = csqrt(mult(dt[:,0,0].real, st[:,0,0].real, sd[:,0,1].real, ss[:,0,1].real,
                                    dt[:,0,1].real, st[:,0,1].real, sd[:,1,1].real, ss[:,1,1].real))
    sigmacal_re[:,1,1] = csqrt(mult(dt[:,1,0].real, st[:,1,0].real, sd[:,0,1].real, ss[:,0,1].real,
                                    dt[:,1,1].real, st[:,1,1].real, sd[:,1,1].real, ss[:,1,1].real))
    sigmacal_im[:,0,0] = csqrt(mult(dt[:,0,0].imag, st[:,0,0].imag, sd[:,0,0].imag, ss[:,0,0].imag,
                                    dt[:,0,1].imag, st[:,0,1].imag, sd[:,1,0].imag, ss[:,1,0].imag))
    sigmacal_im[:,1,0] = csqrt(mult(dt[:,1,0].imag, st[:,1,0].imag, sd[:,0,0].imag, ss[:,0,0].imag,
                                    dt[:,1,1].imag, st[:,1,1].imag, sd[:,1,0].imag, ss[:,1,0].imag))
    sigmacal_im[:,0,1] = csqrt(mult(dt[:,0,0].imag, st[:,0,0].imag, sd[:,0,1].imag, ss[:,0,1].imag,
                                    dt[:,0,1].imag, st[:,0,1].imag, sd[:,1,1].imag, ss[:,1,1].imag))
    sigmacal_im[:,1,1] = csqrt(mult(dt[:,1,0].imag, st[:,1,0].imag, sd[:,0,1].imag, ss[:,0,1].imag,
                                    dt[:,1,1].imag, st[:,1,1].imag, sd[:,1,1].imag, ss[:,1,1].imag))

    # Mask calibrated data.
    logger.info('Masking.')
    calibrated = np.ma.masked_invalid(calibrated)
    calibrated = np.ma.masked_where(ft, calibrated)
    sigmacal = np.ma.masked_where(calibrated.mask, sigmacal_re + 1.j*sigmacal_im)

    # Where are we looking at?
    beam_info = fth5['/{0}/POINTING'.format(beams[0])]

    # Save the calibrated data.
    logger.info('Saving calibrated data to {0}.'.format(output))
    save_hdf5(output, calibrated, sigmacal, tgt_freq, beam_info, ht)

    logger.info('Script run time: {0}'.format(datetime.now() - startTime))

def save_hdf5(output, data, sigma, freq, beam_info, target_head):
    """
    """

    # Create file.
    f = h5py.File(output, "w")

    # Create groups where the data will be stored.
    f0 = f.create_group('0')

    a = f0.create_group('POINTING')
    a.create_dataset('beam_ref_radec', data=beam_info['beam_ref_radec'])
    a.create_dataset('beam_pos_radec', data=beam_info['beam_pos_radec'])
    a.create_dataset('beam_off_radec', data=beam_info['beam_off_radec'])
    # a['beam_ref_radec'] = beam_info['beam_ref_radec'].value
    # a['beam_pos_radec'] = beam_info['beam_pos_radec'].value
    # a['beam_off_radec'] = beam_info['beam_off_radec'].value

    b = f0.create_group('DATA')
    b['data'] = data.filled(np.nan)

    c = f0.create_group('SIGMA')
    c['sigma'] = sigma.filled(np.nan)

    d = f0.create_group('FLAG')
    d.create_dataset('flag', data=data.mask)

    e = f0.create_group('FREQ')
    e.create_dataset('freq', data=freq)

    # Common attributes.
    f.attrs['target_source_name'] = target_head['target_source_name']
    f.attrs['frequency_hz'] = target_head['frequency_hz']
    f.attrs['reference_antennas'] = target_head['reference_antennas']
    f.attrs['target_antennas'] = target_head['target_antennas']
    f.attrs['mjd_start'] = target_head['mjd_start']
    f.attrs['mjd_end'] = target_head['mjd_end']
    f.attrs['utc_start'] = target_head['utc_start']
    f.attrs['utc_end'] = target_head['utc_end']
    f.attrs['time_samples'] = target_head['time_samples']
    f.attrs['frequency_hz'] = target_head['frequency_hz']
    f.attrs['integration_time_sample'] = target_head['integration_time_sample']
    f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']

    # Close file. 
    f.close()
