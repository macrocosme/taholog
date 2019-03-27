"""
"""

import h5py
import logging
import numpy as np

from datetime import datetime

def abs_clip_vis(target, output, threshold=2.):
    """
    """

    start_time = datetime.now()

    logger = logging.getLogger(__name__)

    logger.info('Working on target file: {0}'.format(target))
    fth5 = h5py.File(target, 'r')

    ht = fth5.attrs
    beams = list(fth5.keys())
    dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
    st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
    ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag').value)

    freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq').value for b in beams])

    # Where are we looking at?
    beam_info = fth5['/{0}/POINTING'.format(beams[0])]

    logger.info('Applying masks to data and errors.')
    dt = np.ma.masked_where(ft, dt)
    st = np.ma.masked_where(ft, st)

    logger.info('Clipping real and imaginary parts.')
    logger.info('Values larger than {0} will be flagged'.format(threshold))
    dtm = np.ma.masked_where(abs(dt) > threshold, dt)

    logger.info('Saving time averaged data to: {0}'.format(output))
    save_hdf5(output, dtm, st, freq, beam_info, ht)

    logger.info('Script run time: {0}'.format(datetime.now() - start_time))

def save_hdf5(output, data, sigma, freq, beam_info, target_head):
    """
    """

    # Create file.
    f = h5py.File(output, "w")

    # Create groups where the data will be stored.
    f0 = f.create_group('0')

    a = f0.create_group('POINTING')
    a['beam_ref_radec'] = beam_info['beam_ref_radec'].value
    a['beam_pos_radec'] = beam_info['beam_pos_radec'].value
    a['beam_off_radec'] = beam_info['beam_off_radec'].value

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
    f.attrs['integration_time_sample'] = target_head['integration_time_sample']
    f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']

    # Close file. 
    f.close()
