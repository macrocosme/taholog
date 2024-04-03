#!/usr/bin/env python

import sys
import h5py
import logging
import numpy as np

from numpy.lib.scimath import sqrt as csqrt

from . import misc, flag

from datetime import datetime

def main(target, reference, output, target_time_res=0, rfiflag=False, edges=0.25, rfi_output='', rfi_kwargs={}):
    """
    Cross correlates the target data observed with the tied array with that stored in the reference observation.
    It will remove edge channels, cross-correlate, average in time and perform rfi flagging on the data.
    
    """

    start_time = datetime.now()

    logger = logging.getLogger(__name__)

    logger.info('Working on target file: {0}'.format(target))
    logger.info('And reference file: {0}'.format(reference))
    logger.info('And output file: {0}'.format(output))

    # Open the target station file.
#    try:
    ft = h5py.File(target, 'r')
#    except IOError:
#        logger.info('Target file {0} not found.'.format(target))
        #raise IOError('Target file {0} not found.'.format(target))
        #sys.exit(1)
    ht = ft.attrs
    beams = list(ft.keys())

    # Open the reference station file.
    fr = h5py.File(reference, 'r')
    hr = fr.attrs
    # Load reference data.
    # dr = np.array([fr['/{0}/DATA'.format(b)].get('data').value for b in fr.keys()], dtype=np.complex64)
    dr = np.array([fr['/{0}/DATA'.format(b)].get('data') for b in fr.keys()], dtype=np.complex64)

    ntimes = dr.shape[1]
    nchans = dr.shape[3]
    ch0 = int(nchans*edges)
    chf = int(nchans*(1. - edges))
    logger.info('Will keep channels between: {0}--{1}.'.format(ch0, chf))

    # Load frequency axes and compare.
    # tgt_freq = np.array([ft['/{0}/FREQ'.format(b)].get('freq').value for b in fr.keys()])
    # ref_freq = np.array([fr['/{0}/FREQ'.format(b)].get('freq').value for b in fr.keys()])
    tgt_freq = np.array([ft['/{0}/FREQ'.format(b)].get('freq') for b in fr.keys()])
    ref_freq = np.array([fr['/{0}/FREQ'.format(b)].get('freq') for b in fr.keys()])

    if np.nansum(tgt_freq - ref_freq) != 0:
        logger.info('Target and reference station frequencies do not match.')
        logger.info('Will now exit.')
        sys.exit(1)

    # Number of time slots, channels, jones matrix.
    xcorr = np.zeros((ntimes,)+(chf-ch0,)+(2,2), dtype=np.complex64)
    radec = np.zeros((2))
    logger.info('Data has a shape: {0}'.format(xcorr.shape))

    # Cross-correlate the data.
    logger.info('Will cross-correlate the data.')

    # Load target data. Only one beam per file.
    # dt = ft['/{0}/DATA'.format(beams[0])].get('data').value
    dt = ft['/{0}/DATA'.format(beams[0])].get('data')

    # Cross-correlate: target voltage times complex conjugate of reference voltage.
    xcorr[:,:,0,0] = dt[:,0,ch0:chf]*np.conj(dr[0,:,0,ch0:chf])  # XX
    xcorr[:,:,0,1] = dt[:,0,ch0:chf]*np.conj(dr[0,:,1,ch0:chf])  # XY
    xcorr[:,:,1,0] = dt[:,1,ch0:chf]*np.conj(dr[0,:,0,ch0:chf])  # YX
    xcorr[:,:,1,1] = dt[:,1,ch0:chf]*np.conj(dr[0,:,1,ch0:chf])  # YY 

    # Where are we looking at?
    beam_info = ft['/{0}/POINTING'.format(beams[0])]

    # Averaging.
    if target_time_res > 0:

        # Get the time resolution of the data.
        try:
            integration_s = ht['integration_time_s']
        except KeyError:
            logger.warning('Integration time not present in header. Will determine from observing time and number of time slots.')
            t0 = ht['mjd_start']
            tf = ht['mjd_end']
            obs_time = (tf - t0)*24*3600.
            integration_s = obs_time/ntimes
        logger.info('Input data has an integration time of: {0} s'.format(integration_s))
        # Determine the averaging factor to reach the desired time resolution.
        averaging_values = np.array(list(misc.factors(ntimes)))
        averaging_factor = averaging_values[np.argmin(abs(int(round(target_time_res/integration_s)) - averaging_values))]
        time_res = integration_s*averaging_factor

        logger.info('Averaging data in time by a factor of: {0}'.format(averaging_factor))
        logger.info('Desired time resolution was: {0} s'.format(target_time_res))
        logger.info('Actual time resolution will be: {0} s'.format(time_res))

        # Average to the desired time resolution.
        axis = 0
        xcorr_avgt0 = np.average((xcorr).reshape((-1,averaging_factor,)+xcorr.shape[1:]), axis=1)

        logger.info('Time averaged data has a shape of: {0}'.format(xcorr_avgt0.shape))

        # Estimate the error for each time sample on the averaged data.
        drz = np.zeros(xcorr_avgt0.shape, dtype=np.float32)
        diz = np.zeros(xcorr_avgt0.shape, dtype=np.float32)

        for t in range(xcorr_avgt0.shape[0]):

            t0_ = int(t*averaging_factor)
            tf_ = int((t+1)*averaging_factor)

            drz[t] = np.std(xcorr[t0_:tf_,:,:,:].real, axis=0)/ \
                            np.sqrt(xcorr[t0_:tf_,:,:,:].shape[0] - 1., dtype=np.float64)
            diz[t] = np.std(xcorr[t0_:tf_,:,:,:].imag, axis=0)/ \
                            np.sqrt(xcorr[t0_:tf_,:,:,:].shape[0] - 1., dtype=np.float64)

        logger.info('Masking invalid values.')
        xcorr_avgt0 = np.ma.masked_invalid(xcorr_avgt0)

        # Radio Frequency Interference flag.
        if rfiflag:

            logger.info('Starting RFI flagging.')

            length_of_max_dimension = xcorr_avgt0.shape[0]
            flagging_threshold = rfi_kwargs['flagging_threshold']
            threshold_shrink_power = rfi_kwargs['threshold_shrink_power']
            flag_window_lengths =  2**np.arange(int(np.ceil(np.log(length_of_max_dimension)/np.log(2.))))

            # Flag each polarization product independently.
            logger.info('flagging each polarization.')
            flags_xx = flag.rfi_flag(abs(xcorr_avgt0[:,:,0,0]), flagging_threshold, flag_window_lengths, threshold_shrink_power)
            flags_xy = flag.rfi_flag(abs(xcorr_avgt0[:,:,0,1]), flagging_threshold, flag_window_lengths, threshold_shrink_power)
            flags_yx = flag.rfi_flag(abs(xcorr_avgt0[:,:,1,0]), flagging_threshold, flag_window_lengths, threshold_shrink_power)
            flags_yy = flag.rfi_flag(abs(xcorr_avgt0[:,:,1,1]), flagging_threshold, flag_window_lengths, threshold_shrink_power)

            time_percent = rfi_kwargs['ext_time_percent']
            freq_percent = rfi_kwargs['ext_freq_percent']

            # Extend the flags.
            logger.info('Extending the flags.')
            flags_xx = flag.extend_flags(flags_xx, time_percent=time_percent, freq_percent=freq_percent)
            flags_xy = flag.extend_flags(flags_xy, time_percent=time_percent, freq_percent=freq_percent)
            flags_yx = flag.extend_flags(flags_yx, time_percent=time_percent, freq_percent=freq_percent)
            flags_yy = flag.extend_flags(flags_yy, time_percent=time_percent, freq_percent=freq_percent)

            # Apply new masks.
            xcorr_avgt0[:,:,0,0].mask = [flags_xx]*xcorr_avgt0.shape[0]
            xcorr_avgt0[:,:,0,1].mask = [flags_xy]*xcorr_avgt0.shape[0]
            xcorr_avgt0[:,:,1,0].mask = [flags_yx]*xcorr_avgt0.shape[0]
            xcorr_avgt0[:,:,1,1].mask = [flags_yy]*xcorr_avgt0.shape[0]

            # Mask n_rfi_max largest values.
            for i in range(rfi_kwargs['n_rfi_max']):
                xcorr_avgt0[:,:,0,0].mask[np.unravel_index(np.ma.argmax(abs(xcorr_avgt0[:,:,0,0])), xcorr_avgt0[:,:,0,0].shape)] = True
                xcorr_avgt0[:,:,1,1].mask[np.unravel_index(np.ma.argmax(abs(xcorr_avgt0[:,:,1,1])), xcorr_avgt0[:,:,1,1].shape)] = True

            # Save flags.
            if rfi_output != '':
                logger.info('Saving RFI flag masks to: {0}'.format(rfi_output))
                save_flags_hdf5(rfi_output, xcorr_avgt0, tgt_freq, ht, hr['reference_antennas'], time_res)

        # Compute weights based on the error.
        wrz = np.ma.power(drz, -2.)
        wiz = np.ma.power(diz, -2.)

        logger.info('Averaging in frequency to a single channel.')
        xcorr_avg = np.ma.average(xcorr_avgt0.real, axis=1, weights=wrz)
        xcori_avg = np.ma.average(xcorr_avgt0.imag, axis=1, weights=wiz)
        logger.info('Frequency averaged data has a shape of: {0}'.format(xcorr_avg.shape))

        # Determine new errors based on the averaged data.
        logger.info('Generating weighted error estimates.')
        if rfiflag:
            n_points = np.sum(~xcorr_avgt0.mask, axis=1)
        else:
            n_points = xcorr_avgt0.shape[1]
        weighted_sigma_re = csqrt(n_points/(n_points - 1.)* \
                             np.ma.sum(wrz*np.power(xcorr_avgt0.real - xcorr_avg[:,np.newaxis], 2.), axis=1) \
                             /np.ma.sum(wrz, axis=1))
        weighted_sigma_im = csqrt(n_points/(n_points - 1.)* \
                             np.ma.sum(wiz*np.power(xcorr_avgt0.imag - xcori_avg[:,np.newaxis], 2.), axis=1) \
                             /np.ma.sum(wiz, axis=1))

    else:
        logger.info('No averaging done.')
        xcorr_avg = xcorr.real
        xcori_avg = xcorr.imag
        weighted_sigma_re = np.ma.ones(xcorr.shape)
        weighted_sigma_im = np.ma.ones(xcorr.shape)

    # Join real and imaginary parts.
    logger.info('Joining real and imaginary parts.')
    xcorr = xcorr_avg + 1.j*xcori_avg
    xcorr = np.ma.masked_invalid(xcorr)
    sigma = weighted_sigma_re + 1.j*weighted_sigma_im
    sigma = np.ma.masked_where(xcorr.mask, sigma)

    # Save.
    logger.info('Saving averaged data to {0}.'.format(output))
    save_hdf5(output, xcorr, sigma, tgt_freq, beam_info, ht, hr['reference_antennas'], time_res)

    logger.info('Step run time: {0}'.format(datetime.now() - start_time))

def save_flags_hdf5(output, data, freq, target_head, reference_antenna, time_resolution):
    """
    Saves the flags applied during the cross-correlation step.
    
    """

    logger = logging.getLogger(__name__)

    f = h5py.File(output, "w")

    # Create groups where the data will be stored.
    f0 = f.create_group('0')

    a = f0.create_group('FLAG')
    a['flag'] = data.mask

    b = f0.create_group('FREQ')
    b.create_dataset('freq', data=freq)

    # Common attributes.
    f.attrs['target_source_name'] = target_head['target_source_name']
    f.attrs['frequency_hz'] = target_head['frequency_hz']
    f.attrs['reference_antennas'] = reference_antenna
    f.attrs['target_antennas'] = target_head['target_antennas']
    f.attrs['mjd_start'] = target_head['mjd_start']
    f.attrs['mjd_end'] = target_head['mjd_end']
    f.attrs['utc_start'] = target_head['utc_start']
    f.attrs['utc_end'] = target_head['utc_end']
    f.attrs['time_samples'] = target_head['time_samples']
    f.attrs['integration_time_sample'] = time_resolution
    try:
        f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']
    except KeyError:
        logger.warning('header keyword: "sample_rate_hz" not present.')
        f.attrs['sample_rate_hz'] = np.nan

    f.close()

def save_hdf5(output, data, sigma, freq, beam_info, target_head, reference_antenna, time_resolution):
    """
    Saves the cross correlated data into a new hdf5 file.
    """

    logger = logging.getLogger(__name__)

    # Create file.
    # f = h5py.File(output, "w")
    with h5py.File(output, "w") as f:

        # Create groups where the data will be stored.
        f0 = f.create_group('0')

        a = f0.create_group('POINTING')
        # a['beam_ref_radec'] = beam_info['beam_ref_radec'].value
        # a['beam_pos_radec'] = beam_info['beam_pos_radec'].value
        # a['beam_off_radec'] = beam_info['beam_off_radec'].value
        a.create_dataset('beam_ref_radec', data=beam_info['beam_ref_radec'])
        a.create_dataset('beam_pos_radec', data=beam_info['beam_pos_radec'])
        a.create_dataset('beam_off_radec', data=beam_info['beam_off_radec'])


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
        f.attrs['reference_antennas'] = reference_antenna
        f.attrs['target_antennas'] = target_head['target_antennas']
        f.attrs['mjd_start'] = target_head['mjd_start']
        f.attrs['mjd_end'] = target_head['mjd_end']
        f.attrs['utc_start'] = target_head['utc_start']
        f.attrs['utc_end'] = target_head['utc_end']
        f.attrs['time_samples'] = target_head['time_samples']
        f.attrs['integration_time_sample'] = time_resolution
        try:
            f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']
        except KeyError:
            logger.warning('header keyword: "sample_rate_hz" not present.')
            f.attrs['sample_rate_hz'] = np.nan

    # Close file. 
    # f.close()

