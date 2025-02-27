"""
"""

import h5py
import logging
import numpy as np

from datetime import datetime
from numpy.lib.scimath import sqrt as csqrt

try:
    import holog.uvhol as uvhol
except ModuleNotFoundError:
    pass

from . import misc
from . import to_uvhol

def average_uvhol(input_files, reference_ids, params, output, pol):
    r'''
    Performas an inverse variance weighted average of the beam maps 
    produced using different reference stations.
    '''

    logger = logging.getLogger(__name__)

    refs = {f"{r}/{b}/":[] for b in params['ref_beams'] for r in reference_ids}

    logger.info(f"Init refs: {refs}")

    # for key in refs.keys():
    #     for beam in params['ref_beams']:
    #         k = f"{ref}/{beam}/"
    #         refs[k] = []

    logger.info('Checking keys in filenames')
    for input_file in input_files:
        for ref in refs.keys():
            if ref in input_file:
                refs[ref].append(input_file)

    # numtimes = len(refs[reference_ids[0]])
    logger.info(f"Post init refs: {refs}")

    numtimes = len(refs[list(refs.keys())[0]])
    logger.info('Number of time slots opened: {0}'.format(numtimes))

    # numbeams = np.zeros((len(reference_ids),numtimes), dtype=np.int)
    numbeams = np.zeros((len(refs.keys()),numtimes), dtype=np.int)
    # ref_ants = np.empty(len(reference_ids), dtype='S32')
    ref_ants = np.empty(len(refs.keys()), dtype='S32')

    for i,ref in enumerate(refs.keys()):
        for t in range(numtimes):
            logger.info('Loading {0}'.format(refs[ref][t]))
            hd_ = uvhol.read_uvhol_file(refs[ref][t])[0]
            numbeams[i,t] = len(hd_.l_rad)
            ref_ants[i] = hd_.ref_ants[0]

    template_idx = np.unravel_index(numbeams.argmax(), numbeams.shape)[0]
    # template_ref = reference_ids[template_idx]
    template_ref = list(refs.keys())[template_idx]

    logger.info('Maximum number of beams in all uvhol files: {0}'.format(int(numbeams.max())))

    template = -1*np.ma.ones((numtimes, len(refs.keys()), int(numbeams.max()), 6))
    weights = np.ones((numtimes, int(numbeams.max())))

    for t in range(numtimes):
        logger.info(refs[template_ref][t])

        hd_ = uvhol.read_uvhol_file(refs[template_ref][t])[0]
        len_hd = len(hd_.l_rad)

        if len_hd > 0:

            logger.info('Number of beams in holog file: {0}'.format(len_hd))

            template[t,template_idx,:len_hd,0] = hd_.l_rad
            template[t,template_idx,:len_hd,1] = hd_.m_rad
            template[t,template_idx,:len_hd,2] = hd_.vis.real
            template[t,template_idx,:len_hd,3] = hd_.vis.imag
            template[t,template_idx,:len_hd,4] = hd_.sigma_vis.real
            template[t,template_idx,:len_hd,5] = hd_.sigma_vis.imag

            template[t,template_idx,len_hd:,2] = np.nan
            template[t,template_idx,len_hd:,3] = np.nan
            template[t,template_idx,len_hd:,4] = np.nan
            template[t,template_idx,len_hd:,5] = np.nan

        else:

            template[t,template_idx,:,2:] = np.nan
            weights[t,len_hd:] -= 1

        for i,ref in enumerate(refs.keys()):
            logger.info('Working on: {0}'.format(ref))
            if ref != template_ref:

                hd_ = uvhol.read_uvhol_file(refs[ref][t])[0]

                for b,(l,m) in enumerate(zip(template[t,template_idx,:,0], template[t,template_idx,:,1])):

                    mask = (hd_.l_rad == l) & (hd_.m_rad == m)

                    if mask.astype(np.float).sum() == 1:

                        weights[t,b] += 1
                        template[t,i,b,0] = hd_.l_rad[mask]
                        template[t,i,b,1] = hd_.m_rad[mask]
                        template[t,i,b,2] = np.ma.masked_invalid(hd_.vis.real[mask])
                        template[t,i,b,3] = np.ma.masked_invalid(hd_.vis.imag[mask])
                        template[t,i,b,4] = np.ma.masked_invalid(hd_.sigma_vis.real[mask])
                        template[t,i,b,5] = np.ma.masked_invalid(hd_.sigma_vis.imag[mask])

                    else:

                        template[t,i,b,0] = 0.
                        template[t,i,b,1] = 0.
                        template[t,i,b,2] = np.nan
                        template[t,i,b,3] = np.nan
                        template[t,i,b,4] = np.nan
                        template[t,i,b,5] = np.nan

        template = np.ma.masked_invalid(template)

        weights_re = np.ma.power(template[t,:,:,4], -2.)
        weights_im = np.ma.power(template[t,:,:,5], -2.)

        vis_avg_re = np.ma.average(template[t,:,:,2], axis=0, weights=weights_re)
        vis_avg_im = np.ma.average(template[t,:,:,3], axis=0, weights=weights_im)

        sig_vis_avg_re = 1./np.sum(weights_re, axis=0)*np.sqrt(np.sum(np.power(weights_re*template[t,:,:,4], 2.), axis=0))
        sig_vis_avg_im = 1./np.sum(weights_im, axis=0)*np.sqrt(np.sum(np.power(weights_im*template[t,:,:,5], 2.), axis=0))

        header = to_uvhol.make_uvhol_header(ref_ants,
                                            hd_.antenna.split(','),
                                            pol, hd_.freq_hz*1e-9,
                                            0, 0, 0, 3)
        footer = to_uvhol.make_uvhol_footer(len(refs.keys())*64)

        logger.info('Saving file: {0}'.format(output(t)))
        np.savetxt(output(t), np.c_[template[t,template_idx,:,0],
                                    template[t,template_idx,:,1],
                                    vis_avg_re.filled(np.nan),
                                    vis_avg_im.filled(np.nan),
                                    sig_vis_avg_re.filled(-1.),
                                    sig_vis_avg_im.filled(-1.)],
                   header=header, footer=footer, comments='')

    return template

def time_average_vis(target, output, target_time_res=0, weighted=True, time_res_now=1):
    """
    """

    start_time = datetime.now()

    logger = logging.getLogger(__name__)

    logger.info('Working on target file: {0}'.format(target))
    fth5 = h5py.File(target, 'r')

    ht = fth5.attrs
    beams = list(fth5.keys())
    # dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
    # st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
    # ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag').value)
    dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data'), dtype=np.complex64)
    st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma'), dtype=np.complex64)
    ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag'))

    # freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq').value for b in beams])
    freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq') for b in beams])

    # Where are we looking at?
    beam_info = fth5['/{0}/POINTING'.format(beams[0])]

    logger.info('Applying masks to data and errors.')
    dt = np.ma.masked_where(ft, dt)
    st = np.ma.masked_where(ft, st)

    logger.info('Generating time axis.')
    ntimes = dt.shape[0]
    try:
        time_res_now = ht['integration_time_sample']
    except KeyError:
        time_res_now = time_res_now
        logger.info('Key integration_time_sample not found in header.')
        logger.info('Will use a value of: {0}'.format(time_res_now))
    averaging_values = np.array(list(misc.factors(ntimes)))
    averaging_factor = averaging_values[np.argmin(abs(int(round(target_time_res/time_res_now)) - averaging_values))]
    time_res_avg = time_res_now*averaging_factor
    time_axis_now = np.arange(0, ntimes*time_res_now, time_res_now)
    logger.info('Input time resolution: {0}'.format(time_res_now))
    logger.info('Output time resolution: {0}'.format(time_res_avg))

    logger.info('Generating weight vectors.')
    if weighted:
        weight_real = np.power((st.real).reshape((-1,averaging_factor,)+st.shape[1:]), -2.)
        weight_imag = np.power((st.imag).reshape((-1,averaging_factor,)+st.shape[1:]), -2.)
    else:
        weight_real = np.ones((st).reshape((-1,averaging_factor,)+st.shape[1:]).shape, dtype=np.complex64)
        weight_imag = 1.j*weight_real
    weight = weight_real + 1.j*weight_imag

    logger.info('Averaging data by a factor of {0}.'.format(averaging_factor))
    dt_ = (dt).reshape((-1,averaging_factor,)+dt.shape[1:])
    dt_avg = np.ma.average(dt_, axis=1, weights=weight)
    time_axis_avg = np.average((time_axis_now).reshape((-1,averaging_factor,)), axis=1)

    logger.info('Output data has {0} time slots.'.format(len(time_axis_avg)))

    logger.info('Generating weighted error.')
    n_points = averaging_factor
    wst = csqrt(n_points/(n_points - 1.)*
                np.ma.sum(weight*np.ma.power(dt_ - dt_avg[:,np.newaxis], 2.), axis=1)
                /np.ma.sum(weight, axis=1))
    wst = np.ma.masked_where(dt_avg.mask, wst)

    if output != '':
        logger.info('Saving time averaged data to: {0}'.format(output))
        save_hdf5(output, dt_avg, wst, freq, beam_info, ht, time_res_avg)

    logger.info('Step run time: {0}'.format(datetime.now() - start_time))

def save_hdf5(output, data, sigma, freq, beam_info, target_head, time_resolution):
    """
    """

    # Create file.
    f = h5py.File(output, "w")

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
    f.attrs['reference_antennas'] = target_head['reference_antennas']
    f.attrs['target_antennas'] = target_head['target_antennas']
    f.attrs['mjd_start'] = target_head['mjd_start']
    f.attrs['mjd_end'] = target_head['mjd_end']
    f.attrs['utc_start'] = target_head['utc_start']
    f.attrs['utc_end'] = target_head['utc_end']
    f.attrs['time_samples'] = target_head['time_samples']
    f.attrs['integration_time_sample'] = time_resolution
    #f.attrs['sample_rate_hz'] = target_head['sample_rate_hz']

    # Close file. 
    f.close()


