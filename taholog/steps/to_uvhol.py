"""
"""

import h5py
import logging
import numpy as np

from datetime import datetime

from astropy import time
from . import coordinates

def make_uvhol_header(ref_antennas, tgt_antennas, stokes, frequency_hz, if_num, t_start, t_end, numpoints):
    """
    """

    try:
        ref_antennas = [rfa.decode("utf-8") for rfa in ref_antennas]
        tgt_antennas = [tga.decode("utf-8") for tga in tgt_antennas]
        # ref_antennas = [rfa for rfa in ref_antennas]
        # tgt_antennas = [tga for tga in tgt_antennas]
    except AttributeError:
        pass

    print ('ref_antennas', ref_antennas)
    print ('tgt_antennas', tgt_antennas)
    head = "#! RefAnt = {0} Antenna = {1} Stokes = {2} Freq = {3}\n"\
                       "#! MINsamp = 3 Npoint = 5\n"\
                       "#! IFnumber = {4} Channel = 1\n"\
                       "#! TimeRange = {5}, {6}, {7}".format(','.join(ref_antennas),
                                                             ','.join(tgt_antennas),
                                                             stokes,
                                                             frequency_hz,
                                                             if_num,
                                                             t_start, t_end,
                                                             numpoints)

    return head

def make_uvhol_footer(numsamples):
    """
    """

    footer = '#! Average number samples per point = {0}'.format(numsamples)

    return footer

def main(file_list, output_base, correlations='XX,XY,YX,YY'):
    """
    """

    start_time = datetime.now()

    logger = logging.getLogger(__name__)

    corr = np.array(correlations.split(',')).reshape(2,2)

    logger.info('Will process the files: {0}'.format(file_list))

    fth5 = h5py.File(file_list[0], 'r')
    ht = fth5.attrs
    beams = list(fth5.keys())

    # Load the central beam data.
    logger.info('Loading data from first file to generate array.')
    # dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
    dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data'), dtype=np.complex64)

    ntimes = dt.shape[0]

    xcorr = np.zeros((len(file_list),)+(ntimes,)+(2,2), dtype=np.complex64)
    sigma = np.zeros((len(file_list),)+(ntimes,)+(2,2), dtype=np.complex64)
    radec = np.zeros((len(file_list),2), dtype=np.float64)

    logger.info('Reading files and loading data...')
    for i,filename in enumerate(file_list):

        fth5 = h5py.File(filename, 'r')
        # dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
        # st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
        dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data'), dtype=np.complex64)
        st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma'), dtype=np.complex64)
        xcorr[i] = dt
        sigma[i] = st

        # radec[i] = np.deg2rad(fth5['/{0}/POINTING'.format(beams[0])]['beam_pos_radec'].value)
        radec[i] = np.deg2rad(fth5['/{0}/POINTING'.format(beams[0])]['beam_pos_radec'])

    logger.info('Making time vector.')
    # start_t = time.Time(ht['utc_start'].decode('utf-8'), scale='utc')
    # end_t = time.Time(ht['utc_end'].decode('utf-8'), scale='utc')
    start_t = time.Time(ht['utc_start'], scale='utc')
    end_t = time.Time(ht['utc_end'], scale='utc')
    obs_t = (end_t - start_t)
    int_t = obs_t/ntimes
    mean_time = start_t + obs_t/2.
    mean_time.format = 'iso'
    time_vec = time.Time(np.linspace(start_t.mjd, end_t.mjd, ntimes), scale='utc', format='mjd')
    time_vec.format = 'iso'
    time_vec += int_t/2.

    logger.info('Converting beam locations to (l,m) coordinates.')
    lm_coords = np.empty((ntimes,len(file_list),2))
    for i in range(ntimes):
        lm_coords[i] = coordinates.radec_to_lm(radec, time_vec[i])

    numpoints = ht['time_samples']/ntimes

    # Write to uvhol files.
    logger.info('Will start writing uvhol files.')
    for t in range(xcorr.shape[1]):
        for i,_c in enumerate(corr):
            for j,c in enumerate(_c):

                output = '{0}_{1}_t{2}.uvhol'.format(output_base, c, t)

                header = make_uvhol_header(ht['reference_antennas'],
                                           ht['target_antennas'],
                                           c, ht['frequency_hz']*1e-9,
                                           0, (time_vec[t] - int_t/2.).mjd,
                                           (time_vec[t] + int_t/2.).mjd, 64)
                footer = make_uvhol_footer(numpoints)

                logger.info('Saving file: {0}'.format(output))
                np.savetxt(output, np.c_[lm_coords[t], xcorr.real[:,t,i,j], xcorr.imag[:,t,i,j],
                                         abs(sigma.real[:,t,i,j]), abs(sigma.imag[:,t,i,j])],
                           header=header, footer=footer, comments='')

    logger.info('Step run time: {0}'.format(datetime.now() - start_time))


