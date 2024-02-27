r'''
Coordinate handling functions.
'''

import os
import numpy as np

import astropy.units as u

from astropy.coordinates import SkyCoord

from . import stations, antfield as antennafield

from astropy.utils import iers
iers.conf.auto_max_age = None
# Always try to download the most recent IERS tables.
#from astropy.utils.data import download_file
#from astropy.utils import iers
#iers.IERS.iers_table = iers.IERS_A.open(download_file(iers.IERS_A_URL, cache=True))

def itrs_from_icrs(icrs_position_rad, obstime):
    r'''
    Takes an array RA/Dec ICRS positions in radians and converts those
    to geocentric ITRF unit vectors.
    
    **Parameters**

    icrs_position_rad : numpy array of floats (ra_rad, dec_rad)
        The ICRS position to convert. May also be an array of RA/DEC
        pairs.

    obstime : astropy.time.Time or string
        When a string is provided, it is assumed to be readable by an
        astropy.time.Time instance.

    **Returns**

    An array containing the geocentric cartesian ITRS unit vectors
    corresponding to the icrs_position_rad at obstime.

    **Examples**

    >>> itrs_from_icrs((array([358.7928571, 89.91033405])*u.deg).to(u.rad),
    ...                obstime='2015-01-01 00:00:00')
    array([[  1.25385143e-12,   6.93857340e-12,   1.00000000e+00]])
    >>> itrs_from_icrs((array([[358.7928571, 89.91033405],
    ...                        [90,-20],
    ...                        [30, 60]])*u.deg).to(u.rad),
    ...                        obstime='2015-01-01 00:00:00')
    array([[  1.25385143e-12,   6.93857340e-12,   1.00000000e+00],
           [  9.24930853e-01,  -1.65788673e-01,  -3.42077526e-01],
           [  1.70157684e-01,  -4.68937638e-01,   8.66685557e-01]])

    '''
    ra, dec = np.array(icrs_position_rad, dtype='float64').T*u.rad
    icrs = SkyCoord(ra, dec,frame='icrs',
                    obstime=obstime, equinox='J2000')
    itrs = icrs.itrs
    return np.array([itrs.x, itrs.y, itrs.z], dtype=np.float64).T

def pqr_from_icrs(icrs_rad, obstime, pqr_to_itrs_matrix):
    r'''
    Compute geocentric station-local PQR coordinates of a certain ICRS
    direction. Geocentric here means that parallax between the centre
    of the Earth and the station's phase centre is not taken into
    account.

    **Parameters**

    icrs_rad : numpy.array
        An ICRS position in radians. Either one vector of length 2, or
        an array of shape (N, 2) containing N ICRS [RA, Dec] pairs.

    obstime : string or astropy.time.Time
        The date/time of observation.

    pqr_to_itrs_matrix : 3x3 numpy.array
        The rotation matrix that is used to convert a direction in the
        station's PQR system into an ITRS direction. This matrix is
        found in the /opt/lofar/etc/AntennaField.conf files at the
        stations. These files are also found in the antenna-fields/
        directory of this project.

    **Returns**
    
    A numpy.array instance with shape (3,) or (N, 3) containing the
    ITRS directions.

    **Example**
    
    >>> import astropy
    >>> core_pqr_itrs_mat = array([[ -1.19595000e-01,  -7.91954000e-01,   5.98753000e-01],
    ...                            [  9.92823000e-01,  -9.54190000e-02,   7.20990000e-02],
    ...                            [  3.30000000e-05,   6.03078000e-01,   7.97682000e-01]],
    ...                           dtype=float64)
    >>> obstime='2015-06-19 13:50:00'
    >>> target_3c196 = SkyCoord('08h13m36.0561s', '+48d13m02.636s', frame='icrs')
    >>> target_icrs_rad = array([target_3c196.icrs.ra.rad, target_3c196.icrs.dec.rad])
    >>> pqr_from_icrs(target_icrs_rad, obstime, core_pqr_itrs_mat)
    array([ 0.02131259, -0.08235505,  0.99637513])
    >>> pqr_from_icrs(array([target_icrs_rad, target_icrs_rad]),
    ...                      obstime, core_pqr_itrs_mat)
    array([[ 0.02131259, -0.08235505,  0.99637513],
           [ 0.02131259, -0.08235505,  0.99637513]])
    >>> pqr_from_icrs(target_icrs_rad, astropy.time.Time(obstime)+7*u.minute +18*u.second,
    ...               core_pqr_itrs_mat)
    array([  1.49721138e-05,  -8.26257777e-02,   9.96580636e-01])

    '''

    return np.dot(pqr_to_itrs_matrix.T, itrs_from_icrs(icrs_rad, obstime).T).T.squeeze()

def radec_to_lm(radec, mean_time):
    r'''
    Converts the (RA,DEC) coordinates to (l,m).

    **Parameters**
d
    radec: numpy.array
        Array of shape (N,2) with RA DEC coordinates in radians.
    mean_time: astropy.time.Time
        Time of the observation. Used to convert from RA,DEC to l,m.

    **Returns**
    '''

    afddir = os.path.dirname(antennafield.__file__)

    # Load the array center: CS002.
    # center = antennafield.parse_antenna_field('{0}/AntennaFields/CS002-AntennaField.conf'.format(afddir)) # Same center for LBA and HBA.

    # The coordinates should be rotated to the PQR coordinate system.
    # The rotation matrix is the same for both LBA and HBAs.
    # rotmat = stations.afield_rotation_matrix(center, 'LBA')

    center = antennafield.from_file('{0}/AntennaFields/CS002-AntennaField.conf'.format(afddir)) # Same center for LBA and HBA.
    rotmat = np.array(center['ROTATION_MATRIX']['LBA'])

    mean_time.format = 'iso'

    # Convert to PQR (dipole frame of reference).
    pqr_dirs = pqr_from_icrs(radec, mean_time.utc.value, rotmat)

    # Offset with respect to central beam.
    pierce_points_lm_rad = pqr_dirs[0,0:2][np.newaxis,:] - pqr_dirs[:,0:2]

    return pierce_points_lm_rad
