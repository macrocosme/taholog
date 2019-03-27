r'''
'''

import os
import numpy as np

import antennafield

def afield_rotation_matrix(antenna_field_list, field_name):
    r'''
    Extract a rotation matrix from a list of antenna field entries.

    **Parameters**

    antenna_field_list : list of entries from an AntennaField.conf
        Items as read by the antennafield module.

    field_name : string
        Either 'LBA', 'HBA', 'HBA0' or 'HBA1'

    **Returns**

    A 2D numpy.array of floats containing the pqr-to-itrs rotation
    matrix for the antenna field.


    **Examples**

    >>> afield_items = antennafield.parse_antenna_field('antenna-fields/CS002-AntennaField.conf')
    >>> afield_rotation_matrix(afield_items, 'HBA0')
    array([[ -1.19595000e-01,  -7.91954000e-01,   5.98753000e-01],
           [  9.92823000e-01,  -9.54190000e-02,   7.20990000e-02],
           [  3.30000000e-05,   6.03078000e-01,   7.97682000e-01]])
    '''
    return [item for item in antenna_field_list
            if item.field_name == field_name and \
            item.__class__.__name__ == 'RotationMatrix'][0].rotation_matrix

def afield_phase_centre_itrs(antenna_field_list, field_name):
    r'''
    Extract an ITRS field centre from a list of antenna field entries.

    **Parameters**

    antenna_field_list : list of entries from an AntennaField.conf
        Items as read by the antennafield module.

    field_name : string
        Either 'LBA', 'HBA', 'HBA0' or 'HBA1'

    **Returns**

    A numpy.array of floats containing the 3D ITRS phase centre of
    the antenna field.


    **Examples**

    >>> afield_items = antennafield.parse_antenna_field('antenna-fields/CS302-AntennaField.conf')
    >>> afield_phase_centre_itrs(afield_items, 'HBA0')
    array([ 3827973.18291 ,   459728.671216,  5063975.329   ])
    '''
    return [item for item in antenna_field_list
            if item.field_name == field_name and \
            item.__class__.__name__ == 'AntennaField'][0].centre

def station_offsets_pqr(station_list):
    r'''
    Computes the pqr coordinates of the station_list with respect to center_station.
    Returns (N,3) shape array with the (p,q,r) coordinates of the N stations in the station_list.
    '''

    afddir = os.path.dirname(antennafield.__file__)

    # Prepare station locations.
    antennas = np.array(station_list.split(','))
    nant = len(antennas)

    teo = np.zeros((nant), dtype=object)
    afc = np.zeros((nant, 3))

    # Load the array center: CS002.
    center = antennafield.parse_antenna_field('{0}/../data/CS002-AntennaField.conf'.format(afddir)) # Same center for LBA and HBA.
    matrix = afield_rotation_matrix(center, 'LBA') # Same rotation for HBA and LBA.

    # Check if it is an LBA or HBA data set.
    # This will need another condition if remote or international stations are used.
    if 'HBA' in antennas[0]:

        _c = -4

        # HBA core stations come in ears.
        afc0 = np.zeros((nant//2, 3))
        afc1 = np.zeros((nant//2, 3))

        for i,s in enumerate(antennas[::2]):
            teo[i] = antennafield.parse_antenna_field('{0}/../data/{1}-AntennaField.conf'.format(afddir, s[:_c]))
            afc[i] = afield_phase_centre_itrs(teo[i], 'HBA')
            afc0[i] = afield_phase_centre_itrs(teo[i], 'HBA0')
            afc1[i] = afield_phase_centre_itrs(teo[i], 'HBA1')

        afc0_off = afield_phase_centre_itrs(center, 'LBA') - afc0
        afc1_off = afield_phase_centre_itrs(center, 'LBA') - afc1

        ants0 = np.dot(matrix.T, afc0_off.T).T
        ants1 = np.dot(matrix.T, afc1_off.T).T

        pqr = np.zeros((nant,3))
        pqr[::2] = ants0
        pqr[1::2] = ants1

    elif 'LBA' in antennas[0]:

        _c = -3

        for i,s in enumerate(antennas):
            teo[i] = antennafield.parse_antenna_field('{0}/../data/{1}-AntennaField.conf'.format(afddir, s[:_c]))
            afc[i] = afield_phase_centre_itrs(teo[i], 'LBA')

        afc_off = afield_phase_centre_itrs(center, 'LBA') - afc

        pqr = np.dot(matrix.T, afc_off.T).T

    return pqr
