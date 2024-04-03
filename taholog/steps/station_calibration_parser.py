#!/usr/bin/env python
"""
Updates an input parset file with the solutions specified in property_map.
The time delay corrections are loaded from solutions_lba_{x,y} for the LBA
and solutions_hba_{x,y} for the HBA.
Only the lines for the subarrays in sols_subarr and for the filters in sols_filter will be updated.
For example, if you only want to udpate the LBA stations between 30 MHz and 70 MHz, set:
sols_filter = ['LBA_30_70']
"""

import pickle

_sign = 1

_keys = {
         'PIC' : 0,
         'Core': 1,
         'Station': 2,
         'SubArray': 3,
         'Filter': 4,
         'Property': 5,
         'Polarization': 6
         }

def update_parset(solutions, old_parset, new_parset, subarray, filters):
    """
    """

    
    
    with open(old_parset, 'r') as infile, open(new_parset, 'w') as outfile:

        for line in infile:

            desc,val = line.rstrip().split('=')
            desc_ = desc.rstrip().split('.')

            if (desc_[_keys['Station']] in solutions['sols_lba_x'].keys() \
                or desc_[_keys['Station']] in solutions['sols_hba_x'].keys()) \
                and desc_[_keys['SubArray']] in subarray and desc_[_keys['Filter']] in filters:

                if 'LBA' in desc_[_keys['Station']]:
                    solsx = solutions['sols_lba_x']
                    solsy = solutions['sols_lba_y']
                elif 'HBA' in desc_[_keys['Station']]:
                    solsx = solutions['sols_hba_x']
                    solsy = solutions['sols_hba_y']

                if desc_[_keys['Polarization']] == 'X':
                    val_ = solsx[desc_[_keys['Station']]][property_map[desc_[_keys['Property']]]][0,0]
                elif desc_[_keys['Polarization']] == 'Y':
                    val_ = solsy[desc_[_keys['Station']]][property_map[desc_[_keys['Property']]]][0,0]

                newval = float(val) + _sign*val_

                newline = '{0:>61}= {1}\n'.format(desc, '{0:.6e}'.format(newval))

                outfile.write(newline)

            """
            For the HBA_JOINED mode use the mean of the time delays between the two HBA fields.
            """
            elif desc_[_keys['SubArray']] == 'HBA_JOINED' and \
                 desc_[_keys['Station']] + '0' in solutions['sols_hba_x'].keys() and \
                 desc_[_keys['SubArray']] in subarray and desc_[_keys['Filter']] in filters:

                stat0 = '{0}0'.format(desc_[_keys['Station']])
                stat1 = '{0}1'.format(desc_[_keys['Station']])

                if desc_[keys['Polarization']] == 'X':
                    sol0 = solutions['sols_hba_x'][stat0][property_map[desc_[_keys['Property']]]][0,0]
                    sol1 = solutions['sols_hba_x'][stat1][property_map[desc_[_keys['Property']]]][0,0]
                else:
                    sol0 = solutions['sols_hba_y'][stat0][property_map[desc_[_keys['Property']]]][0,0]
                    sol1 = solutions['sols_hba_y'][stat1][property_map[desc_[_keys['Property']]]][0,0]

                val_ = (sol0 + sol1)/2.

                newval = float(val) + _sign*val_

                newline = '{0:>61}= {1}\n'.format(desc, '{0:.6e}'.format(newval))

                outfile.write(newline)


            else:

                outfile.write(line)


if __name__ == '__main__':
    
    input_file = 'StationCalibration.parset'
    output_file = 'StationCalibrationAll_20190121.parset'
    solutions_lba_x = '/data/scratch/holography/tutorial/data/L658168_xcorr/avg0.4_cal_clip_avg6000_XX_sols.pickle'
    solutions_lba_y = '/data/scratch/holography/tutorial/data/L658168_xcorr/avg0.4_cal_clip_avg6000_YY_sols.pickle'
    solutions_hba_x = '/data/scratch/holography/tutorial/data/L658168_xcorr/avg0.4_cal_clip_avg6000_XX_sols.pickle'
    solutions_hba_y = '/data/scratch/holography/tutorial/data/L658168_xcorr/avg0.4_cal_clip_avg6000_YY_sols.pickle'
    subarray = ['HBA_ZERO', 'HBA_ONE', 'HBA_DUAL', 'HBA_DUAL_INNER', 'LBA_OUTER', 'LBA_INNER', 'LBA_SPARSE_ODD', 'LBA_SPARSE_EVEN'] 
    filters = ['HBA_110_190', 'HBA_170_230', 'HBA_210_250', 'LBA_10_70', 'LBA_30_70', 'LBA_10_90', 'LBA_30_90']

    # Which solutions to use to update the delay and phase0.
    """
    tau: time delay fitted with a non-zero phase 0 offset.
    tau0: phase 0 offset.
    tau_fix: time delay fitted with a zero phase 0 offset.
    tau0_fix: phase 0 offset, set to zero when fitting.
    """
    property_map = {'delay': 'tau_fix',
                    'phase0': 'tau0_fix'}
     
    sols_lba_x = pickle.load(open(solutions_lba_x, "rb"))
    sols_lba_y = pickle.load(open(solutions_lba_y, "rb"))
    sols_hba_x = pickle.load(open(solutions_hba_x, "rb"))
    sols_hba_y = pickle.load(open(solutions_hba_y, "rb"))
    
    solutions = {'sols_lba_x': sols_lba_x,
                 'sols_lba_x': sols_lba_y,
                 'sols_hba_x': sols_hba_x,
                 'sols_hba_x': sols_hba_y,}

    update_parset(solutions, input_file, output_file, subarray, filters, property_map)
