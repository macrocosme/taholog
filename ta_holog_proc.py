#!/usr/bin/env python

"""
This script runs all the steps required to process raw voltage data into complex valued beam maps.
It assumes that the data will be located in trunk_dir, and each observation (target and reference)
will be in a separate directory named as their sasid.
"""

import re
import sys
import os
import glob
import logging
import numpy as np
import multiprocessing as mp

from cProfile import Profile
from pstats import SortKey, Stats

import matplotlib
matplotlib.use('Agg')

from taholog import procs

from taholog import misc, to_freq, xcorr, gencal, applycal, \
                    clip, average, to_uvhol, solve, order, plot



def run_pipeline(params, verbose=False):
    """
    Runs the steps required to process holography data (as recorded by COBALT).
    """
   
    logger = logging.getLogger(__name__)
 
    # Set up logger.
    mp.log_to_stderr(logging.DEBUG)
    
    steps = params['steps']
    logger.info('Will run the following steps: {0}'.format(steps))
    
    # Unpack some kwargs.
    target_id = params['target_id']
    reference_ids = params['reference_ids']
    trunk_dir = params['trunk_dir']
    output_dir = params['output_dir']
    cs_str = params['cs_str']
    debug = params['debug']

    # Setup polarization mapping
    num_pol = params['to_freq_num_pol']
    mfp = '0,1,2,3'
    mfpl = mfp.split(',')
    polmap = np.array(mfpl, dtype=int).reshape(num_pol, len(mfpl)//num_pol)

    cwd = os.getcwd()

    # Go to where the data is.
    os.chdir(trunk_dir)

    if 'to_freq' in steps:
        with Profile() as profile:
            print(f"{procs._to_freq(trunk_dir, output_dir, cs_str, target_id, reference_ids, params, num_pol, polmap, logger, debug, verbose) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    os.chdir(trunk_dir)

    logger.info('Checking that there are enough output files.')
    if verbose: 
        print ('Checking that there are enough output files.')
    for ref in reference_ids: 
        all_files = glob.glob(f'{trunk_dir}{ref}/{cs_str}/*spw*.h5')
        if len(all_files) != params['spws']:
            logger.error('The number of channelized files is different than expected for reference: {0}'.format(ref))
            logger.error('Will not continue.')
            sys.exit(1)   
    all_files = glob.glob(f'{trunk_dir}{target_id}/{cs_str}/*spw*.h5')
    if len(all_files) != params['target_beams']*params['spws']:
        logger.error('The number of channelized files is different than expected for reference: {0}'.format(ref))
        logger.error('Will not continue.')
        sys.exit(1)
 
    logger.info('The number of channelized files is as expected. Continuing...')


    xcorr_dt = params['xcorr_dt']
    if 'xcorr' in steps:
        with Profile() as profile:
            print(f"{procs._xcorr(trunk_dir, cs_str, reference_ids, target_id, xcorr_dt, params, debug, verbose) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'plot_beam' in steps:
        with Profile() as profile:
            print(f"{procs._plot_beam(params, verbose) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'gencal' in steps:
        with Profile() as profile:
            print(f"{procs._gencal(trunk_dir, target_id, xcorr_dt, reference_ids, params, debug, verbose) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'applycal' in steps:
        with Profile() as profile:
            print(f"{procs._applycal(trunk_dir, target_id, xcorr_dt, params, reference_ids, debug, verbose) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'clip' in steps:
        with Profile() as profile:
            print(f"{procs._clip(trunk_dir, target_id, reference_ids, xcorr_dt, params, debug) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    average_t_dt = params['average_t_dt']

    if 'average_t' in steps:
        with Profile() as profile:
            print(f"{procs._average_t(trunk_dir, target_id, average_t_dt, reference_ids, xcorr_dt, params, debug) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'to_uvhol' in steps:
        with Profile() as profile:
            print(f"{procs._to_uvhol(trunk_dir, target_id, xcorr_dt, average_t_dt, reference_ids, params, debug) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    if 'average_uvhol' in steps:
        with Profile() as profile:
            print(f"{procs._average_uvhol(trunk_dir, target_id, xcorr_dt, average_t_dt, params, reference_ids) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
        
    if 'solve_uvhol' in steps:
        with Profile() as profile:
            print(f"{procs._solve_uvhol(trunk_dir, target_id, xcorr_dt, average_t_dt, params, logger) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
        
    if 'order_sols' in steps:
        with Profile() as profile:
            print(f"{procs._order_sols(trunk_dir, target_id, xcorr_dt, average_t_dt, params) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
        
    if 'plot_report' in steps:
        with Profile() as profile:
            print(f"{procs._plot_report(trunk_dir, target_id, xcorr_dt, average_t_dt, params) = }")
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )

    logger.info('Done with steps: {0}'.format(steps))
