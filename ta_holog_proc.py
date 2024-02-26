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

def check_ends_with_slash(path):
    if path[-1] != '/':
    path = path + '/'
    return path

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
    trunk_dir = check_ends_with_slash(params['trunk_dir'])
    output_dir = check_ends_with_slash(params['output_dir'])
    cs_str = params['cs_str']
    debug = params['debug']

    # Setup polarization mapping
    num_pol = params['to_freq_num_pol']
    mfp = '0,1,2,3'
    mfpl = mfp.split(',')
    polmap = np.array(mfpl, dtype=int).reshape(num_pol, len(mfpl)//num_pol)

    cwd = os.getcwd()

    # Go to where the data is.
    # os.chdir(trunk_dir)

    if 'to_freq' in steps:
        procs._to_freq(trunk_dir, output_dir, cs_str, target_id, reference_ids, params, num_pol, polmap, logger, debug, verbose)


    # os.chdir(trunk_dir)

    logger.info('Checking that there are enough output files.')
    if verbose: 
        print ('Checking that there are enough output files.')
    for ref in reference_ids: 
        all_files = glob.glob(f'{output_dir}{ref}/{cs_str}/*spw*.h5')
        if len(all_files) != params['spws']:
            logger.error('The number of channelized files is different than expected for reference: {0}'.format(ref))
            logger.error('Will not continue.')
            sys.exit(1)   
    all_files = glob.glob(f'{output_dir}{target_id}/{cs_str}/*spw*.h5')
    if len(all_files) != params['target_beams']*params['spws']:
        logger.error('The number of channelized files is different than expected for reference: {0}'.format(ref))
        logger.error('Will not continue.')
        sys.exit(1)
 
    logger.info('The number of channelized files is as expected. Continuing...')


    xcorr_dt = params['xcorr_dt']
    if 'xcorr' in steps:
        procs._xcorr(output_dir, cs_str, target_id, reference_ids, params, debug, verbose)

    if 'plot_beam' in steps:
        procs._plot_beam(params, debug, verbose)

    if 'gencal' in steps:
        procs._gencal(output_dir, target_id, xcorr_dt, reference_ids, params, debug, verbose)

    if 'applycal' in steps:
        procs._applycal(output_dir, target_id, xcorr_dt, params, reference_ids, debug, verbose)

    if 'clip' in steps:
        procs._clip(output_dir, target_id, reference_ids, xcorr_dt, params, debug)

    average_t_dt = params['average_t_dt']

    if 'average_t' in steps:
        procs._average_t(output_dir, target_id, average_t_dt, reference_ids, xcorr_dt, params, debug)

    if 'to_uvhol' in steps:
        procs._to_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, reference_ids, params, debug)

    if 'average_uvhol' in steps:
        procs._average_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, reference_ids)
        
    if 'solve_uvhol' in steps:
        procs._solve_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, logger)
        
    if 'order_sols' in steps:
        procs._order_sols(output_dir, target_id, xcorr_dt, average_t_dt, params)
        
    if 'plot_report' in steps:
        procs._plot_report(output_dir, target_id, xcorr_dt, average_t_dt, pol, params)

    logger.info('Done with steps: {0}'.format(steps))
