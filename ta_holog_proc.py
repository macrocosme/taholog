#!/usr/bin/env python

"""
This script runs all the steps required to process raw voltage data into complex valued beam maps.
It assumes that the data will be located in trunk_dir, and each observation (target and reference)
will be in a separate directory named as their sasid.

Edit (May 2024): reference stations observations can now also be included as individual beams under one sasid. 
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

from taholog.testing.checks import check_channelized_file_count, check_correlated_file_count

import matplotlib
matplotlib.use('Agg')

import procs

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
    parallel = params['parallel']

    # Setup polarization mapping
    num_pol = params['to_freq_num_pol']
    mfp = '0,1,2,3'
    mfpl = mfp.split(',')
    polmap = np.array(mfpl, dtype=int).reshape(num_pol, len(mfpl)//num_pol)

    cwd = os.getcwd()

    # Go to where the data is.
    os.chdir(trunk_dir)

    if 'to_freq' in steps:
        procs._to_freq(trunk_dir, output_dir, cs_str, target_id, reference_ids, params, num_pol, polmap, logger, parallel, verbose)
    check_channelized_file_count(logger, output_dir, target_id, reference_ids, params, verbose)
    # Should also add a similar function as for xcorr where in case of missing files, re-run them and continue.
    # Also, finding the reason of the failed jobs would render that extra step redundant.
    # So far, to_freq seems to work just fine with (--parallel=True --no-use_numba --no-use_gpu --no-use_pyfftw --to_disk)

    xcorr_dt = params['xcorr_dt']
    logger.info(f'xcorr_dt: {xcorr_dt}')
    if 'xcorr' in steps:
        procs._xcorr(output_dir, cs_str, reference_ids, target_id, xcorr_dt, params, verbose)
    
    _continue, missing = check_correlated_file_count(logger, output_dir, target_id, reference_ids, xcorr_dt, params, verbose, return_missing=True)
    # TODO: Should consider a way to exit the loop in case infini-loop is in action...
    while not _continue:
        logger.info(f're-running xcorr on {missing}')
        # if not parallel:
        for m in missing:
            refid, ref_beam, spw, ibm = m
            xcorr_output_dir = f'{output_dir}{target_id}_xcorr'
            procs._redo_missing_xcorr(output_dir, xcorr_output_dir, target_id, params, xcorr_dt, refid, ref_beam, spw, ibm)
        # TODO: Parallel processing for these missing files would make sense, but doesn't currently work. I have not investigated.
        # else:
        #     pool = mp.Pool(processes=params['xcorr_cpus'])
        #     for m in missing:
        #         refid, ref_beam, spw, ibm = m
        #         xcorr_output_dir = f'{output_dir}{target_id}_xcorr'
        #         pool.apply_async(procs._redo_missing_xcorr, args=(output_dir, xcorr_output_dir, target_id, params, xcorr_dt, refid, ref_beam, spw, ibm))
        #     pool.close()
        #     pool.join()

        _continue, missing = check_correlated_file_count(logger, output_dir, target_id, reference_ids, xcorr_dt, params, verbose, return_missing=True)

    if 'plot_beam' in steps:
        procs._plot_beam(output_dir, params, verbose)

    if 'gencal' in steps:
        procs._gencal(output_dir, target_id, xcorr_dt, reference_ids, params, parallel, verbose)

    if 'applycal' in steps:
        procs._applycal(output_dir, target_id, xcorr_dt, params, reference_ids, parallel, verbose)

    if 'clip' in steps:
        procs._clip(output_dir, target_id, reference_ids, xcorr_dt, params, parallel)

    average_t_dt = params['average_t_dt']

    if 'average_t' in steps:
        procs._average_t(output_dir, target_id, average_t_dt, reference_ids, xcorr_dt, params, parallel)

    if 'to_uvhol' in steps:
        procs._to_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, reference_ids, params, parallel)

    if 'average_uvhol' in steps:
        procs._average_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, reference_ids)
        
    if 'solve_uvhol' in steps:
        procs._solve_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, logger)
        
    if 'order_sols' in steps:
        procs._order_sols(output_dir, target_id, xcorr_dt, average_t_dt, params)
        
    if 'plot_report' in steps:
        procs._plot_report(output_dir, target_id, xcorr_dt, average_t_dt, params)

    logger.info('Done with steps: {0}'.format(steps))
