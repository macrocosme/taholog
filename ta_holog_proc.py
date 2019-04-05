#!/usr/bin/env python

"""
This script runs all the steps required to process raw voltage data into complex valued beam maps.
It assumes that the data will be located in trunk_dir, and each observation (target and reference)
will be in a separate directory named as their sasid.
"""

import re
import os
import glob
import logging
import numpy as np
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')

from taholog import misc, to_freq, xcorr, gencal, applycal, \
                    clip, average, to_uvhol, solve, order, plot

def run_pipeline(params):
    """
    Runs the steps required to process holography data (as recorded by COBALT).
    """
    
    # Set up logger.
    mp.log_to_stderr(logging.DEBUG)
    
    steps = params['steps']
    logger.info('Will run the following steps: {0}'.format(steps))
    
    # Unpack some kwargs.
    target_id = params['target_id']
    reference_ids = params['reference_ids']

    
    # Setup polarization mapping
    num_pol = params['to_freq_num_pol']
    mfp = '0,1,2,3'
    mfpl = mfp.split(',')
    polmap = np.array(mfpl, dtype=int).reshape(num_pol, len(mfpl)//num_pol)

    cwd = os.getcwd()

    # Go to where the data is.
    os.chdir(trunk_dir)

    if 'to_freq' in steps:

        # Start with the reference stations.
        for i,ref in enumerate(reference_ids.split(',')):
            os.chdir(ref)
            input_file = '{0}_SAP000_B{1:03d}_S0_P000_bf.h5'.format(ref, 0)
            output_base = '{0}_SAP000_B{1:03d}_P000_bf'.format(ref, 0)
            to_freq.main(input_file, output_base, 
                         params['to_freq_num_chan'], 
                         num_pol, 
                         params['to_freq_num_files'], 
                         polmap)
            os.chdir(trunk_dir)

        # Now the target stations.
        os.chdir(target_id)

        if not debug:
            
            pool = mp.Pool(processes=params['to_freq_cpus'])
            
            for beam in params['to_freq_beams']:

                input_file = '{0}_SAP000_B{1:03d}_S0_P000_bf.h5'.format(target_id, beam)
                output_base = '{0}_SAP000_B{1:03d}_P000_bf'.format(target_id, beam)

                pool.apply_async(to_freq.main,
                                 args=(input_file, output_base, 
                                       params['to_freq_num_chan'], 
                                       num_pol, 
                                       params['to_freq_num_files'],
                                       polmap))

            pool.close()
            pool.join()

        else:

            for beam in params['to_freq_beams']:

                input_file = '{0}_SAP000_B{1:03d}_S0_P000_bf.h5'.format(target_id, beam)
                output_base = '{0}_SAP000_B{1:03d}_P000_bf'.format(target_id, beam)

                to_freq.main(input_file, output_base, params['to_freq_num_chan'], 
                             num_pol, params['to_freq_num_files'], polmap)
                
        logger.info('Finished with to_freq step.')

    os.chdir(trunk_dir)
    
    logger.info('Checking that there are enough output files.')
    expected_to_freq_output = params['to_freq_beams']*params['spws']
    

    xcorr_dt = params['xcorr_dt']
    

    if 'xcorr' in steps:

        # Make output directories if necessary.
        xcorr_output_dir = '{0}_xcorr'.format(target_id)
        if not os.path.isdir(xcorr_output_dir):
            os.makedirs(xcorr_output_dir, exist_ok=True)
            for ref in reference_ids.split(','):
                os.makedirs('{0}/{1}'.format(xcorr_output_dir, ref), exist_ok=True)
        
        target = lambda ibm, ifn: '{0}/{0}_SAP000_B{1:03d}_P000_bf_spw{2}.h5'.format(target_id, ibm, ifn)
        refers = lambda ref, ifn: '{0}/{0}_SAP000_B000_P000_bf_spw{1}.h5'.format(ref, ifn)
        output = lambda ref, ibm, ifn: '{0}/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}.h5'.format(xcorr_output_dir, ref, ibm, ifn, xcorr_dt)
        rfi_output = lambda ref, ibm, ifn: '{0}/{1}/SAP000_B{2:03d}_P000_spw{3}_rfiflags.h5'.format(xcorr_output_dir, ref, ibm, ifn)

        rfi_kwargs = {'flagging_threshold': params['xcorr_flagging_threshold'],
                      'threshold_shrink_power': params['xcorr_threshold_shrink_power'],
                      'ext_time_percent': 0.5,
                      'ext_freq_percent': 0.5,
                      'n_rfi_max': 1
                                    }

        kwargs = {'target_time_res': xcorr_dt,
                  'rfiflag': params['xcorr_rfiflag'],
                  'edges': params['xcorr_edges'],
                  'rfi_output': rfi,
                  'rfi_kwargs': rfi_kwargs}

        if not debug:

            pool = mp.Pool(processes=params['xcorr_cpus'])

            for refid in reference_ids.split(','):
                for spw in xcorr_spws:
                    for ibm in xcorr_beams:

                        tgt = target(ibm, spw)
                        ref = refers(refid, spw)
                        out = output(refid, ibm, spw)
                        rfi = rfi_output(refid, ibm, spw)

                        pool.apply_async(xcorr.main, args=(tgt, ref, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in xcorr_spws:
                    for ibm in xcorr_beams:

                        tgt = target(ibm, spw)
                        ref = refers(refid, spw)
                        out = output(refid, ibm, spw)
                        rfi = rfi_output(refid, ibm, spw)

                        xcorr.main(tgt, ref, out, **kwargs)

    if 'plot_beam' in steps:

        plot.plot_phase_beam(params['plot_beam_ffun'], 
                             params['plot_beam_outp'], 
                             params['plot_beam_spws'], 
                             params['plot_beam_refs'])

    if 'gencal' in steps:

        target = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}.h5'.format(target_id, ref, spw, xcorr_dt)
        output = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, ref, spw, xcorr_dt)
        
        kwargs = {'smooth': params['gencal_smooth']}
        
        if not debug:

            pool = mp.Pool(processes=params['gencal_cpus'])

            for refid in reference_ids.split(','):
                for spw in params['gencal_spws']:

                    tgt = target(refid, spw)
                    out = output(refid, spw)
                    
                    pool.apply_async(gencal.main, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in params['gencal_spws']:

                    tgt = target(refid, spw)
                    out = output(refid, spw)

                    gencal.main(tgt, out, **kwargs)

    if 'applycal' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal.h5'.format(target_id, ref, beam, spw, xcorr_dt)

        if not debug:

            pool = mp.Pool(processes=params['applycal_cpus'])

            for refid in reference_ids.split(','):
                for spw in params['applycal_spws']:
                
                    solutions_file = '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, refid, spw, xcorr_dt)

                    for beam in params['applycal_beams']:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        pool.apply_async(applycal.main,
                                         args=(solutions_file, tgt, out))

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in params['applycal_spws']:
 
                    solutions_file = '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, refid, spw, xcorr_dt)

                    for beam in applycal_beams:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        applycal.main(solutions_file, tgt, out)

    if 'clip' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip.h5'.format(target_id, ref, beam, spw, xcorr_dt)

        kwargs = {'threshold': params['clip_threshold']}

        if not debug:

            pool = mp.Pool(processes=params['clip_cpus'])

            for refid in reference_ids.split(','):
                for spw in params['clip_spws']:
                    for beam in params['clip_beams']:
                    
                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        pool.apply_async(clip.abs_clip_vis, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in params['clip_spws']:
                    for beam in params['clip_beams']:
 
                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        clip.abs_clip_vis(tgt, out, **kwargs)

    average_t_dt = params['average_t_dt']

    if 'average_t' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip_avg{5}.h5'.format(target_id, ref, beam, spw, xcorr_dt, average_t_dt)

        kwargs = {'target_time_res': average_t_dt,
                  'weighted': params['average_t_weighted']}

        if not debug:

            pool = mp.Pool(processes=params['average_t_cpus'])

            for refid in reference_ids.split(','):
                for spw in params['average_t_spws']:
                    for beam in params['average_t_beams']:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        pool.apply_async(average.time_average_vis, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in params['average_t_spws']:
                    for beam in params['average_t_beams']:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        average.time_average_vis(tgt, out, **kwargs)

    if 'to_uvhol' in steps:

        target = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B*_P000_spw{2}_avg{3}_cal_clip_avg{4}.h5'.format(target_id, ref, spw, xcorr_dt, average_t_dt)
        output = lambda ref, spw: '{0}_xcorr/{1}/spw{2}_avg{3}_cal_clip_avg{4}'.format(target_id, ref, spw, xcorr_dt, average_t_dt)

        if not debug:

            pool = mp.Pool(processes=params['to_uvhol_cpus'])

            for refid in reference_ids.split(','):
                for spw in params['to_uvhol_spws']:

                    tgt = target(refid, spw)
                    out = output(refid, spw)

                    tgt_list = glob.glob(tgt)
                    misc.natural_sort(tgt_list)

                    pool.apply_async(to_uvhol.main, args=(tgt_list, out))

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in params['to_uvhol_spws']:

                    tgt = target(refid, spw)
                    out = output(refid, spw)

                    tgt_list = glob.glob(tgt)
                    misc.natural_sort(tgt_list)
                    
                    to_uvhol.main(tgt_list, out)

    if 'average_uvhol' in steps:

        for pol in params['average_uvhol_pols']:
            for spw in params['average_uvhol_spws']:

                file_list = []

                # <- from Here
                for refid in reference_ids.split(','): 
            
                    file_list.append(glob.glob('{0}_xcorr/{1}/*spw{2}_avg{3}_cal_clip_avg{4}_{5}_t*.uvhol'.format(target_id, refid, spw, xcorr_dt, average_t_dt, pol)))

                file_list = [item for sublist in file_list for item in sublist]
                # <- to Here
                # The above can be replaced by this line:
                #file_list = glob.glob('{0}_xcorr/L*/*spw{1}_avg{2}_cal_clip_avg{3}_{4}_t*.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol))
                misc.natural_sort(file_list)

                output = lambda t: '{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t{5}.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol, t)

                average.average_uvhol(file_list, reference_ids.split(','), output, pol)

    if 'solve_uvhol' in steps:

        for pol in params['solve_uvhol_pols']:
            for spw in params['solve_uvhol_spws']:

                files = glob.glob('{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t*.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol))

                for f in files:

                    inp = f
                    t = re.findall('t\d+', f)[0][1:]
                    out = '{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t{5}'.format(target_id, spw, xcorr_dt, average_t_dt, pol, t)

                    logger.info('Solving: {0} to {1}'.format(inp, out))

                    try:
                        solve.uvhol(inp, out, weighted=params['solve_weighted'])
                    except (np.linalg.linalg.LinAlgError, ValueError):
                        logger.info("Could not solve: {0}".format(inp))


    if 'order_sols' in steps:

        for pol in params['solve_uvhol_pols']:
            
            sol_files = glob.glob('{0}_xcorr/spw*_avg{1}_cal_clip_avg{2}_{3}_t*.pickle'.format(target_id, xcorr_dt, average_t_dt, pol))
            output = '{0}_xcorr/avg{1}_cal_clip_avg{2}_{3}_sols.pickle'.format(target_id, xcorr_dt, average_t_dt, pol)
            
            order.main(sol_files, output, params['order_sols_phase_reference_station'], params['order_sols_degree'])
            

    if 'plot_report' in steps:
        
        for pol in params['solve_uvhol_pols']:

            solutions_file =  '{0}_xcorr/avg{1}_cal_clip_avg{2}_{3}_sols.pickle'.format(target_id, xcorr_dt, average_t_dt, pol)
            uvhol_files_func = lambda spw: '{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t0.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol)
            
            plot.plot_report(plot_report_output(pol), solutions_file, uvhol_files_func, params['order_sols_phase_reference_station'])

    logger.info('Done with steps: {0}'.format(steps))
