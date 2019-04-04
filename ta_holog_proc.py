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

import sys
sys.path.insert(0, '/home/salasp/python')
sys.path.insert(0, '/home/salasp/python/antennafield-0.1')
sys.path.insert(0, '/home/salasp/python/aoflagger')
sys.path.insert(0, '/home/salasp/python/holog-0.2')
from taholog import misc, to_freq, xcorr, gencal, applycal, \
                    clip, average, to_uvhol, solve, order, plot

if __name__ == '__main__':

    # User input.
    target_id = 'L697741' # SAS id of the observation with the map (many rings)
    reference_ids = 'L697733,L697735,L697737,L697739'
    trunk_dir = '/data/scratch/holography/tutorial/data/new_holography_obs/hba_cs002/'
    target_beams = 169
    spws = 10
    logfile = 'taholog_proc_vlad_debug.log'
    debug = False # Run steps without multiprocessing to get back all the error messages?
    """
    When using the multiprocessing module, I cannot log the error messages that take place in a child process.
    So, e.g., if a file is missing during the xcorr step, the script will not stop and it will continue without
    a beam. Less beams reduces the signal-to-noise ratio of the time delays.
    """

    # to_freq step options.
    # Converts the volatges from time to frequency-time domain.
    num_chan = 64  # Number of channels per spectra.
    num_pol = 2    # Number of polarizations recorded by the backend.
    num_files = 4  # Number of files across which the complex valued polarizations are spread, two for X two for Y.
    to_freq_cpus = 12
    to_freq_beams = range(0,target_beams)

    # xcorr step options.
    # Cross-correlates the data from the reference_ids with that of the target_id.
    xcorr_spws = range(0,spws)
    xcorr_beams = range(0,target_beams)
    edges = 0.125 # Flag this percentage of edge channels.
    xcorr_dt = 0.4 # Time resolution in seconds after correlating.
    rfiflag = True
    flagging_threshold = 2.0      # Data is flagged if above this time the local rms.
    threshold_shrink_power = 0.05 # How big are the regions where the local rms is computed.
    xcorr_cpus = 12

    # plot_beam options.
    # Will produce a plot of the phase of a beam for every refernce station and spectral window.
    plot_beam_spws = range(0,spws)
    plot_beam_refs = reference_ids
    plot_beam_outp = '{0}/beam_plots.pdf'.format(trunk_dir)
    plot_beam_beam = 0
    plot_beam_ffun = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}.h5'.format(target_id, ref, plot_beam_beam, spw, xcorr_dt)

    # gencal step options.
    # Finds the inverse of the Jones matrix of the central beam.
    gencal_spws = range(0,spws)
    gencal_smooth = False
    gencal_cpus = 4   # Set this value to the number of reference observations.

    # applycal step options.
    # Multiplies all the visibilities with the inverse of the Jones matrix of the central beam.
    applycal_spws = range(0,spws)
    applycal_beams = range(0,target_beams)
    applycal_cpus = 20

    # clip step options.
    # Flahs beams with an amplitude larger than clip_threshold.
    clip_spws = range(0,spws)
    clip_beams = range(0,target_beams)
    clip_threshold = 1.6      # Flags beams with amplitudes larger than this value. Absolute amplitude (after the calibration the central beam will an amplitude of one).
    clip_cpus = 20

    # average_t step options.
    # Averages the visibilities in time.
    average_t_spws = range(0,spws)
    average_t_beams = range(0,target_beams)
    average_t_dt = 6000  # seconds. Set it to the lenght of the observation unless you want to make beam movies.
    average_t_weighted = True # Use the weights when averaging the data in time. Set to True.
    average_t_cpus = 20

    # to_uvhol step options.
    # Writes the complex beam map to a .uvhol file (AIPS friendly).
    to_uvhol_spws = range(0,spws)
    to_uvhol_cpus = 20
  
    # average_uvhol step options.
    # Averages the complex beam maps observed with different reference stations.
    average_uvhol_pols = ['XX', 'YY', 'XY', 'YX'] # Polarizations to average in time.
    average_uvhol_spws = range(0,spws)

    # solve_uvhol step options.
    # Solves for the phases and amplitudes of the stations in the tied-array given a .uvhol file.
    solve_uvhol_pols = ['XX', 'YY'] # Polarizations for which we want phase and amplitude values. Do 'XX' and 'YY' only, as the cross terms should contain noise.
    solve_uvhol_spws = range(0,spws)
    solve_weighted = True  # Use wights when solving for the amplitudes and phases of the stations?

    # Order_sols step options.
    # Puts all the solutions for one polarization in a single file.
    order_sols_phase_reference_station = 'CS026HBA1' # Station used to reference the phases. 
    # Stations away from the superterp give better results. 
    # Choose an antenna with an almost flat phase as a function of frequency.
    order_sols_degree = 1 # Polynomial order used when fitting the phase as a unction of frequency. Use 1.

    # plot_report step options.
    plot_report_output = lambda pol: 'taholog_hba_report_{0}.pdf'.format(pol)

    # After each step a new file will be created.
    # For to_freq the new files will be in trunkdir/<SAS id>.
    # For the remaining steps the files will be in trunkdir/<target SAS id>_xcorr/<reference SAS id>
    # xcorr will add a _avg<xcorr_dt> to the file name.
    # gencal will create new files for the central beam B000 with an _sol to the filename.
    # applycal adds a _cal to the filename.
    # clip will add a _clip to the filenames.
    # average adds a _avg<average_dt> to the filenames.
    # to_uvhol creates .uvhol files with the same filename as the output from average.
 
    # All these steps are necessary.
    #steps = [#'to_freq','xcorr', 
    #         'gencal', 'applycal', 
    #         'clip', 'average_t', 
             #'to_uvhol', 
		#'average_uvhol', 'solve_uvhol', 
             #'order_sols', 'plot_report'] 
    steps = ['plot_beam']
    #steps = [ 'average_t', 
    #         'to_uvhol', 'average_uvhol', 'solve_uvhol', 
    #         'order_sols', 'plot_report'] 
    #steps = ['plot_report']
    # End of user input.

    # Set up logger.
    mp.log_to_stderr(logging.DEBUG)

    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format=formatter)
    logger = logging.getLogger(__name__)

    # Setup polarization mapping
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
            to_freq.main(input_file, output_base, num_chan, num_pol, num_files, polmap)
            os.chdir(trunk_dir)

        # Now the target stations.
        os.chdir(target_id)

        if not debug:
            
            pool = mp.Pool(processes=to_freq_cpus)
            
            for beam in to_freq_beams:

                input_file = '{0}_SAP000_B{1:03d}_S0_P000_bf.h5'.format(target_id, beam)
                output_base = '{0}_SAP000_B{1:03d}_P000_bf'.format(target_id, beam)

                pool.apply_async(to_freq.main,
                                 args=(input_file, output_base, num_chan, num_pol, num_files, polmap))

            pool.close()
            pool.join()

        else:

            for beam in to_freq_beams:

                input_file = '{0}_SAP000_B{1:03d}_S0_P000_bf.h5'.format(target_id, beam)
                output_base = '{0}_SAP000_B{1:03d}_P000_bf'.format(target_id, beam)

                to_freq.main(input_file, output_base, num_chan, num_pol, num_files, polmap)

    os.chdir(trunk_dir)

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

        if not debug:

            pool = mp.Pool(processes=xcorr_cpus)

            for refid in reference_ids.split(','):
                for spw in xcorr_spws:
                    for ibm in xcorr_beams:

                        tgt = target(ibm, spw)
                        ref = refers(refid, spw)
                        out = output(refid, ibm, spw)
                        rfi = rfi_output(refid, ibm, spw)

                        rfi_kwargs = {'flagging_threshold': flagging_threshold,
                                      'threshold_shrink_power': threshold_shrink_power,
                                      'ext_time_percent': 0.5,
                                      'ext_freq_percent': 0.5,
                                      'n_rfi_max': 1
                                     }

                        kwargs = {'target_time_res': xcorr_dt,
                                  'rfiflag': rfiflag,
                                  'edges': edges,
                                  'rfi_output': rfi,
                                  'rfi_kwargs': rfi_kwargs}

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

                        rfi_kwargs = {'flagging_threshold': flagging_threshold,
                                      'threshold_shrink_power': threshold_shrink_power,
                                      'ext_time_percent': 0.5,
                                      'ext_freq_percent': 0.5,
                                      'n_rfi_max': 1
                                     }

                        kwargs = {'target_time_res': xcorr_dt,
                                  'rfiflag': rfiflag,
                                  'edges': edges,
                                  'rfi_output': rfi,
                                  'rfi_kwargs': rfi_kwargs}

                        xcorr.main(tgt, ref, out, **kwargs)

    if 'plot_beam' in steps:

        plot.plot_phase_beam(plot_beam_ffun, plot_beam_outp, plot_beam_spws, plot_beam_refs)

    if 'gencal' in steps:

        target = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}.h5'.format(target_id, ref, spw, xcorr_dt)
        output = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, ref, spw, xcorr_dt)
        
        if not debug:

            pool = mp.Pool(processes=gencal_cpus)

            for refid in reference_ids.split(','):
                for spw in gencal_spws:

                    tgt = target(refid, spw)
                    out = output(refid, spw)
                    kwargs = {'smooth': gencal_smooth}

                    pool.apply_async(gencal.main, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in gencal_spws:

                    tgt = target(refid, spw)
                    out = output(refid, spw)
                    kwargs = {'smooth': gencal_smooth}

                    gencal.main(tgt, out, **kwargs)

    if 'applycal' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal.h5'.format(target_id, ref, beam, spw, xcorr_dt)

        if not debug:

            pool = mp.Pool(processes=applycal_cpus)

            for refid in reference_ids.split(','):
                for spw in applycal_spws:
                
                    solutions_file = '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, refid, spw, xcorr_dt)

                    for beam in applycal_beams:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        pool.apply_async(applycal.main,
                                         args=(solutions_file, tgt, out))

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in applycal_spws:
 
                    solutions_file = '{0}_xcorr/{1}/SAP000_B000_P000_spw{2}_avg{3}_sol.h5'.format(target_id, refid, spw, xcorr_dt)

                    for beam in applycal_beams:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        applycal.main(solutions_file, tgt, out)

    if 'clip' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip.h5'.format(target_id, ref, beam, spw, xcorr_dt)

        if not debug:

            pool = mp.Pool(processes=clip_cpus)

            for refid in reference_ids.split(','):
                for spw in clip_spws:
                    for beam in clip_beams:
                    
                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        kwargs = {'threshold': clip_threshold}

                        pool.apply_async(clip.abs_clip_vis, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in clip_spws:
                    for beam in clip_beams:
 
                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        kwargs = {'threshold': clip_threshold}

                        clip.abs_clip_vis(tgt, out, **kwargs)

    if 'average_t' in steps:

        target = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip.h5'.format(target_id, ref, beam, spw, xcorr_dt)
        output = lambda ref, beam, spw: '{0}_xcorr/{1}/SAP000_B{2:03d}_P000_spw{3}_avg{4}_cal_clip_avg{5}.h5'.format(target_id, ref, beam, spw, xcorr_dt, average_t_dt)

        if not debug:

            pool = mp.Pool(processes=average_t_cpus)

            for refid in reference_ids.split(','):
                for spw in average_t_spws:
                    for beam in average_t_beams:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        kwargs = {'target_time_res': average_t_dt,
                                  'weighted': average_t_weighted}

                        pool.apply_async(average.time_average_vis, args=(tgt, out), kwds=kwargs)

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in average_t_spws:
                    for beam in average_t_beams:

                        tgt = target(refid, beam, spw)
                        out = output(refid, beam, spw)

                        kwargs = {'target_time_res': average_t_dt,
                                  'weighted': average_t_weighted}

                        average.time_average_vis(tgt, out, **kwargs)

    if 'to_uvhol' in steps:

        target = lambda ref, spw: '{0}_xcorr/{1}/SAP000_B*_P000_spw{2}_avg{3}_cal_clip_avg{4}.h5'.format(target_id, ref, spw, xcorr_dt, average_t_dt)
        output = lambda ref, spw: '{0}_xcorr/{1}/spw{2}_avg{3}_cal_clip_avg{4}'.format(target_id, ref, spw, xcorr_dt, average_t_dt)

        if not debug:

            pool = mp.Pool(processes=to_uvhol_cpus)

            for refid in reference_ids.split(','):
                for spw in to_uvhol_spws:

                    tgt = target(refid, spw)
                    out = output(refid, spw)

                    tgt_list = glob.glob(tgt)
                    misc.natural_sort(tgt_list)

                    pool.apply_async(to_uvhol.main, args=(tgt_list, out))

            pool.close()
            pool.join()

        else:

            for refid in reference_ids.split(','):
                for spw in to_uvhol_spws:

                    tgt = target(refid, spw)
                    out = output(refid, spw)

                    tgt_list = glob.glob(tgt)
                    misc.natural_sort(tgt_list)
                    
                    to_uvhol.main(tgt_list, out)

    if 'average_uvhol' in steps:

        for pol in average_uvhol_pols:
            for spw in average_uvhol_spws:

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

        for pol in solve_uvhol_pols:
            for spw in solve_uvhol_spws:

                files = glob.glob('{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t*.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol))

                for f in files:

                    inp = f
                    t = re.findall('t\d+', f)[0][1:]
                    out = '{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t{5}'.format(target_id, spw, xcorr_dt, average_t_dt, pol, t)

                    logger.info('Solving: {0} to {1}'.format(inp, out))

                    try:
                        solve.uvhol(inp, out, weighted=solve_weighted)
                    except (np.linalg.linalg.LinAlgError, ValueError):
                        logger.info("Could not solve: {0}".format(inp))


    if 'order_sols' in steps:

        for pol in solve_uvhol_pols:
            
            sol_files = glob.glob('{0}_xcorr/spw*_avg{1}_cal_clip_avg{2}_{3}_t*.pickle'.format(target_id, xcorr_dt, average_t_dt, pol))
            output = '{0}_xcorr/avg{1}_cal_clip_avg{2}_{3}_sols.pickle'.format(target_id, xcorr_dt, average_t_dt, pol)
            
            order.main(sol_files, output, order_sols_phase_reference_station, order_sols_degree)
            

    if 'plot_report' in steps:
        
        for pol in solve_uvhol_pols:

            solutions_file =  '{0}_xcorr/avg{1}_cal_clip_avg{2}_{3}_sols.pickle'.format(target_id, xcorr_dt, average_t_dt, pol)
            uvhol_files_func = lambda spw: '{0}_xcorr/spw{1}_avg{2}_cal_clip_avg{3}_{4}_t0.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol)
            
            plot.plot_report(plot_report_output(pol), solutions_file, uvhol_files_func, order_sols_phase_reference_station)

    logger.info('Done with steps: {0}'.format(steps))
