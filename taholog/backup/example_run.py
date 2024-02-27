#!/usr/bin/env python

import logging

import sys

from taholog import ta_holog_proc as thp

if __name__ == '__main__':
    
    # User input.
    target_id = 'L697741' # SAS id of the observation with the map (many rings)
    reference_ids = 'L697733,L697735,L697737,L697739'
    trunk_dir = '/data/scratch/holography/tutorial/data/new_holography_obs/hba_cs002/'
    target_beams = 169
    spws = 10
    logfile = 'taholog_{0}.log'.format(target_id)
    debug = False # Run steps without multiprocessing to get back all the error messages?
    cs_str = 'cs' # In CEP4 the files are stored in a CS directory.
    """
    When using the multiprocessing module, I cannot log the error messages that take place in a child process.
    So, e.g., if a file is missing during the xcorr step, the script will not stop and it will continue without
    a beam. Less beams reduces the signal-to-noise ratio of the time delays.
    """

    # to_freq step options.
    # Converts the volatges from time to frequency-time domain.
    to_freq_num_chan = 64  # Number of channels per spectra.
    to_freq_num_pol = 2    # Number of polarizations recorded by the backend.
    to_freq_num_files = 4  # Number of files across which the complex valued polarizations are spread, two for X two for Y.
    to_freq_cpus = 12
    to_freq_beams = range(0,target_beams)

    # xcorr step options.
    # Cross-correlates the data from the reference_ids with that of the target_id.
    xcorr_spws = range(0,spws)
    xcorr_beams = range(0,target_beams)
    xcorr_edges = 0.125 # Flag this percentage of edge channels.
    xcorr_dt = 0.4 # Time resolution in seconds after correlating.
    xcorr_rfiflag = True
    xcorr_flagging_threshold = 2.0      # Data is flagged if above this time the local rms.
    xcorr_threshold_shrink_power = 0.05 # How big are the regions where the local rms is computed.
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
    plot_report_output = lambda pol: 'taholog_report_{0}_{1}.pdf'.format(target_id, pol)

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
    steps = ['to_freq','xcorr', 'plot_beam',
             'gencal', 'applycal', 
             'clip', 'average_t', 
             'to_uvhol', 
             'average_uvhol', 'solve_uvhol', 
             'order_sols', 'plot_report'] 
    # End of user input.
    
    kwargs = {'target_id': target_id,
              'reference_ids': reference_ids,
              'trunk_dir': trunk_dir,
              'target_beams': target_beams,
              'spws': spws,
              'logfile': logfile,
              'debug': debug,
              'cs_str': cs_str,
              'steps': steps,
              'to_freq_num_chan': to_freq_num_chan,
              'to_freq_num_pol': to_freq_num_pol,
              'to_freq_num_files': to_freq_num_files,
              'to_freq_cpus': to_freq_cpus,
              'to_freq_beams': to_freq_beams,
              'xcorr_spws': xcorr_spws,
              'xcorr_beams': xcorr_beams,
              'xcorr_edges': xcorr_edges,
              'xcorr_dt': xcorr_dt,
              'xcorr_rfiflag': xcorr_rfiflag,
              'xcorr_flagging_threshold': xcorr_flagging_threshold,
              'xcorr_threshold_shrink_power': xcorr_threshold_shrink_power,
              'xcorr_cpus': xcorr_cpus,
              'plot_beam_spws': plot_beam_spws,
              'plot_beam_refs': plot_beam_refs,
              'plot_beam_outp': plot_beam_outp,
              'plot_beam_beam': plot_beam_beam,
              'plot_beam_ffun': plot_beam_ffun,
              'gencal_spws': gencal_spws,
              'gencal_smooth': gencal_smooth,
              'gencal_cpus': gencal_cpus,
              'applycal_spws': applycal_spws,
              'applycal_beams': applycal_beams,
              'applycal_cpus': applycal_cpus, 
              'clip_spws': clip_spws,
              'clip_beams': clip_beams,
              'clip_threshold': clip_threshold,
              'clip_cpus': clip_cpus,
              'average_t_spws': average_t_spws,
              'average_t_beams': average_t_beams,
              'average_t_dt': average_t_dt,
              'average_t_weighted': average_t_weighted,
              'average_t_cpus': average_t_cpus,
              'to_uvhol_spws': to_uvhol_spws,
              'to_uvhol_cpus': to_uvhol_cpus,
              'average_uvhol_pols': average_uvhol_pols,
              'average_uvhol_spws': average_uvhol_spws,
              'solve_uvhol_pols': solve_uvhol_pols,
              'solve_uvhol_spws': solve_uvhol_spws,
              'solve_weighted': solve_weighted,
              'order_sols_phase_reference_station': order_sols_phase_reference_station,
              'order_sols_degree': order_sols_degree,
              'plot_report_output': plot_report_output}

    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=logfile, filemode='a', level=logging.DEBUG, format=formatter)
    logger = logging.getLogger(__name__)
    
    
    thp.run_pipeline(kwargs)
