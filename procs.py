import multiprocessing as mp
import os
import glob
import re
import numpy as np
from datetime import datetime
import logging
from utils import check_folder_exists_or_create
from taholog.steps import to_freq, xcorr, plot, gencal, applycal, clip, average, to_uvhol, misc, solve, order

def _to_freq(trunk_dir, output_dir, cs_str, target_id, reference_ids, params, num_pol, polmap, logger, parallel, verbose=False):
    """ Convert stations' spectral windows (subbands) to frequency regime.

    For each reference and target stations, convert time series at each subband into spectra.

    Parameters
    ----------
    trunk_dir: str
        Trunk directory where original data is stored
    output_dir: str
        Output directory where processed data will be stored
    cs_str: str
        CS String
    target_id: str
        Target ID
    reference_ids: str
        Reference ID
    params: dict
        Parameters metadata
    num_pol: int
        Number of polarization vectors
    polmap: [[],[]]
        Polarization map
    logger: logging.logger
        Logger
    parallel: bool
        Whether or not to run in parallel
    verbose: bool
        Print extra information to stdout
    """

    # Reference stations.
    if verbose: 
        print ('to_freq.main for each reference station')

    current_directories = []
    input_files = []
    output_bases = []

    # Construct files list
    # First, reference stations
    for ref in reference_ids:
        current_dir = f"{trunk_dir}{ref}/{cs_str}/"
        _outdir = check_folder_exists_or_create(f"{output_dir}{ref}", return_folder=True)
        os.chdir(current_dir)

        for beam in params['ref_beams']:
            current_directories.append( current_dir )
            input_files.append( f'{current_dir}{ref}_SAP000_B{beam:03d}_S0_P000_bf.h5' )
            outdir = check_folder_exists_or_create(f"{_outdir}{beam}", return_folder=True)
            output_bases.append( f'{outdir}{ref}_SAP000_B{0:03d}_P000_bf' )

    # Second, target stations
    current_dir = f"{trunk_dir}{target_id}/{cs_str}/"
    outdir = check_folder_exists_or_create(f"{output_dir}{target_id}", return_folder=True)
    for beam in params['to_freq_beams']:
        current_directories.append( current_dir )
        input_files.append( f'{current_dir}{target_id}_SAP000_B{beam:03d}_S0_P000_bf.h5' )
        output_bases.append( f'{outdir}{target_id}_SAP000_B{beam:03d}_P000_bf' )

    # Run FFTs over all files
    start_time = datetime.now()
    if not parallel:
        for current_dir, input_file, output_base in zip(current_directories, input_files, output_bases):
            to_freq.main(input_file, 
                         output_base, 
                         current_dir,
                         params['to_freq_num_chan'], 
                         num_pol, 
                         params['to_freq_num_files'], 
                         polmap,
                         params['to_freq_cpus'],
                         params['to_disk'], 
                         params['use_pyfftw'], 
                         params['use_gpu'])
    else:
        logger.info(f"Multiprocessing: {params['to_freq_cpus']} processes.")
        pool = mp.Pool(processes=params['to_freq_cpus'])
        for current_dir, input_file, output_base in zip(current_directories, input_files, output_bases):
            pool.apply_async(to_freq.main,
                            args=(
                                input_file, 
                                output_base, 
                                current_dir,
                                params['to_freq_num_chan'], 
                                num_pol, 
                                params['to_freq_num_files'], 
                                polmap,
                                params['to_freq_cpus'],
                                params['to_disk'], 
                                params['use_pyfftw'], 
                                params['use_gpu']
                            ))
        pool.close()
        pool.join()
    logger.info(f'Processed {len(input_files)} files in {datetime.now() - start_time}')
            
    logger.info('Finished with to_freq step.')

def _xcorr(output_dir, cs_str, reference_ids, target_id, xcorr_dt, params, verbose=False):
    """ Correlate all spectral windows between all referece stations and the target station

    Parameters
    ----------
    output_dir: str
        Output directory
    cs_str: str
        Sub-folder string for original data
    reference_ids: list
        Reference stations ID(s)
    target_id: str
        Target ID
    xcorr_dt: float
        Averaging factor
    params: dict
        Parameters sent to the main function
    parallel: bool
        Process data on multiple CPUs using multiprocessing
    verbose: bool (default: False)
        Print extra information to stdout (other than log file)
    """

    if verbose: 
        print ('xcorr')
        print ('Make output directories if necessary. _xcorr')
    # Make output directories if necessary.
    os.chdir(output_dir)
    
    xcorr_output_dir = f'{output_dir}{target_id}_xcorr'
    print (f'check if {xcorr_output_dir} exists')
    if not os.path.isdir(xcorr_output_dir):
        check_folder_exists_or_create(xcorr_output_dir, return_folder=False)
        for ref in reference_ids:
            check_folder_exists_or_create(f'{xcorr_output_dir}/{ref}', return_folder=False)
            for ref_beam in params['ref_beams']:
                check_folder_exists_or_create(f'{xcorr_output_dir}/{ref}/{ref_beam}', return_folder=False)
                

    # target = lambda ibm, spw: f'{output_dir}{target_id}/{cs_str}/{target_id}_SAP000_B{ibm:03d}_P000_bf_spw{spw}.h5' 
    # refers = lambda ref, refbm, spw: f'{output_dir}{ref}/{cs_str}/{ref}_SAP000_B{refbm:03d}_P000_bf_spw{spw}.h5' 
    target = lambda ibm, spw: f'{output_dir}{target_id}/{target_id}_SAP000_B{ibm:03d}_P000_bf_spw{spw}.h5' 
    refers = lambda ref, refbm, spw: f'{output_dir}{ref}/{refbm}/{ref}_SAP000_B000_P000_bf_spw{spw}.h5' 
    output = lambda ref, refbm, ibm, spw: f'{xcorr_output_dir}/{ref}/{refbm}/SAP000_B{ibm:03d}_P000_spw{spw}_avg{xcorr_dt}.h5' 
    rfi_output = lambda ref, refbm, ibm, spw: f'{xcorr_output_dir}/{ref}/{refbm}/SAP000_B{ibm:03d}_P000_spw{spw}_rfiflags.h5'

    rfi_kwargs = {'flagging_threshold': params['xcorr_flagging_threshold'],
                  'threshold_shrink_power': params['xcorr_threshold_shrink_power'],
                  'ext_time_percent': 0.5,
                  'ext_freq_percent': 0.5,
                  'n_rfi_max': 1}

    kwargs = {'target_time_res': xcorr_dt,
              'rfiflag': params['xcorr_rfiflag'],
              'edges': params['xcorr_edges'],
              'rfi_kwargs': rfi_kwargs, 
              'parallel': params['parallel'],
              'use_numba': params['use_numba']}

    if not params['parallel']:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['xcorr_spws']:
                    for ibm in params['xcorr_beams']:

                        tgt = target(ibm, spw)
                        ref = refers(refid, ref_beam, spw)
                        out = output(refid, ref_beam, ibm, spw)
                        rfi = rfi_output(refid, ref_beam, ibm, spw)

                        kwargs['rfi_output'] = rfi

                        xcorr.main(tgt, ref, out, **kwargs)
    else:
        pool = mp.Pool(processes=params['xcorr_cpus'])
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['xcorr_spws']:
                    for ibm in params['xcorr_beams']:

                        tgt = target(ibm, spw)
                        ref = refers(refid, ref_beam, spw)
                        out = output(refid, ref_beam, ibm, spw)
                        rfi = rfi_output(refid, ref_beam, ibm, spw)

                        kwargs['rfi_output'] = rfi

                        pool.apply_async(xcorr.main, args=(tgt, ref, out), kwds=kwargs)
        pool.close()
        pool.join()

def _redo_missing_xcorr(output_dir, xcorr_output_dir, target_id, params, xcorr_dt, refid, ref_beam, spw, ibm):
    target = lambda _ibm, _spw: f'{output_dir}{target_id}/{target_id}_SAP000_B{_ibm:03d}_P000_bf_spw{_spw}.h5' 
    refers = lambda _ref, _refbm, _spw: f'{output_dir}{_ref}/{_refbm}/{_ref}_SAP000_B000_P000_bf_spw{_spw}.h5' 
    output = lambda _ref, _refbm, _ibm, _spw: f'{xcorr_output_dir}/{_ref}/{_refbm}/SAP000_B{_ibm:03d}_P000_spw{_spw}_avg{xcorr_dt}.h5' 
    rfi_output = lambda _ref, _refbm, _ibm, _spw: f'{xcorr_output_dir}/{_ref}/{_refbm}/SAP000_B{_ibm:03d}_P000_spw{_spw}_rfiflags.h5'

    tgt = target(int(ibm), spw)
    ref = refers(refid, ref_beam, spw)
    out = output(refid, ref_beam, int(ibm), spw)
    rfi = rfi_output(refid, ref_beam, int(ibm), spw)

    rfi_kwargs = {'flagging_threshold': params['xcorr_flagging_threshold'],
                  'threshold_shrink_power': params['xcorr_threshold_shrink_power'],
                  'ext_time_percent': 0.5,
                  'ext_freq_percent': 0.5,
                  'n_rfi_max': 1}

    kwargs = {'target_time_res': xcorr_dt,
              'rfiflag': params['xcorr_rfiflag'],
              'edges': params['xcorr_edges'],
              'rfi_kwargs': rfi_kwargs, 
              'parallel': params['parallel'],
              'use_numba': params['use_numba']}

    kwargs['rfi_output'] = rfi

    xcorr.main(tgt, ref, out, **kwargs)

def _plot_beam(output_dir, params, verbose=False):
    if verbose: 
        print ('plot_beam')

    os.chdir(output_dir)

    print (params['plot_beam_ffun'], 
           params['plot_beam_outp'], 
           params['plot_beam_spws'], 
           params['plot_beam_refs'],
           params['ref_beams'])

    plot.plot_phase_beam(params['plot_beam_ffun'], 
                         params['plot_beam_outp'], 
                         params['plot_beam_spws'], 
                         params['plot_beam_refs'],
                         params['ref_beams'])

def _gencal(output_dir, target_id, xcorr_dt, reference_ids, params, parallel, verbose=False):
    if verbose: 
        print ('gencal')
    target = lambda ref, refbm, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B000_P000_spw{spw}_avg{xcorr_dt}.h5'
    output = lambda ref, refbm, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B000_P000_spw{spw}_avg{xcorr_dt}_sol.h5'
    
    kwargs = {'smooth': params['gencal_smooth']}
    
    if not parallel:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['gencal_spws']:

                    tgt = target(refid, ref_beam, spw)
                    out = output(refid, ref_beam, spw)

                    gencal.main(tgt, out, **kwargs)
    else:
        pool = mp.Pool(processes=params['gencal_cpus'])
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['gencal_spws']:

                    tgt = target(refid, ref_beam, spw)
                    out = output(refid, ref_beam, spw)
                    
                    pool.apply_async(gencal.main, args=(tgt, out), kwds=kwargs)
        pool.close()
        pool.join()

def _applycal(output_dir, target_id, xcorr_dt, params, reference_ids, parallel, verbose=False):
    if verbose: 
        print ('applycal: _spw_avg')

    target = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}.h5'
    output = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}_cal.h5'

    if not parallel:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['applycal_spws']:

                    solutions_file = f'{output_dir}{target_id}_xcorr/{refid}/{ref_beam}/SAP000_B000_P000_spw{spw}_avg{xcorr_dt}_sol.h5'

                    for beam in params['applycal_beams']:

                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        applycal.main(solutions_file, tgt, out)
    else:
        pool = mp.Pool(processes=params['applycal_cpus'])
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['applycal_spws']:
                
                    solutions_file = f'{output_dir}{target_id}_xcorr/{refid}/{ref_beam}/SAP000_B000_P000_spw{spw}_avg{xcorr_dt}_sol.h5'

                    for beam in params['applycal_beams']:

                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        pool.apply_async(applycal.main,
                                            args=(solutions_file, tgt, out))
        pool.close()
        pool.join()

def _clip(output_dir, target_id, reference_ids, xcorr_dt, params, parallel):

    target = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}_cal.h5'
    output = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}_cal_clip.h5'

    kwargs = {'threshold': params['clip_threshold']}

    if not parallel:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['clip_spws']:
                    for beam in params['clip_beams']:

                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        clip.abs_clip_vis(tgt, out, **kwargs)
    else:
        pool = mp.Pool(processes=params['clip_cpus'])
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['clip_spws']:
                    for beam in params['clip_beams']:
                    
                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        pool.apply_async(clip.abs_clip_vis, args=(tgt, out), kwds=kwargs)
        pool.close()
        pool.join()

def _average_t(output_dir, target_id, average_t_dt, reference_ids, xcorr_dt, params, parallel):
    target = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}_cal_clip.h5'
    output = lambda ref, refbm, beam, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B{beam:03d}_P000_spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}.h5'

    kwargs = {'target_time_res': average_t_dt,
                'weighted': params['average_t_weighted']}

    if not parallel:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['average_t_spws']:
                    for beam in params['average_t_beams']:

                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        average.time_average_vis(tgt, out, **kwargs)
    else:
        pool = mp.Pool(processes=params['average_t_cpus'])
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['average_t_spws']:
                    for beam in params['average_t_beams']:

                        tgt = target(refid, ref_beam, beam, spw)
                        out = output(refid, ref_beam, beam, spw)

                        pool.apply_async(average.time_average_vis, args=(tgt, out), kwds=kwargs)
        pool.close()
        pool.join()

def _to_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, reference_ids, params, parallel):
    target = lambda ref, refbm, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/SAP000_B*_P000_spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}.h5'
    output = lambda ref, refbm, spw: f'{output_dir}{target_id}_xcorr/{ref}/{refbm}/spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}'

    logger = logging.getLogger(__name__)

    if not parallel:
        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['to_uvhol_spws']:

                    tgt = target(refid, ref_beam, spw)
                    out = output(refid, ref_beam, spw)

                    tgt_list = glob.glob(tgt)
                    misc.natural_sort(tgt_list)
                    
                    to_uvhol.main(tgt_list, out)
    else:
        pool = mp.Pool(processes=params['to_uvhol_cpus'])

        for refid in reference_ids:
            for ref_beam in params['ref_beams']:
                for spw in params['to_uvhol_spws']:

                    tgt = target(refid, ref_beam, spw)
                    out = output(refid, ref_beam, spw)

                    tgt_list = glob.glob(tgt)

                    misc.natural_sort(tgt_list)

                    pool.apply_async(to_uvhol.main, args=(tgt_list, out))

        pool.close()
        pool.join()

def _average_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, reference_ids):
    for pol in params['average_uvhol_pols']:
        for spw in params['average_uvhol_spws']:

            file_list = []

            # <- from Here
            for refid in reference_ids: 
                for ref_beam in params['ref_beams']:
    
                    file_list.append(glob.glob(f'{output_dir}{target_id}_xcorr/{refid}/{ref_beam}/*spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t*.uvhol'))

            file_list = [item for sublist in file_list for item in sublist]

            print (file_list)

            # <- to Here
            # The above can be replaced by this line:
            #file_list = glob.glob('{0}_xcorr/L*/*spw{1}_avg{2}_cal_clip_avg{3}_{4}_t*.uvhol'.format(target_id, spw, xcorr_dt, average_t_dt, pol))
            misc.natural_sort(file_list)

            output = lambda t: f'{output_dir}{target_id}_xcorr/spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t{t}.uvhol'

            average.average_uvhol(file_list, reference_ids, params, output, pol)

def _solve_uvhol(output_dir, target_id, xcorr_dt, average_t_dt, params, logger):
    for pol in params['solve_uvhol_pols']:
        for spw in params['solve_uvhol_spws']:

            files = glob.glob(f'{output_dir}{target_id}_xcorr/spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t*.uvhol')

            for f in files:

                inp = f
                t = re.findall('t\d+', f)[0][1:]
                out = f'{output_dir}{target_id}_xcorr/spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t{t}'

                logger.info('Solving: {0} to {1}'.format(inp, out))

                try:
                    solve.uvhol(inp, out, weighted=params['solve_weighted'])
                except (np.linalg.linalg.LinAlgError, ValueError):
                    logger.info("Could not solve: {0}".format(inp))

def _order_sols(output_dir, target_id, xcorr_dt, average_t_dt, params):
    for pol in params['solve_uvhol_pols']:
        sol_files = glob.glob(f'{output_dir}{target_id}_xcorr/spw*_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t*.pickle')
        output = f'{output_dir}{target_id}_xcorr/avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_sols.pickle'
        order.main(sol_files, output, params['order_sols_phase_reference_station'], params['order_sols_degree'])

def _plot_report(output_dir, target_id, xcorr_dt, average_t_dt, params):
    for pol in params['solve_uvhol_pols']:
        solutions_file =  f'{output_dir}{target_id}_xcorr/avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_sols.pickle'
        uvhol_files_func = lambda spw: f'{output_dir}{target_id}_xcorr/spw{spw}_avg{xcorr_dt}_cal_clip_avg{average_t_dt}_{pol}_t0.uvhol'
        plot.plot_report(params['plot_report_output'](pol), solutions_file, uvhol_files_func, params['order_sols_phase_reference_station'])
