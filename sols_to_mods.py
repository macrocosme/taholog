
import numpy as np
import pylab as plt
import scipy.constants as const

from scipy.interpolate import griddata

import holog
from taholog import beam, stations

def make_grid(xaxis, yaxis, symmetric=True):
    """
    """

    dx = min(abs(xaxis[:-1:2] - xaxis[1::2]))
    dy = min(abs(yaxis[:-1:2] - yaxis[1::2]))
    
    if symmetric:
        dd = min((dx, dy))
        dx = dd
        dy = dd
        
    grid = np.mgrid[xaxis.min():xaxis.max():dx, 
                    yaxis.min():yaxis.max():dy]
    
    return grid

def make_model(spws, sols, uvhol_files_func, phase_ref_station=''):
    """
    """

    # Load the UVHOL files.
    hd = np.empty(len(spws), dtype=object)

    for i,spw in enumerate(spws):
        
        print('Loading UVHOL file: {0}'.format(uvhol_files_func(spw)))

        hd[i] = holog.uvhol.read_uvhol_file(uvhol_files_func(spw))[0]
        
    # The (l,m) coordinates are always the same.
    laxis = hd[0].l_rad
    maxis = hd[0].m_rad

    lm_grid = make_grid(laxis, maxis, symmetric=True)
    
    obs_beam = np.empty((len(spws),)+lm_grid[0].shape, dtype=np.complex64)

    for i,spw in enumerate(spws):
                
        obs_beam[i] = griddata(np.array([hd[i].l_rad, hd[i].m_rad]).T, hd[i].vis, (lm_grid[0], lm_grid[1]), 
                               method='linear', fill_value=np.nan, rescale=False)
        
    obs_beam = np.ma.masked_invalid(obs_beam)
    
    # Get antenna locations.
    stats = sorted(sols.keys())
    if isinstance(stats[0], np.bytes_):
        stats = np.array([stat.decode() for stat in stats])
    station_list = ','.join(stats)
    station_pqr = stations.station_offsets_pqr(station_list)
    
    # Order the complex gain solutions.
    amp_sol = np.empty((len(spws), len(stats)))
    phs_sol = np.empty((len(spws), len(stats)))
    phs_sol_fit = np.empty((len(spws), len(stats)))

    for i,k in enumerate(stats):
        amp_sol[:,i] = sols[k]['amp'][0,0]
        phs_sol[:,i] = sols[k]['ph'][0,0]
        phs_sol_fit[:,i] = 2.*np.pi*sols[k]['freq']*sols[k]['tau'] + sols[k]['tau0'] + np.deg2rad(sols[phase_ref_station]['ph'][0,0,:].mean())
        
    # Make model beams using the complex gain solutions.
    mod_hd = np.empty((len(spws)), dtype=object)
    mod_beam_vis = np.empty((len(spws),)+laxis.shape, dtype=np.complex64)
    mod_beam_vis_fit = np.empty((len(spws),)+laxis.shape, dtype=np.complex64)

    mod_beam = np.empty((len(spws),)+lm_grid[0].shape, dtype=np.complex64)
    mod_beam_fit = np.empty((len(spws),)+lm_grid[0].shape, dtype=np.complex64)

    for i,spw in enumerate(spws):
        
        mod_beam_vis[i] = beam.simple_beam_model(np.array([hd[i].l_rad, hd[i].m_rad]).T, 
                                                station_pqr[:,:-1]*sols[stats[0]]['freq'][spw]/const.c, 
                                                amp_sol[i],
                                                np.deg2rad(phs_sol[i]))
        mod_beam_vis[i] = mod_beam_vis[i]/np.max(abs(mod_beam_vis[i]))*np.max(abs(obs_beam[i]))
        mod_hd[i] = holog.HologData(hd[i].l_rad, hd[i].m_rad, mod_beam_vis[i], 
                                    np.ones(len(mod_beam_vis[i]), dtype=np.complex64)*0.001, 
                                    'Beam Model', sols[stats[0]]['freq'][spw], 'XX', ['RS508HBA0'])
        
        mod_beam[i] = griddata(np.array([hd[i].l_rad, hd[i].m_rad]).T, mod_beam_vis[i], 
                               (lm_grid[0],lm_grid[1]), method='linear', 
                               fill_value=np.nan, rescale=True)
        
        mod_beam_vis_fit[i] = beam.simple_beam_model(np.array([hd[i].l_rad, hd[i].m_rad]).T, 
                                                     station_pqr[:,:-1]*sols[stats[0]]['freq'][spw]/const.c, 
                                                     amp_sol[i],
                                                     #np.ones(len(amp_sol[i])),
                                                     phs_sol_fit[i])
        mod_beam_vis_fit[i] = mod_beam_vis_fit[i]/np.max(abs(mod_beam_vis_fit[i]))*np.max(abs(obs_beam[i]))
        
        mod_beam_fit[i] = griddata(np.array([hd[i].l_rad, hd[i].m_rad]).T, mod_beam_vis_fit[i], 
                                   (lm_grid[0],lm_grid[1]), method='linear', 
                                   fill_value=np.nan, rescale=True)
        
    mod_beam = np.ma.masked_invalid(mod_beam)
    mod_beam_fit = np.ma.masked_invalid(mod_beam_fit)
    
    residuals = obs_beam - mod_beam
    residuals_fit = obs_beam - mod_beam_fit

    res_vis = np.empty((len(spws), len(laxis)), dtype=np.complex64)
    res_fit_vis = np.empty((len(spws), len(laxis)), dtype=np.complex64)
    res_hd = np.empty((len(spws)), dtype=object)
    res_fit_hd = np.empty((len(spws)), dtype=object)
    
    for i, spw in enumerate(spws):
        res_vis[i] = hd[i].vis - mod_beam_vis[i]
        res_fit_vis[i] = hd[i].vis - mod_beam_vis_fit[i]
        
        res_hd[i] = holog.HologData(laxis, maxis, res_vis[i], 
                                    np.ones(len(res_vis[i]), dtype=np.complex64)*0.001, 
                                    'Beam Model', hd[i].freq_hz, 'XX', ['RS508HBA0'])
        res_fit_hd[i] = holog.HologData(laxis, maxis, res_fit_vis[i], 
                                    np.ones(len(res_fit_vis[i]), dtype=np.complex64)*0.001, 
                                    'Beam Model', hd[i].freq_hz, 'XX', ['RS508HBA0'])
    
    results = {'station_pqr': station_pqr,
               'laxis': laxis, 'maxis': maxis,
               'lm_grid': lm_grid, 'holog_data': hd,
               'obs_beam': obs_beam, 'mod_beam': mod_beam,
               'mod_beam_fit': mod_beam_fit, 'residuals': residuals,
               'residuals_fit': residuals_fit, 'mod_beam_vis': mod_beam_vis,
               'mod_beam_vis_fit': mod_beam_vis_fit,
               'residual_visibilities': res_vis,
               'residual_visibilities_fit': res_fit_vis,
               'residual_hologdata': res_hd,
               'residual_fit_hologdata': res_fit_hd,
               'model_hologdata': mod_hd,
               'phs_sol': phs_sol, 'phs_sol_fit': phs_sol_fit}

    return results
