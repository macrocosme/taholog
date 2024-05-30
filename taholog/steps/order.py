"""
"""

import re
import pickle
import logging
import numpy as np

from lmfit.models import PolynomialModel
npfit = False
#try:
#    from lmfit.models import PolynomialModel
#    npfit = False
#except ImportError:
#    npfit = True

def unwrap_phase_iter(phase, discont_step=np.pi/2., deg=0):
    """
    """

    disconts = list(np.arange(-np.pi, np.pi+discont_step, discont_step))
    disconts.pop(len(disconts)//2)

    residual = np.zeros(len(disconts)+1, dtype=np.float32)
    x = np.arange(0,len(phase))

    polyfit = np.polyfit(x, phase, deg=deg)
    residual[0] = np.std(phase - np.poly1d(polyfit)(x))

    for i,discont in enumerate(disconts):

        pu = np.unwrap(phase, discont=discont)

        polyfit = np.polyfit(x, pu, deg=deg)
        residual[i+1] = np.std(pu - np.poly1d(polyfit)(x))

    if residual.argmin() == 0:
        return False
    else:
        return disconts[residual.argmin()-1]

def main(files, out, phase_ref_station, degree):
    r'''
    '''

    logger = logging.getLogger(__name__)

    # Make arrays to keep the solutions.
    data = np.empty(len(files), dtype=object)
    stations = np.empty(len(files), dtype=object)
    freqs = np.empty(len(files), dtype=object)
    refss = np.empty(len(files), dtype=object)
    times = np.empty(len(files), dtype=object)

    # Load the solutions.
    print ()
    print (f"solution files: {files}")
    for i,f in enumerate(files):
        data[i] = pickle.load(open(f, "rb")) # One time, one frequency, one correlation, all stations
        stations[i] = list(data[i].keys())
        times[i] = re.findall('t\d+', f)[0][1:]
        freqs[i] = data[i][stations[i][0]]['freq']
        refss[i] = data[i][stations[i][0]]['reference_station']

    # Determine how many stations and frequencies there are
    ustats = np.unique(stations)[0]
    ufreqs = np.unique(freqs)
    urefss = np.unique(refss)
    utimes = np.unique(times)
    taxis = max(map(int, utimes)) + 1

    logger.info('Frequencies loaded: {0}'.format(ufreqs))

    # Join all solutions for every station
    sols = dict.fromkeys(ustats)
    skey = ['amp', 'ph', 'ph_unwrap', 'ph_ref', 
            'tau', 'amp_err', 'ph_err', 'ph_ref_err', 
            'tau_err', 'freq', 'reference_station', 
            'ph_ref_station']

    for i,k in enumerate(sols.keys()):
        print (f"{k}, {np.zeros((taxis, len(urefss)), dtype=np.float64).shape}")

        sols[k] = dict.fromkeys(skey)
        sols[k]['amp'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['ph'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['ph_unwrap'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['tau'] = np.zeros((taxis, len(urefss)), dtype=np.float64)
        sols[k]['tau0'] = np.zeros((taxis, len(urefss)), dtype=np.float64)
        sols[k]['amp_err'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['ph_err'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['tau_err'] = np.zeros((taxis, len(urefss)), dtype=np.float64)
        sols[k]['tau0_err'] = np.zeros((taxis, len(urefss)), dtype=np.float64)
        sols[k]['tau_fit'] = np.zeros((taxis, len(urefss), len(ufreqs)), dtype=np.float64)
        sols[k]['freq'] = np.array(ufreqs, dtype=np.float)
        sols[k]['reference_station'] = urefss
        sols[k]['ph_ref_station'] = phase_ref_station

    for i,d in enumerate(data):
        for j,s in enumerate(ustats):
            for k,r in enumerate(urefss):

                tindx = int(re.findall('t\d+', files[i])[0][1:])
                findx = np.where(sols[s]['freq'] == d[s]['freq'])
                rindx = np.where(sols[s]['reference_station'] == d[s]['reference_station'])

                sols[s]['amp'][tindx,rindx,findx] = d[s]['amp']
                sols[s]['ph'][tindx,rindx,findx] = d[s]['ph']
                sols[s]['amp_err'][tindx,rindx,findx] = d[s]['amp_err']
                sols[s]['ph_err'][tindx,rindx,findx] = d[s]['ph_err']

    # Unwrap phases.
    for station in sols.keys():
        for t in range(sols[station]['ph'].shape[0]):
            for r in range(len(urefss)):

                phase = np.deg2rad(sols[station]['ph'][t,r,:])
                discont = unwrap_phase_iter(phase, discont_step=np.pi/4., deg=0)
                if discont:
                    phase = np.unwrap(phase, discont=discont)
                #phase -= phase.mean()

                sols[station]['ph_unwrap'][t,r,:] = phase

    if phase_ref_station != '':

        phase_ref = sols[phase_ref_station]['ph_unwrap']
        phase_ref_err = np.deg2rad(sols[phase_ref_station]['ph_err'])

        for station in sols.keys():

            delay = np.zeros(sols[station]['ph'].shape[:-1])
            delay0 = np.zeros(sols[station]['ph'].shape[:-1])
            delay_err = np.zeros(sols[station]['ph'].shape[:-1])
            delay0_err = np.zeros(sols[station]['ph'].shape[:-1])
            delay_fit = np.zeros(sols[station]['ph'].shape)
            delay_fix = np.zeros(sols[station]['ph'].shape[:-1])
            delay0_fix = np.zeros(sols[station]['ph'].shape[:-1])
            delay_err_fix = np.zeros(sols[station]['ph'].shape[:-1])
            delay0_err_fix = np.zeros(sols[station]['ph'].shape[:-1])
            delay_fit_fix = np.zeros(sols[station]['ph'].shape)

            phase_sol = sols[station]['ph_unwrap'][:,:,:] - phase_ref[:,:,:]

            phase_sol_err = np.sqrt(np.power(np.deg2rad(sols[station]['ph_err']), 2.) + np.power(phase_ref_err, 2.))

            if station != phase_ref_station:

                for t in range(sols[station]['ph'].shape[0]):
                    for r in range(len(urefss)):

                        if not npfit:
                            mod = PolynomialModel(degree)

                            # Fit a polynomial with a phase offset.
                            params = mod.make_params()
                            params = mod.guess(phase_sol[t,r], x=sols[station]['freq'])
                            fit = mod.fit(phase_sol[t,r][1:-1], x=sols[station]['freq'][1:-1], 
                                          params=params)#, weights=np.power(phase_sol_err[t,r], -2.))

                            # delay[t,r] = fit.params['c1'].value/(2.*np.pi)
                            # delay0[t,r] = fit.params['c0'].value
                            delay[t,r] = fit.params['c1']/(2.*np.pi)
                            delay0[t,r] = fit.params['c0']
                            delay_err[t,r] = fit.params['c1'].stderr/(2.*np.pi)
                            delay0_err[t,r] = fit.params['c0'].stderr
                            delay_fit[t,r] = fit.eval(x=sols[station]['freq'])

                            # Fit a polynomial with a fixed phase offset of zero.
                            if station != phase_ref_station:
                                params['c0'].set(value=0, vary=False)
                                fit = mod.fit(phase_sol[t,r][1:-1], x=sols[station]['freq'][1:-1], 
                                              params=params)#, weights=np.power(phase_sol_err[t,r], -2.))
                                # delay_fix[t,r] = fit.params['c1'].value/(2.*np.pi)
                                # delay0_fix[t,r] = fit.params['c0'].value
                                delay[t,r] = fit.params['c1']/(2.*np.pi)
                                delay0[t,r] = fit.params['c0']
                                delay_err_fix[t,r] = fit.params['c1'].stderr/(2.*np.pi)
                                delay0_err_fix[t,r] = fit.params['c0'].stderr
                                delay_fit_fix[t,r] = fit.eval(x=sols[station]['freq'])
                            else:
                                delay_fix[t,r] = 0
                                delay0_fix[t,r] = 0
                                delay_err_fix[t,r] = 0
                                delay0_err_fix[t,r] = 0
                                delay_fit_fix[t,r] = [0]*len(sols[station]['freq'])

            sols[station]['tau'] = delay    # Time delay with non-zero phase offset.
            sols[station]['tau0'] = delay0
            sols[station]['tau_err'] = delay_err
            sols[station]['tau0_err'] = delay0_err
            sols[station]['tau_fit'] = delay_fit
            sols[station]['ph_ref'] = phase_sol
            sols[station]['ph_ref_err'] = phase_sol_err
            sols[station]['tau_fix'] = delay_fix # Time delay with zero phase offset.
            sols[station]['tau0_fix'] = delay0_fix
            sols[station]['tau_fix_err'] = delay_err_fix
            sols[station]['tau0_fix_err'] = delay0_err_fix
            sols[station]['tau_fix_fit'] = delay_fit_fix

    # Save the joint solutions
    with open(out, 'wb') as out:
        pickle.dump(sols, out, protocol=pickle.HIGHEST_PROTOCOL)
