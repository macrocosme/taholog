"""
"""

import pickle
import logging
import numpy as np
import scipy.constants as const

#from scipy.linalg import ldl
from numpy.lib.scimath import sqrt as csqrt

import holog
from taholog import stations, beam

def covariance_matrix(holog_data, pqr):
    r"""
    Computes the conditional variance of the error term given the beam model.
    This is used to apply general least squares to the problem.
    """
    
    l_rad = holog_data.l_rad
    m_rad = holog_data.m_rad
    
    nbeams = len(l_rad)
    
    beam_position = np.array([l_rad, m_rad]).T
    
    cov = np.zeros((nbeams,nbeams), dtype=np.complex64)

    amp = np.ones(len(pqr))    

    for i in range(nbeams):
    
        phase = np.dot(pqr[:,:-1]*holog_data.freq_hz/const.c, beam_position[i].T)*2.*np.pi
        cov[i] = beam.simple_beam_model(np.array([l_rad, m_rad]).T, 
                                        pqr[:,:-1]*holog_data.freq_hz/const.c,
                                        amp,
                                        phase)
        
    return cov

def fit_lstsq(hd, xs, ys, weighted=True, cov_err=0):
    """
    Solves the problem of finding the amplitudes and phases.

    :param hd: Holography data in HologData format.
    :type hd: HologData
    :param xs: X coordinate of the station locations.
    :type xs: np.array
    :param ys: Y coordinate of the station locations.
    :type ys: np.array
    """

    freq_hz = hd.freq_hz

    if weighted == True:
        w = np.diag(np.c_[hd.sigma_vis.real, hd.sigma_vis.imag].ravel()**-2.0)
    else:
        w = np.diag(np.ones_like(zip(hd.sigma_vis.real, hd.sigma_vis.imag)).ravel())
    w = np.matrix(np.array(w))

    b_vect = np.matrix(hd.vis)
    M = m_matrix(hd.l_rad, hd.m_rad,
                 xs, ys, freq_hz)

    M, b_vect = real_linear_problem_from_complex(M, b_vect.T)

    if hasattr(cov_err, "__len__"):
        cov_err_real = cov_err.real
        cov_err_imag = cov_err.imag
        lu, d, perm = ldl(cov_err_real + 1j*cov_err_imag, lower=0, hermitian=False)
        cov_err_chol = lu.dot(csqrt(d))
        cov_err_chol_inv = np.linalg.inv(cov_err_chol)
        cov_err_mat = np.matrix(real_linear_problem_from_complex_matrix(cov_err_chol_inv))
    else:
        cov_err_mat = 1.

    g_vect = np.linalg.lstsq(np.array(w*cov_err_mat*np.matrix(M)),
                             np.array(w*cov_err_mat*np.matrix(b_vect).T))[0]
    cov = np.linalg.inv(np.matrix(M).T*w*M) # Covariance matrix
    M_beta = np.diag(cov)

    # Make these complex quantities again
    g_vect = g_vect[0::2] + 1.0j*g_vect[1::2]
    M_beta = csqrt(np.array(M_beta)[0::2]) + 1.0j*csqrt(np.array(M_beta)[1::2])
    complex_cov = cov[0::2] + 1.0j*cov[1::2]

    return g_vect, M_beta, complex_cov

def fit_lstsq_complex(hd, xs, ys, weighted=True):
    """
    Solves the problem of finding the amplitudes and phases.

    :param hd: Holography data in HologData format.
    :type hd: HologData
    :param xs: X coordinate of the station locations.
    :type xs: np.array
    :param ys: Y coordinate of the station locations.
    :type ys: np.array
    """

    freq_hz = hd.freq_hz

    if weighted == True:
        w = np.diag(np.power(hd.sigma_vis.real + 1.j*hd.sigma_vis.imag, -2.0))
    else:
        w = np.diag(np.ones_like(hd.sigma_vis.real + 1.j*hd.sigma_vis.imag))

    # Make it a matrix.
    w = np.matrix(np.array(w))

    b_vect = np.matrix(hd.vis)
    M = m_matrix(hd.l_rad, hd.m_rad,
                 xs, ys, freq_hz)

    g_vect = np.linalg.lstsq(np.array(w*np.matrix(M)),
                             np.array(w*np.matrix(b_vect).T))[0]

    cov = np.linalg.inv(np.matrix(M).T*w*M) # Covariance matrix for the complex problem

    return g_vect, cov

def m_matrix(l, m, x, y, nu):
    """
    """

    L_k = np.matrix(l)
    M_k = np.matrix(m)
    X_p = np.matrix(x)
    Y_p = np.matrix(y)

    arg = (-2.*np.pi*1.j*nu*(L_k.T*X_p + M_k.T*Y_p)/const.c)
    result = np.exp(arg)/len(x)
    return result

def make_solutions(g, m):
    """
    Use the complex solutions to the least squares problem to generate amplitude and phase solutions.
    
    :param g: Complex gain solutions to the least squares problem.
    :type g:
    :param m: Covariance matrix of the complex gain solutions.
    :type m:
    :returns: Amplitude and phase solutions for every station with their respective errors.
    :rtype:
    """

    gain = np.sort(abs(g))
    mgain = np.mean(gain)
    cor_amp = abs(g)/mgain
    cor_amp_err = (1.0/np.sqrt(2.0))*abs(m/mgain)
    cor_phs = np.rad2deg(np.angle(g))
    #cor_phs_err = np.rad2deg(2.0*np.arcsin(cor_amp_err/(2.0*cor_amp.T[0])))
    r = g.imag/g.real
    r = r[:,0]
    cor_phs_err = 1./(1. + np.power(r, 2.))/(g[:,0].real)*np.sqrt(np.power(m.imag, 2.) + np.power(r*m.real, 2.))

    return cor_amp, cor_amp_err, cor_phs, cor_phs_err

def real_linear_problem_from_complex(matrix_complex, vector_complex):
    """
    Rewrites a complex valued matrix equation as a real-valued
    equivalent. The structure of the returned vector is: [Re0, Im0,
    Re1, Im1,...,ReN, ImN]. The returned matrix has twice the
    dimensions of the input matrix.
    """

    vr = np.zeros((vector_complex.shape[0]*2), dtype=np.float64)
    vre = vector_complex.real
    vim = vector_complex.imag

    vr[0::2] = vre.squeeze()
    vr[1::2] = vim.squeeze()

    Mr = real_linear_problem_from_complex_matrix(matrix_complex)

    return (Mr, vr)

def real_linear_problem_from_complex_matrix(matrix_complex):
    """
    """

    Mr = np.zeros((matrix_complex.shape[0]*2, matrix_complex.shape[1]*2), dtype=np.float64)
    
    Mre = matrix_complex.real
    Mim = matrix_complex.imag

    Mr[0::2, 0::2] = +Mre
    Mr[0::2, 1::2] = -Mim
    Mr[1::2, 0::2] = +Mim
    Mr[1::2, 1::2] = +Mre

    return Mr

def uvhol(target, out, weighted=True, use_cov=True):
    r'''
    Finds the phase and amplitude of the stations given a .uvhol data file.
    The output will be a dictionary with a key for each station and for each station the following keys:
    'amp'              : station amplitude. One value per subband.
    'phs'              : station phase. One value per subband.
    'amp_err'          : error on the amplitude. One value per subband.
    'phs_err'          : error on the phase. One value per subband.
    'freq'             : frequency in Hz. One value per subband.
    'reference_station': station used to reference the visibilities.
    '''

    logger = logging.getLogger(__name__)

    # Load holography results
    hd = holog.uvhol.read_uvhol_file(target)

    if hd[0].vis.size == 0:
        logger.info('Holography visibilities are empty.')
        logger.info('Will not solve.')
        raise ValueError('Holography visibilities for file {0} are empty.'.format(hdfile))

    # Prepare station locations
    logger.info('Preparing station locations.')
    antennas = np.array(hd[0].antenna.split(','))
    nant = len(antennas)

    pqr = stations.station_offsets_pqr(hd[0].antenna)
    xs = pqr[:,0]
    ys = pqr[:,1]

    cov_err = 0
    if use_cov:
        cov_err = covariance_matrix(hd[0], pqr)

    # Solve
    logger.info('Solving.')
    #g, m = fit_lstsq_complex(hd[0], xs, ys, weighted=weighted)
    #cor_amp, cor_amp_err, cor_phs, cor_phs_err = make_solutions(g, csqrt(np.diag(m)))
    g, b, m = fit_lstsq(hd[0], xs, ys, weighted=weighted, cov_err=cov_err)
    cor_amp, cor_amp_err, cor_phs, cor_phs_err = make_solutions(g, csqrt(np.diag(m)))

    # Save solutions
    keys = antennas
    sols = dict.fromkeys(keys)
    skey = ['amp', 'ph', 'amp_err', 'ph_err', 'freq', 'reference_station']
    for i,k in enumerate(sols.keys()):
        sols[k] = dict.fromkeys(skey)
        sols[k]['amp'] = cor_amp[np.where(keys == k)[0]][0][0]
        sols[k]['ph'] = cor_phs[np.where(keys == k)[0]][0][0]
        sols[k]['amp_err'] = cor_amp_err[np.where(keys == k)[0]][0]
        sols[k]['ph_err'] = cor_phs_err[np.where(keys == k)[0]][0]
        sols[k]['freq'] = hd[0].freq_hz
        sols[k]['reference_station'] = hd[0].ref_ants[0]

    with open('{0}.pickle'.format(out), 'wb') as outfile:

        logger.info('Saving solutions to: {0}.pickle .'.format(out))
        pickle.dump(sols, outfile, protocol=pickle.HIGHEST_PROTOCOL)

