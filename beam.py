"""
"""

import numpy as np

def simple_beam_model(beam_positions, pqrs, amplitudes, phases):
    """
    Evaluates the tied-array beam at the beam_positions given
    the station positions, pqrs, amplitudes and phases.
    """
    
    expf = []
    for i in range(len(beam_positions)):
        expf.append(np.dot(pqrs, beam_positions[i].T))
    expf = np.array(expf)
    
    bmod = np.ma.sum(amplitudes*np.exp(-2.j*np.pi*(expf) + phases*1j), axis=1)/len(pqrs)
    
    return bmod
