"""
Miscellaneous functions.
"""

import re
import numpy as np

from functools import reduce

def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    
    :param s: String
    :returns: List with strings and integers.
    :rtype: list
    
    :Example:
    
    >>> alphanum_key('z23a')
    ['z', 23, 'a']
    
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def factors(n):
    """
    Returns the factors by which n can be divided.
    """

    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def natural_sort(list):
    """ 
    Sort the given list in the way that humans expect. \
    Sorting is done in place.
    
    :param list: List to sort.
    :type list: list
    
    :Example:
    
    >>> my_list = ['spec_3', 'spec_4', 'spec_1']
    >>> natural_sort(my_list)
    >>> my_list
    ['spec_1', 'spec_3', 'spec_4']
    """

    list.sort(key=alphanum_key)

def tryint(str):
    """
    Returns an integer if `str` can be represented as one.
    
    :param str: String to check.
    :type str: string
    :returns: True is str can be cast to an int.
    :rtype: int
    """

    try:
        return int(str)
    except:
        return str

def gaussian(x, sigma, center, amplitude):
    """
    Gaussian function in one dimension.
    
    :param x: x values for which to evaluate the Gaussian.
    :type x: array
    :param sigma: Standard deviation of the Gaussian.
    :type sigma: float
    :param center: Center of the Gaussian.
    :type center: float
    :param amplitude: Peak value of the Gaussian.
    :type amplitude: float
    :returns: Gaussian function of the given amplitude and standard deviation evaluated at x.
    :rtype: array
    """

    return amplitude*np.exp(-np.power((x - center), 2.)/(2.*np.power(sigma, 2.)))

def gauss_area(amplitude, sigma):
    """
    Returns the area under a Gaussian of a given amplitude and sigma.
    
    .. math:
    
        Area=\\sqrt(2\\pi)A\\sigma
        
    :param amplitude: Amplitude of the Gaussian, :math:`A`.
    :type A: array
    :param sigma: Standard deviation fo the Gaussian, :math:`\\sigma`.
    :type sigma: array
    :returns: The area under a Gaussian of a given amplitude and standard deviation.
    :rtype: array
    """
    
    return amplitude*sigma*np.sqrt(2.*np.pi)
