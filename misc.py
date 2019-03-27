"""
Miscellaneous functions.
"""

import re

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

