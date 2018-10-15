# Various utilities moved from burstclass.py
#
# def g(M,R,Newt=False,units='cm/s^2'):
# def opz(M,R):
# def calc_mr(g,opz):
# def solve_radius(M,R_Newt,eta=1e-6):
# def decode_LaTeX(string):

import numpy as np
from math import *
import astropy.constants as const
import astropy.units as u
import re

# ------- --------- --------- --------- --------- --------- --------- ---------

def g(M,R,Newt=False,units='cm/s^2'):
    '''
    This function calculates the surface gravity given a mass M and radius R,
    using the Newtonian expression if the flag Newt is set to True

    The result is returned in units of cm/s^2 by default
    '''

    if Newt:
        return (const.G*M/R**2).to(units)
    else:
        #        return const.G*M/(R**2*sqrt(1.-2.*const.G*M/(const.c**2*R))).to(units)
        return opz(M,R)*g(M,R,Newt=True).to(units)

# ------- --------- --------- --------- --------- --------- --------- ---------

def opz(M,R):
    '''
    This function calculates the gravitational redshift 1+z
    '''

    return 1./sqrt(1.-2.*const.G*M/(const.c**2*R))

# ------- --------- --------- --------- --------- --------- --------- ---------

def calc_mr(g,opz):
    ''''
    this function calculates neutron star mass and radius given a surface
    gravity and redshift
    '''

    # First some checks

    if hasattr(opz,'unit'):
        assert (opz.unit == '')
    try:
        test = g.to('cm / s2')
    except ValueError:
        print ("Incorrect units for surface gravity")
        return -1, -1

    # Now calculate the mass and radius and convert to cgs units

    R_NS = (const.c**2*(opz**2-1)/(2.*g*opz)).to(u.cm)
    M_NS = (g*R_NS**2/(const.G*opz)).to(u.g)

    return M_NS, R_NS

# ------- --------- --------- --------- --------- --------- --------- ---------

def solve_radius(M,R_Newt,eta=1e-6):
    '''
    This routine determines the GR radius given the NS mass and Newtonian
    radius, assuming the GR and Newtonian masses are identical

    Solving is tricky so we just use an iterative approach
    '''

    R_NS = R_Newt	# trial
    while (abs(g(M,R_NS)-g(M,R_Newt,Newt=True))/g(M,R_NS) > eta):
        R_NS = R_Newt*sqrt(opz(M,R_NS))

    return R_NS

# ------- --------- --------- --------- --------- --------- --------- ---------

def decode_LaTeX(string):
    '''
    This function converts a LaTeX numerical value (with error) to one
    or more floating point values
    '''

    assert (type(string) == str) or (type(string) == np.str_)

    # Look for the start of a LaTeX numerical expression
    # We no longer explicitly look for the $ sign, as this pattern can match a
    # general numeric string. We also include the mantissa as optional

    val_match = re.search('([0-9]+(\.[0-9]+)?)',string)

    # If not found, presumably you just have a numerical expression as string

    if val_match == None:
        return float(string), None

    # Otherwise, convert the first part to a float, and look for the error

    val = float(val_match.group(1))
    err_match = re.search('pm *([0-9]+\.[0-9]+)',string)

    if err_match == None:
        return val, None
    else:
        return val, float(err_match.group(1))

