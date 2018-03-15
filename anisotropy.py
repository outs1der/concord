import numpy as np
from math import *
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from scipy.interpolate import interp1d
import astropy.units as u
import os

CONCORD_PATH = os.environ['CONCORD_PATH']
he16_models = ['he16_a', 'he16_b', 'he16_c', 'he16_c_short', 'he16_d']


def load_he16(model):
    """Reads in and returns the He & Keek (2016) model specified."""
    model_str = model.split('he16_')[1]
    he16_filename = 'anisotropy_{}.txt'.format(model_str)
    he16_filepath = os.path.join(CONCORD_PATH, he16_filename)
    a=ascii.read(he16_filepath)

    return a



def anisotropy(inclination, model='he16_a', test=False):
    '''This function returns the burst and persistent anisotropy factors

    Factors are defined as for Fujimoto et al. 1988, i.e. the xi_b,p such that
    L_b,p = 4pi d^2 xi_b,p F_b,p
    This can be understood as xi_b,p<1 indicating flux that is beamed
    preferentially towards us (so that the luminosity would otherwise be
    exaggerated), and xi_b,p>1 indicating flux beamed preferentially away

    Generate a test of the model using
    xi_b, xi_p = anisotropy(45.,test=True)'''

    global anisotropy_he16

    if test == True:

# Optionally plot a figure showing the behaviour
# want to replicate Figure 2 from fuji88

        _theta = np.arange(50)/49.*pi/2.*u.radian
        ct = np.cos(_theta)

        plt.figure(figsize=(6,10))
        plt.ylim(0,3)
        plt.xlim(0,1)
        plt.xlabel(r"$\cos\theta$")
        plt.ylabel(r"$\xi_p/\xi_b$")

        s='-'
        for m in ['fuji88' ,'he16']:
            xi_p = np.zeros(len(_theta))
            xi_b = np.zeros(len(_theta))
            for i, _th in enumerate(_theta):
                xi_b[i], xi_p[i] = anisotropy(_th,model=m)

            plt.plot(ct,1./xi_b,'b'+s,label=r"$\xi_b^{-1}$ ("+m+")")
            plt.plot(ct,1./xi_p,'r'+s,label=r"$\xi_p^{-1}$ ("+m+")")
            plt.plot(ct,xi_p/xi_b,'g'+s,label=r"$\xi_p/\xi_b$ ("+m+")")

            s=':'

        plt.legend()

# Calculate the values for the passed quantity, and return
# numpy cos will correctly treat the units, so no need to do a conversion
# in that case

    theta = inclination
    if (hasattr(inclination,'unit') == False):
        print ("** WARNING ** assuming inclination in degrees")
        theta *= u.degree

    if model == 'fuji88':
        xi_b = 1./(0.5+abs(np.cos(theta)))
        xi_p = 0.5/abs(np.cos(theta))

        return xi_b, xi_p


    elif model in he16_models:
        model_str = model.split('he16_')[1]   # cut out prefix

        if 'anisotropy_he16' not in globals():
            anisotropy_he16 = {}

        if model_str not in globals()['anisotropy_he16']:
            a = load_he16(model=model)
            v = np.stack((a['col2'],a['col3'],a['col4']),axis=1).T
            anisotropy_he16[model_str] = interp1d(a['col1'],v)

        inv_xi_d, inv_xi_r, inv_xi_p = anisotropy_he16[model_str](theta.to(u.degree))

        with np.errstate(divide='ignore'):
            xi_b = 1./(inv_xi_d+inv_xi_r)
            xi_p = 1./inv_xi_p

        return xi_b, xi_p


    else:
        print ("** ERROR ** model ",model," not yet implemented!")
        return None, None


def inclination(xi, model='he16_a', burst=True):
    '''This function returns the inclination given a burst or persistent
       anisotropy factor. It is the inverse of anisotropy():

    burst=True indicates that the given xi is the burst anisotropy factor
    burst=False indicates that it is the persistent anisotropy factor

    results are returned in units of degrees'''

    if model == 'fuji88':
        if burst:
            return np.arccos(1.0/xi - 0.5) * 180./np.pi * u.degree / u.rad
        else:
            return np.arccos(0.5/xi) * 180./np.pi * u.degree / u.rad

    elif model in he16_models:
        a = load_he16(model=model)

        if burst:
            with np.errstate(divide='ignore'):
                q = 1./(a['col2'] + a['col3'])

            incl_he16 = interp1d(q, a['col1'])

        else:
            with np.errstate(divide='ignore'):
                incl_he16 = interp1d(1./a['col4'], a['col1'])

        return incl_he16(xi) * u.degree

    else:
        print ("** ERROR ** model ",model," not yet implemented!")
        return None



def inclination_ratio(xi_ratio, model='he16_a'):
    '''Returns the inclination corresponding to a given xi_p/xi_b ratio.
        Returned in units of degrees'''
    inc = np.linspace(0*u.deg, 90*u.deg, 180)

    xi_b, xi_p = anisotropy(inclination=inc, model=model)

    with np.errstate(invalid='ignore'):
        x=xi_p/xi_b
    # === check and replace nans with adjacent value ===
    nan_idx = np.where(np.isnan(x))[0]
    x[nan_idx] = x[nan_idx - 1]

    # === account for ratio being outside the model
    try:
        inc_func = interp1d(x=x, y=inc)
        inc_out = inc_func(xi_ratio)
    except ValueError:
        x_max = np.max(x)
        inc_max = inc[np.argmax(x)]
        print('!Ratio not possible in this model! - maximum ratio is {max:.3f} at {i:.3f} deg'.format(max=x_max, i=inc_max))
        return np.nan

    return inc_out * u.degree
