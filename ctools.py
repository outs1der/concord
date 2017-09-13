# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kepdump
import lcdata
import sys, os
import emcee
import astropy.units as u
import astropy.constants as const

# homebrew
import burstclass
from misc import try_mkdir
from params import *
from printing import *

#============================================
# Author: Zac Johnston (2017)
# zac.johnston@monash.edu
#
# Tools in progress for using X-ray burst matcher Concord
#============================================
# TODO: - function to run mcmc
#       - save sampler for future analysis
#       -  generalised function to plot best fit contours/lightcurves
#============================================



def load_obs(source='gs1826',
                obs_path = '/home/zacpetej/projects/kepler_grids/obs_data/'):
    """
    Loads observed burst data
    """
    source_path = os.path.join(obs_path, source)
    obs = []
    obs_files = {'gs1826':['gs1826-24_3.530h.dat',
                            'gs1826-24_4.177h.dat',
                            'gs1826-24_5.14h.dat'],
                '4u1820': ['4u1820-303_1.892h.dat',
                            '4u1820-303_2.681h.dat']}

    for ob_file in obs_files[source]:
        b = burstclass.ObservedBurst(ob_file, path=source_path)
        obs.append(b)

    return obs



def load_models(runs,
                  batches,
                  source = 'gs1826',
                  basename = 'xrb',
                  params_prefix = 'params',
                  summ_prefix = 'summ',
                  mean_path = '/home/zacpetej/projects/kepler_grids/gs1826/mean_lightcurves',
                  source_path = '/home/zacpetej/projects/kepler_grids/gs1826/'):
    """
    Loads a set of models (parameters and lightcurves)
    """
    #========================================================
    # Parameters
    #--------------------------------------------------------
    # runs    = [] : list of models to use
    # batches = [] : batches that the models in runs[] belong to (one-to-one correspondence)
    #========================================================
    models = []

    for i, run in enumerate(runs):
        batch = batches[i]

        batch_str = '{source}_{batch}'.format(source=source, batch=batch)
        mean_str = '{batch_str}_{base}{run}_mean.data'.format(batch_str=batch_str, base=basename, run=run)
        param_str = '{prefix}_{batch_str}.txt'.format(prefix=params_prefix, batch_str=batch_str)
        summ_str = '{prefix}_{batch_str}.txt'.format(prefix=summ_prefix, batch_str=batch_str)

        param_file = os.path.join(source_path, param_str)
        mean_file = os.path.join(mean_path, mean_str)   # currently not used
        summ_file = os.path.join(source_path, summ_str)

        #----------------------------------
        # TODO: - account for different Eddington composition
        #----------------------------------

        mtable = pd.read_csv(summ_file)
        ptable = pd.read_table(param_file, delim_whitespace=True)  # parameter table
        idx = np.where(ptable['id'] == run)[0][0]    # index of model/run
        # NOTE: Assumes that models in ptable exactly match those in mtable

        # ====== Extract model parameters/properties ======
        xi = 1.12             # currently constant, could change in future
        R_NS = 12.1 * u.km    # add this as colum to parameter file (or g)?
        M_NS = ptable['mass'][idx] * const.M_sun
        X = ptable['x'][idx]
        Z = ptable['z'][idx]
        lAcc = ptable['accrate'][idx] * ptable['xi'][idx]    # includes xi_p multiplier
        opz = 1./sqrt(1.-2.*const.G*M_NS/(const.c**2*R_NS))
        g = const.G*M_NS/(R_NS**2/opz)
        tdel = mtable['tDel'][idx]
        tdel_err = mtable['uTDel'][idx]

        m = burstclass.KeplerBurst(filename = mean_str,
                        path = mean_path,
                        tdel = tdel,
                        tdel_err = tdel_err,
                        g = g,
                        R_NS = R_NS,
                        xi = xi,
                        lAcc = lAcc,
                        Z = Z,
                        X = X)

        models.append(m)

    return models
