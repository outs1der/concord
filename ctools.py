import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kepdump
import lcdata
import sys, os
import emcee

# Custom
from burstclass import *
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
#       - function to plot best fit contours/lightcurves



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
        b = ObservedBurst(ob_file, path=source_path)
        obs.append(b)

    return obs



def load_models(runs,
                  batches,
                  model_path = '/home/zacpetej/projects/kepler_grids/gs1826/mean_lightcurves/',
                  param_path = '/home/zacpetej/projects/kepler_grids/gs1826/',
                  source = 'gs1826',
                  basename = 'xrb',
                  params_prefix = 'params_'):
    """
    Loads model parameters and data
    """
    #========================================================
    # Parameters
    #--------------------------------------------------------
    # runs    = [] : list of models to use
    # batches = [] : batches that the models in runs[] belong to
    #========================================================

    models = []

    for i, run in enumerate(runs):
        batch = batches[i]
        param_str = '{prefix}{source}_{batch}.txt'.format(prefix=params_prefix,
                        source=source, batch=batch)
        model_str = '{source}_{batch}_{base}{run}_mean.data'.format(source=source,
                        batch=batch, base=basename, run=run)
        param_file = os.path.join(param_path, param_str)

        # TODO: -load model params
        #       -load recurrence times from summ
        # ===== Load model burst data =====
        ptable = pd.read_table(param_file, delim_whitespace=True)
        # m = KeplerBurst()
        # models.append(m)

    return models
