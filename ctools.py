import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kepdump
import lcdata
import sys, os
import
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
# Tools for using X-ray burst matcher Concord
#============================================

# TODO: - function to run mcmc
#       - function to plot best fit contours/lightcurves

def setup(runs,
          batches):
    """
    Loads in observed and modelled data for comparison
    """
    #========================================================
    # Parameters
    #--------------------------------------------------------
    # runs    = [] : list of models to use
    # batches = [] : batches that the models in runs[] belong to
    #========================================================
    obs_path = '/home/zacpetej/projects/codes/kepler_grids/obs_data/gs1826/'
    model_path = '/home/zacpetej/projects/kepler_grids/gs1826/mean_lightcurves/'
    sourcename = 'gs1826'
    basename = 'xrb'

    obs_files = ['gs1826-24_3.530h.dat',
                 'gs1826-24_4.177h.dat',
                 'gs1826-24_5.14h.dat']

    obs = []
    models = []

    for i, run in enumerate(runs):
        batch = batches[i]
        model_str = '{source}_{batch}_{base}{run}_mean.data'.format(source=sourcename,
                        batch=batch, base=basename, run=run)
        # TODO: -load model params
        #       -load recurrence times from summ
        b = ObservedBurst(obs_files[i], path=obs_path)
        m = KeplerBurst()
        obs.append(b)
        models.append(m)

    return obs, models
