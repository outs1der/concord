import numpy as np
import pandas as pd
import sys, os

class Source(object):
    """========================================================
    Class that holds the setup parameters for a given source object
    ========================================================"""
    def __init__(self, source):
        self.source = source
        self.obs_files = get_obs_files(source=source)
        self.mdots = get_mdots(source=source)
        self.pos = get_pos(source=source)
        self.n_epochs = len(self.mdots)


def check_source(source):
    """========================================================
    Check if source is defined
    ========================================================"""
    check_source(source=source)
    defined_sources = ['gs1826', '4u1820']
    if source not in defined_sources:
        print(f'ERROR: source ({source}) not in {defined_sources}')
        sys.exit()


def get_obs_files(source):
    """========================================================
    Names of files containing observation data
    ========================================================"""
    check_source(source=source)
    obs_files = {'gs1826':['gs1826-24_3.530h.dat',
                            'gs1826-24_4.177h.dat',
                            'gs1826-24_5.14h.dat'],
                '4u1820': ['4u1820-303_1.892h.dat',
                            '4u1820-303_2.681h.dat']}
    return obs_files[source]


def get_mdots(source):
    """========================================================
    Accretion rates of different epochs (Eddington fraction)
    ========================================================"""
    check_source(source=source)
    mdots = {'gs1826':[0.0796, 0.0692, 0.0513],
             '4u1820':[0.226, 0.144]}
    return mdots[source]


def get_pos(source):
    """========================================================
    Initial source params [distance (kpc), inclination (deg), redshift]
    ========================================================"""
    check_source(source=source)
    pos = {'gs1826':[6.0, 60., 1.35],
           '4u1820':[5.0, 60., 1.35]}
    return pos[source]


def get_n_epochs(source):
    """========================================================
    Get number of epohcs being matches
    ========================================================"""
    check_source(source=source)
    n_epochs = {'gs1826': 3,
                '4u1820': 2}
    return n_epochs[source]
