import numpy as np
import pandas as pd
import sys, os

class Source(object):
    """========================================================
    Class that holds the setup parameters for a given source object
    ========================================================"""

    def __init__(self, source):
        sources = ['gs1826', '4u1820']
        self.source = source

        if source not in sources:
            print(f'ERROR: source given ({source}) not in {sources}')
            sys.exit()

        # ===== define various source properties here =====
        # === Names of files containing observation data ===
        obs_files = {'gs1826':['gs1826-24_3.530h.dat',
                                'gs1826-24_4.177h.dat',
                                'gs1826-24_5.14h.dat'],
                    '4u1820': ['4u1820-303_1.892h.dat',
                                '4u1820-303_2.681h.dat']}

        # accretion rates of different epochs
        mdots = {'gs1826':[0.0796, 0.0692, 0.0513],
                 '4u1820':[0.226, 0.144]}

        self.obs_files = obs_files[source]
        self.mdots = mdots[source]
        self.n_epochs = len(self.mdots)
