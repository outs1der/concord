import numpy as np
import pandas as pd
import sys, os

GRIDS_PATH = os.environ['KEPLER_GRIDS']
CONCORD_PATH = os.environ['CONCORD_PATH']

def construct_t_params(n):
    """========================================================
    Creates list of time-param labels (t1, t2, t3...)
    ========================================================
    Parameters
    --------------------------------------------------------
    n = int  : number of time parameters
    ========================================================"""
    t_params = []

    for i in range(n):
        tname = 't' + str(i+1)
        t_params.append(tname)

    return t_params



def full_string(batches, source, run=None, step=None, con_ver=None):
    """========================================================
    constructs a standardised string for a batch model
    ========================================================
    special cases:
        - run = 0 : don't include run
        - con_ver = 0 : don't include con version
    ========================================================"""
    batches = expand_batches(batches, source)
    batch_str = daisychain(batches)

    if run == None:
        run_str = ''
    else:
        run_str = f'_R{run}'

    if step == None:
        step_str = ''
    else:
        step_str = f'_S{step}'

    if con_ver == None:
        con_str = ''
    else:
        con_str = f'_C{con_ver:02}'

    full_str = f'{source}_{batch_str}{run_str}{step_str}{con_str}'
    return full_str



def triplet_string(batches, source):
    """========================================================
    Returns triplet string, e.g.: gs1826_12-13-14
    ========================================================"""
    batches = expand_batches(batches, source)
    batch_str = daisychain(batches)
    triplet_str = f'{source}_{batch_str}'

    return triplet_str



def expand_batches(batches, source):
    """========================================================
    Checks format of 'batches' parameter and returns relevant array
    if batches is arraylike: keep
    if batches is integer N: assume first batch of batch set
    ========================================================"""
    N = {'gs1826': 3, '4u1820': 2}  # number of epochs
    special = {4, 7}    # special cases (reverse order)

    if type(batches) == int or type(batches) == np.int64:   # assume batches gives first batch
        if batches in special and source == 'gs1826':
            batches_out = np.arange(batches, batches-3, -1)
        else:
            batches_out = np.arange(batches, batches+N[source])

    elif type(batches) == list   or   type(batches) == np.ndarray:
        batches_out = batches

    else:
        print('Invalid type(batches). Must be array-like or int')
        sys.exit()

    return batches_out


def expand_runs(runs):
    """========================================================
    Checks format of 'runs' parameter and returns relevant array
    if runs is arraylike: keep
    if runs is integer N: assume there are N runs from 1 to N
    ========================================================"""
    if type(runs) == int:   # assume runs = n_runs
        runs_out = np.arange(1, runs+1)
    elif type(runs) == list   or   type(runs) == np.ndarray:
        runs_out = runs
    else:
        print('Invalid type(runs)')
        sys.exit()

    return runs_out



def get_nruns(batch, source, **kwargs):
    """========================================================
    Returns number of runs (models) in a given batch
    ========================================================"""
    path = kwargs.get('path', GRIDS_PATH)
    print(f'get nruns: {source}')
    batch_str = full_string(batches=[batch], source=source)
    filename = f'params_{batch_str}.txt'
    filepath = os.path.join(path, 'sources', source, 'params', filename)

    params = pd.read_table(filepath, delim_whitespace=True)

    return len(params.iloc[:,0])



def daisychain(daisies, delim='-'):
    """========================================================
    returns a string of daisies seperated by a delimiter (e.g. 1-2-3-4)
    ========================================================
    daisies  = [int]
    delim    = str     : delimiter to place between IDs
    ========================================================"""
    daisy = ''

    for i in daisies[:-1]:
        daisy += str(i)
        daisy += delim

    daisy += str(daisies[-1])

    return daisy



def try_mkdir(path, skip=False):
    """========================================================
    Tries to create directory, skip if exists or ask to overwrite
    ========================================================"""
    try:
        print(f'Creating directory  {path}')
        subprocess.run(['mkdir', path], check=True)
    except:
        if skip:
            print('Directory already exists - skipping')
        else:
            print('Directory exists')
            cont = input('Overwrite? (DESTROY) [y/n]: ')

            if cont == 'y' or cont == 'Y':
                subprocess.run(['rm', '-r', path])
                subprocess.run(['mkdir', path])
            elif cont =='n' or cont == 'N':
                sys.exit()