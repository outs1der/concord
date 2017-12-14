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



def full_string(run, batches, step, con_ver, source='gs1826'):
    """========================================================
    constructs a standardised string for a batch model
    ========================================================
    special cases:
        - run = 0 : don't include run
        - con_ver = 0 : don't include con version
    ========================================================"""
    batches = expand_batches(batches, source)
    b_string = daisychain(batches)

    if run == 0:
        run_str = ''
    else:
        run_str = '_R{run}'.format(run=run)

    if step == 0:
        step_str = ''
    else:
        step_str = '_S{step}'.format(step=step)

    if con_ver == 0:
        con_str = ''
    else:
        con_str = '_C{cv:02}'.format(cv=con_ver)

    full_str = '{src}_{b_str}{run_str}{step_str}{c_str}'.format(src=source, b_str=b_string,
                                                    run_str=run_str, step_str=step_str, c_str=con_str)

    return full_str



def triplet_string(batches, source='gs1826'):
    """========================================================
    Returns triplet string, e.g.: gs1826_12-13-14
    ========================================================"""
    batches = expand_batches(batches, source)
    batch_daisy = daisychain(batches)
    triplet_str = '{source}_{batches}'.format(source=source, batches=batch_daisy)

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



def get_nruns(batch, source='gs1826', **kwargs):
    """========================================================
    Returns number of runs (models) in a given batch
    ========================================================"""
    path = kwargs.get('path', GRIDS_PATH)
    print(f'get nruns: {source}')
    batch_str = full_string(run=0, batches=[batch], step=0,
                            source=source, con_ver=0)
    filename = 'params_{batch_str}.txt'.format(batch_str=batch_str)
    filepath = os.path.join(path, source, 'params', filename)

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
        print('Creating directory  {}'.format(path))
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
