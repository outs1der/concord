# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import emcee
import subprocess
import astropy.units as u
import astropy.constants as const
from math import sqrt
from chainconsumer import ChainConsumer

# homebrew
import burstclass

#============================================
# Author: Zac Johnston (2017)
# zac.johnston@monash.edu
# Tools in progress for using X-ray burst matcher Concord
#============================================
# TODO:
#
# --- Functions ---
#
# def plot_walkers():
#     """
#     plots walkers of mcmc chain
#     """
#
# def best_fits():
#     """
#     extracts best fits from given batch, prints params
#     """
#============================================


#===================================================
# GLOBAL PATHS
#---------------------------------------------------
# If you wish to use a different path for a specific function call,
# include it as the parameter 'path' when calling the function
#---------------------------------------------------
# You need to set these as bash environment variables
GRIDS_PATH = os.environ['KEPLER_GRIDS']
CONCORD_PATH = os.environ['CONCORD_PATH']
#===================================================

def load_obs(source='gs1826',
                **kwargs):
    """
    ========================================================
    Loads observed burst data
    ========================================================
    Parameters
    --------------------------------------------------------
    source   = str : astrophysical source being matched (gs1826, 4u1820)
    ========================================================
    """
    obs = []
    path = kwargs.get('path', GRIDS_PATH)
    obs_path = os.path.join(path, 'obs_data')
    source_path = os.path.join(obs_path, source)

    obs_files = {'gs1826':['gs1826-24_3.530h.dat',
                            'gs1826-24_4.177h.dat',
                            'gs1826-24_5.14h.dat'],
                '4u1820': ['4u1820-303_1.892h.dat',
                            '4u1820-303_2.681h.dat']}

    for ob_file in obs_files[source]:
        b = burstclass.ObservedBurst(ob_file, path=source_path)
        obs.append(b)

    obs = tuple(obs)

    return obs



def load_models(runs,
                  batches,
                  source = 'gs1826',
                  basename = 'xrb',
                  params_prefix = 'params',
                  summ_prefix = 'summ',
                  **kwargs):
    """
    ========================================================
    Loads a set of models (parameters and lightcurves)
    ========================================================
    Parameters
    --------------------------------------------------------
    runs    = [] : list of models to use (assumed to be identical if only one given)
    batches = [] : batches that the models in runs[] belong to (one-to-one correspondence)
    ========================================================
    """
    runs = expand_runs(runs)
    batches = expand_batches(batches, source)
    models = []
    path = kwargs.get('path', GRIDS_PATH)

    #
    if len(runs) == 1:
        nb = len(batches)
        runs = np.full(nb, runs[0])

    for i, run in enumerate(runs):
        batch = batches[i]

        batch_str = '{source}_{batch}'.format(source=source, batch=batch)
        mean_str = '{batch_str}_{base}{run}_mean.data'.format(batch_str=batch_str, base=basename, run=run)
        param_str = '{prefix}_{batch_str}.txt'.format(prefix=params_prefix, batch_str=batch_str)
        summ_str = '{prefix}_{batch_str}.txt'.format(prefix=summ_prefix, batch_str=batch_str)

        source_path = os.path.join(path, source)
        param_file = os.path.join(source_path, 'params', param_str)
        summ_file = os.path.join(source_path, 'summary', summ_str)
        mean_path = os.path.join(source_path, 'mean_lightcurves', batch_str)
        #----------------------------------
        # TODO: - account for different Eddington composition
        #----------------------------------

        mtable = pd.read_table(summ_file, delim_whitespace=True)
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
        tdel = mtable['tDel'][idx]/3600
        tdel_err = mtable['uTDel'][idx]/3600

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

    models = tuple(models)

    return models



def setup_sampler(obs,
                    models,
                    pos = None,
                    threads = 4,
                    **kwargs):
    """
    Initialises and returns EnsembleSampler object
    """
    if pos == None:
        pos = setup_positions(obs=obs, **kwargs)

    nwalkers = len(pos)
    ndim = len(pos[0])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, burstclass.lhoodClass,
                                    args=(obs,models), threads=threads)
    return sampler



def setup_positions(obs,
                        nwalkers = 200,
                        params0 = [6.09, 60., 1.28],
                        tshift = -6.5,
                        mag = 1e-3):
    """
    Sets up and returns posititons of walkers
    """
    params = list(params0)   # prevent persistence between calls
    for i in range(len(obs)):
        params.append(tshift)

    ndim = len(params)
    pos = [params * (1 + mag * np.random.randn(ndim)) for i in range(nwalkers)]

    return pos



def run_sampler(sampler,
                    pos,
                    nsteps,
                    restart=False):
    """
    Runs emcee chain for nsteps
    """
    if restart:
        sampler.reset()

    pos_new, lnprob, rstate = sampler.run_mcmc(pos,nsteps)

    try:
        autocorr = sampler.get_autocorr_time()
        print('Autocorrelation: ', autocorr)
    except:
        print('Too few steps for autocorrelation estimate')

    return pos_new, lnprob, rstate



def load_chain(run,
                batches,
                step,
                con_ver,
                source='gs1826',
                **kwargs):
    """
    ========================================================
    Load chain file from completed emcee run
    ========================================================
    Parameters
    --------------------------------------------------------
    run
    batches
    step
    source
    ========================================================
    """
    batches = expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    chain_path = os.path.join(path, source, 'concord')

    full_str = full_string(run=run, batches=batches, step=step, source=source, con_ver=con_ver)
    chain_str = 'chain_{full}.npy'.format(full=full_str)
    chain_file = os.path.join(chain_path, chain_str)

    print('Loading chain: ', chain_file)
    chain = np.load(chain_file)

    return chain



def get_summary(run,
                batches,
                step,
                con_ver,
                ignore=250,
                source='gs1826',
                param_names=["d", "i", "1+z"],
                **kwargs):
    """
    ========================================================
    Get summary stats (mean + std) from a given mcmc chain
    ========================================================
    """
    batches = expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)

    chain = load_chain(run=run, batches=batches, step=step, source=source, con_ver=con_ver, path=path)
    chain = chain[:, ignore:, :]  # cut out "burn-in" steps

    # ===== Construct time parameter strings =====
    ndim = np.shape(chain)[2]
    n_time = ndim - len(param_names)
    t_params = construct_t_params(n_time)
    param_names = param_names + t_params

    # ===== Get summary values =====
    cs = ChainConsumer()
    cs.add_chain(chain.reshape(-1, ndim), parameters=param_names)
    summary = cs.analysis.get_summary()

    return summary



def save_summaries(n_runs,
                        batches,
                        step,
                        con_ver,
                        ignore=250,
                        source='gs1826',
                        param_names=['d', 'i', '1+z'],
                        exclude=[],
                        **kwargs):
    """
    ========================================================
    Extracts summary mcmc stats for a batch and saves as a table
    ========================================================
    Parameters
    --------------------------------------------------------
    n_runs  = int   : number of runs in each batch
    exclude = [int] : runs to skip over/exclude from analysis
    --------------------------------------------------------
    Notes:
            - Assumes each batch contains models numbered from 1 to [n_runs]
    ========================================================
    """
    batches = expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    obs = load_obs(source=source, **kwargs)

    n_obs = len(obs)
    t_params = construct_t_params(n_obs)
    param_names = param_names + t_params


    # ===== Setup dict to store summary values =====
    results = {}
    results['run'] = np.arange(1, n_runs+1)
    results['lhood'] = np.zeros(n_runs)       # likelihood values
    sigma_bounds_names = []

    for p in param_names:
        p_low = p + '_low'                     # 1-sigma lower/upper boundaries
        p_high = p + '_high'                   #
        sigma_bounds_names += [p_low, p_high]

        results[p] = np.zeros(n_runs)
        results[p_low] = np.zeros(n_runs)
        results[p_high] = np.zeros(n_runs)


    # ===== get summaries from each set =====
    unconstrained_flag = False

    for run in range(1, n_runs+1):
        if run in exclude:
            results['lhood'][run-1] = np.nan
            continue

        models = load_models(runs=[run], batches=batches, source=source, **kwargs)
        summary = get_summary(run=run, batches=batches, source=source, step=step,
                                con_ver=con_ver, ignore=ignore, param_names=param_names, **kwargs)

        # ===== get mean +/- 1-sigma for each param =====
        means = []
        for p in param_names:
            results[p][run-1] = summary[p][1]
            results[p + '_low'][run-1] = summary[p][0]
            results[p + '_high'][run-1] = summary[p][2]

            means.append(summary[p][0])

            # ===== Test for unconstrained parameter =====
            if summary[p][0] == None:       # an unconstrained param won't have any bounds
                unconstrained_flag = True

        # ===== get likelihood value =====
        if unconstrained_flag:
            lhood = np.nan
            unconstrained_flag = False
        else:
            lhood = burstclass.lhoodClass(params=means, obs=obs, model=models)

        results['lhood'][run-1] = lhood


    # ========== format and save table ==========
    # --- number formatting stuff ---
    flt0 = '{:.0f}'.format
    flt4 = '{:.4f}'.format
    FORMATTERS = {'lhood':flt0}
    for p in param_names[:3]:
        FORMATTERS[p] = flt4
        FORMATTERS[p + '_low'] = flt4
        FORMATTERS[p + '_high'] = flt4

    out_table = pd.DataFrame(results)
    col_order = ['run', 'lhood'] + param_names + sigma_bounds_names
    out_table = out_table[col_order]    # fix column order

    # batch_str = full_string(run=run, batches=batches, step=step, source=source)
    batch_str = '{src}_{b}_S{s}_C{c:02}'.format(src=source, b=daisychain(batches), s=step, c=con_ver)
    file_str = 'mcmc_' + batch_str + '.txt'
    file_path = os.path.join(path, source, 'mcmc', file_str)

    table_str = out_table.to_string(index=False, justify='left', col_space=8, formatters=FORMATTERS)

    with open(file_path, 'w') as f:
        f.write(table_str)

    return out_table



def write_batch(nruns,
                batches,
                con_ver,
                n0=1,
                source='gs1826',
                qos = 'short',
                auto_qos = True,
                prepend='con',
                time=10,
                threads=4,
                **kwargs):
    """
    ========================================================
    Writes batch script for job-submission on Monarch
    ========================================================
    Parameters
    --------------------------------------------------------
    auto_qos  = bool   : split jobs between node types (overrides qos)
    n0        = int    : run to start with (assumes all runs between n0 and nruns)
    prepend   = str    : label to prepend filename with
    time      = int    : time in hours
    threads   = int    : number of cores per run
    ========================================================"""
    batches = expand_batches(batches, source)
    path = kwargs.get('path', os.path.join(GRIDS_PATH))
    log_path = os.path.join(path, source, 'logs')

    print('Writing slurm sbatch script')
    triplet_str = triplet_string(batches=batches, source=source)
    run_str = '{n0}-{n1}'.format(n0=n0, n1=nruns)
    filename = '{prep}_{triplet}_{runs}.sh'.format(prep=prepend, triplet=triplet_str, runs=run_str)
    filepath = os.path.join(log_path, filename)

    job_str = '{prep}_{src}{b1}'.format(prep=prepend, src=source[:2], b1=batches[0])
    time_str = '{hr:02}:00:00'.format(hr=time)
    batch_list = ''

    for b in batches:
        batch_list += '{b} '.format(b=b)

    with open(filepath, 'w') as f:
        f.write("""#!/bin/bash

#SBATCH --job-name={job_str}
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array={run_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={threads}
#SBATCH --qos={qos}_qos
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu

######################
# Begin work section #
######################

module load python/3.5.1-gcc

N=$SLURM_ARRAY_TASK_ID
cd /home/zacpetej/id43/python/concord/
python3 run_concord.py {source} {batch_list} $N {con_ver} no_restart""".format(job_str=job_str,
            run_str=run_str, source=source, batch_list=batch_list, threads=threads,
            qos=qos, time_str=time_str, con_ver=con_ver))



def plot_lightcurves(run,
                        batches,
                        step,
                        con_ver,
                        source='gs1826',
                        **kwargs):
    """
    ========================================================
    Plots lightcurves with best-fit params from an mcmc chain
    ========================================================
    Parameters
    --------------------------------------------------------
    run
    batches    = [int] :
    step       = int   : emcee step to load (used in file label)
    source
    path       = str   : path to kepler_grids directory
    ========================================================
    """
    batches = expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    source_path = os.path.join(path, source)

    obs = load_obs(source=source, **kwargs)
    models = load_models(runs=[run], batches=batches, source=source, **kwargs)

    n = len(obs)
    param_names = ['d', 'i', '1+z']
    t_params = construct_t_params(n)
    param_names = param_names + t_params

    # special case string without run
    # batch_str = '{src}_{b}_S{s}_C{c:02}'.format(src=source, b=daisychain(batches), s=step, c=con_ver)
    batch_str = full_string(run=0, batches=batches, source=source, step=step, con_ver=con_ver)
    table_name = 'mcmc_' + batch_str + '.txt'
    table_filepath = os.path.join(source_path, 'mcmc', table_name)

    table = pd.read_table(table_filepath, delim_whitespace=True)
    run_idx = np.argwhere(table['run'] == run)[0][0]

    params = {}

    for p in param_names:
        params[p] = table[p][run_idx]

    print(params)

    for i in range(n):
        t = 't' + str(i+1)
        base_input_params = [params['d']*u.kpc, params['i']*u.degree, params['1+z']]
        input_params = base_input_params + [params[t]*u.s] # append relevant time only

        obs[i].compare(models[i], input_params, breakdown=True, plot=True)

    plt.show(block=False)



def save_contours(runs,
                batches,
                step,
                con_ver,
                ignore=250,
                source='gs1826',
                **kwargs):
    """
    ========================================================
    Save contour plots from multiple concord runs
    ========================================================
    Parameters
    --------------------------------------------------------
    run
    batches    = [int] :
    step       = int   : emcee step to load (used in file label)
    ignore     = int   : number of initial chain steps to ignore (burn-in phase)
    source
    path       = str   : path to kepler_grids directory
    ========================================================
    """
    batches = expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    ndim = 6
    parameters=[r"$d$",r"$i$",r"$1+z$"]
    c = ChainConsumer()

    triplet_str = triplet_string(batches=batches, source=source)

    chain_dir = os.path.join(path, source, 'concord')
    plot_dir = os.path.join(path, source, 'plots')
    save_dir = os.path.join(plot_dir, triplet_str)

    try_mkdir(save_dir, skip=True)

    print('Source: ', source)
    print('Loading from : ', chain_dir)
    print('Saving to    : ', save_dir)
    print('Batches: ', batches)
    print('Runs: ', runs)

    for run in runs:
        print('Run ', run)
        full_str = full_string(run=run, batches=batches, source=source, step=step, con_ver=con_ver)

        chain_str = 'chain_{full_str}.npy'.format(full_str=full_str)
        save_str = 'contour_{full_str}.png'.format(full_str=full_str)


        chain_file = os.path.join(chain_dir, chain_str)
        save_file = os.path.join(save_dir, save_str)

        chain = np.load(chain_file)[:, ignore:, :]
        chain = chain.reshape((-1, ndim))

        c.add_chain(chain, parameters=parameters)

        fig = c.plotter.plot()
        fig.set_size_inches(7,7)
        fig.savefig(save_file)

        plt.close(fig)
        c.remove_chain()

    print('Done!')



def animate_contours(run,
                        step,
                        dt=5,
                        fps=30,
                        ffmpeg=True,
                        **kwargs):
    """
    Saves frames of contour evolution, to make an animation
    """
    path = kwargs.get('path', CONCORD_PATH)

    parameters=[r"$d$",r"$i$",r"$1+z$"]
    chain_str = 'chain_{r}'.format(r=run)
    chain_file = os.path.join(path, 'temp', '{chain}_{st}.npy'.format(chain=chain_str, st=step))
    chain = np.load(chain_file)
    nwalkers, nsteps, ndim = np.shape(chain)

    mtarget = os.path.join(path, 'animation')
    ftarget = os.path.join(mtarget, 'frames')

    c = ChainConsumer()

    for i in range(dt, nsteps, dt):
        print('frame  ', i)
        subchain = chain[:, :i, :].reshape((-1,ndim))
        c.add_chain(subchain, parameters=parameters)

        fig = c.plotter.plot()
        fig.set_size_inches(6,6)
        cnt = round(i/dt)

        filename = '{chain}_{n:04d}.png'.format(chain=chain_str, n=cnt)
        filepath = os.path.join(ftarget, filename)
        fig.savefig(filepath)

        plt.close(fig)
        c.remove_chain()

    if ffmpeg:
        print('Creating movie')
        framefile = os.path.join(ftarget, '{chain}_%04d.png'.format(chain=chain_str))
        savefile = os.path.join(mtarget, '{chain}.mp4'.format(chain=chain_str))
        subprocess.run(['ffmpeg', '-r', str(fps), '-i', framefile, savefile])



def construct_t_params(n):
    """
    ========================================================
    Creates list of time-param labels (t1, t2, t3...)
    ========================================================
    Parameters
    --------------------------------------------------------
    n = int  : number of time parameters
    ========================================================
    """
    t_params = []

    for i in range(n):
        tname = 't' + str(i+1)
        t_params.append(tname)

    return t_params



def full_string(run,
                batches,
                step,
                con_ver,
                source='gs1826'):
    """
    ========================================================
    constructs a standardised string for a batch model
    ========================================================
    """
    batches = expand_batches(batches, source)
    b_string = daisychain(batches)

    if run == 0:
        run_str = ''
    else:
        run_str = '_R{}'.format(run)

    if con_ver == 0:
        con_str = ''
    else:
        con_str = '_C{:02}'.format(con_ver)

    full_str = '{src}_{bstr}{run}_S{stp}{cv}'.format(src=source, bstr=b_string,
                                                    run=run_str, stp=step, cv=con_str)

    return full_str



def triplet_string(batches,
                   source='gs1826'):
    batches = expand_batches(batches, source)
    batch_daisy = daisychain(batches)
    triplet_str = '{source}_{batches}'.format(source=source, batches=batch_daisy)

    return triplet_str



def expand_batches(batches, source):
    """Checks format of 'batches' parameter and returns relevant array
    if batches is arraylike: keep
    if batches is integer N: assume first batch of batch set"""
    N = {'gs1826': 3, '4u1820': 2}  # number of epochs

    if type(batches) == int:   # assume batches gives first batch
        batches_out = np.arange(batches, batches+N[source])

    elif type(batches) == list   or   type(batches) == np.ndarray:
        batches_out = batches

    else:
        print('Invalid type(batches). Must be array-like or int')
        sys.exit()

    return batches_out


def expand_runs(runs):
    """Checks format of 'runs' parameter and returns relevant array
    if runs is arraylike: keep
    if runs is integer N: assume there are N runs from 1 to N"""
    if type(runs) == int:   # assume runs = n_runs
        runs_out = np.arange(1, runs+1)
    elif type(runs) == list   or   type(runs) == np.ndarray:
        runs_out = runs
    else:
        print('Invalid type(runs)')
        sys.exit()

    return runs_out


def daisychain(daisies,
                delim='-'):
    """
    ========================================================
    returns a string of daisies seperated by a delimiter (e.g. 1-2-3-4)
    ========================================================
    daisies  = [int]
    delim    = str     : delimiter to place between IDs
    ========================================================
    """
    daisy = ''

    for i in daisies[:-1]:
        daisy += str(i)
        daisy += delim

    daisy += str(daisies[-1])

    return daisy


def try_mkdir(path, skip=False):
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
