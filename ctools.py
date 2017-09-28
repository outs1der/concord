# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import emcee
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
#    - generalised function to plot best fit contours/lightcurves
#    - write simple pipeline/recipe for setting up grids --> analyser --> concord
#
# --- Functions ---
# def setup_all():
#     """
#     Sets up observed/modelled burst objects and initiallises sampler
#     """
#
# def plot_walkers():
#     """
#     plots walkers of mcmc chain
#     """
#============================================


#===================================================
# GLOBAL PATHS
#---------------------------------------------------
# If you wish to use a different path for a specific function call,
# include it as the parameter 'path' when calling the function
#---------------------------------------------------
GRIDS_PATH = '/home/zacpetej/projects/kepler_grids/'
CONCORD_PATH = '/home/zacpetej/projects/codes/concord/'
#===================================================

def load_obs(source='gs1826',
                **kwargs):
    """
    Loads observed burst data
    """
    #========================================================
    # Parameters
    #--------------------------------------------------------
    # source   = str : astrophysical source being matched (gs1826, 4u1820)
    # obs_path = str : path to directory containing observational data
    #========================================================
    obs = []
    obs_path = kwargs.get('path', GRIDS_PATH+'obs_data/')
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
    Loads a set of models (parameters and lightcurves)
    """
    #========================================================
    # Parameters
    #--------------------------------------------------------
    # runs    = [] : list of models to use
    # batches = [] : batches that the models in runs[] belong to (one-to-one correspondence)
    #========================================================
    models = []
    path = kwargs.get('path', GRIDS_PATH)

    for i, run in enumerate(runs):
        batch = batches[i]

        batch_str = '{source}_{batch}'.format(source=source, batch=batch)
        mean_str = '{batch_str}_{base}{run}_mean.data'.format(batch_str=batch_str, base=basename, run=run)
        param_str = '{prefix}_{batch_str}.txt'.format(prefix=params_prefix, batch_str=batch_str)
        summ_str = '{prefix}_{batch_str}.txt'.format(prefix=summ_prefix, batch_str=batch_str)

        source_path = os.path.join(path, source)
        param_file = os.path.join(source_path, param_str)
        summ_file = os.path.join(source_path, summ_str)
        mean_path = os.path.join(source_path, 'mean_lightcurves')

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

    return pos_new, lnprob, rstate



def write_batch(run,
                batches,
                qos = 'medium',
                auto_qos = True,
                **kwargs):
    """
    ========================================================
    Writes batch script for job-submission on Monarch
    ========================================================
    Parameters
    --------------------------------------------------------
     auto_qos  = bool   : split jobs between node types (overrides qos)
    ========================================================
    """

#TODO: ---Needs finishing--
    path = kwargs.get('path', os.path.join(GRIDS_PATH, 'logs'))
    print('Writing slurm sbatch script')
    span = '{n0}-{n1}'.format(n0=n0, n1=n1)
    slurmfile = path+'{prep}{gbase}{grid}_{span}.sh'.format(prep=prepend, gbase=grid_basename, grid=grid_num, span=span)

    with open(slurmfile, 'w') as f:
        f.write("""#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array={n0}-{n1}
#SBATCH --time={time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={threads}
{qos_str}
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu

######################
# Begin work section #
######################

N=$SLURM_ARRAY_TASK_ID
module load python/3.5.1-gcc
cd /home/zacpetej/id43/python/concord/
python3 run_concord {source} {batches} $N no_restart""".format(jobname=jobname,
                n0=n0, n1=n1, source=source, batches=batches, threads=threds, qos_str=qos_str, time=time))



def batch_string(batches,
                    delim='-'):
    """
    ========================================================
    constructs a string of generic number of batch IDs
    ========================================================
    batches  = [int]
    delim    = str     : delimiter to place between IDs
    ========================================================
    """
    b_string = ''

    for b in batches[:-1]:
        b_string += str(b)
        b_string += delim

    b_string += str(batches[-1])

    return b_string



def load_chain(run,
                batches,
                step,
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
    path = kwargs.get('path', GRIDS_PATH)
    b_str = batch_string(batches)

    chain_path = os.path.join(path, source, 'concord')
    chain_str = 'chain_{src}_{bstr}_R{run}_S{stp}.npy'.format(src=source, bstr = b_str,
                                        run=run, stp=step)
    chain_file = os.path.join(chain_path, chain_str)
    chain = np.load(chain_file)

    return chain



def save_contours(runs,
                batches,
                step,
                ignore=100,
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
    path = kwargs.get('path', GRIDS_PATH)
    ndim = 6
    parameters=[r"$d$",r"$i$",r"$1+z$"]
    c = ChainConsumer()

    chain_dir = os.path.join(path, source, 'concord')
    save_dir = os.path.join(path, source, 'contours')

    print('Source: ', source)
    print('Loading from : ', chain_dir)
    print('Saving to    : ', save_dir)
    print('Batches: ', batches)
    print('Runs: ', runs)

    for run in runs:
        print('Run ', run)
        batch_str = '{src}_{b1}-{b2}-{b3}_R{r}_S{s}'.format(src=source, b1=batches[0], b2=batches[1], b3=batches[2], r=run, s=step)
        chain_str = 'chain_{batch_str}.npy'.format(batch_str=batch_str)
        save_str = 'contour_{batch_str}.png'.format(batch_str=batch_str)

        chain_file = os.path.join(chain_dir, chain_str)
        save_file = os.path.join(save_dir, save_str)

        chain = np.load(chain_file)[:, ignore:, :]
        chain = chain.reshape((-1, ndim))

        c.add_chain(chain, parameters=parameters)

        fig = c.plotter.plot()
        fig.set_size_inches(6,6)
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
