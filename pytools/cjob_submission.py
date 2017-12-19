import numpy as np
import os, sys

import manipulation

GRIDS_PATH = os.environ['KEPLER_GRIDS']
CONCORD_PATH = os.environ['CONCORD_PATH']

def write_all_submission_scripts(last_batch, con_ver, source, **kwargs):
    """========================================================
    Writes entire set of submission scripts for newly-defined con_ver
    ========================================================"""
    if source == 'gs1826':
        batches = np.arange(12, last_batch+1, 3)
        batches = np.concatenate([[4,7,9], batches])
    elif source == '4u1820':
        batches = np.arange(2, last_batch+1, 2)

    for batch in batches:
        write_submission_script(batches=batch, con_ver=con_ver,
                                source=source, **kwargs)



def write_submission_script(batches, source, con_ver, n0=1,
                qos = 'short', time=4, threads=8, **kwargs):
    """========================================================
    Writes batch script for job-submission on Monarch and ICER
    ========================================================
    Parameters
    --------------------------------------------------------
    n0        = int    : run to start with (assumes all runs between n0 and nruns)
    time      = int    : time in hours
    threads   = int    : number of cores per run
    ========================================================"""
    batches = manipulation.expand_batches(batches, source)
    nruns = manipulation.get_nruns(batches[0], source=source)
    path = kwargs.get('path', GRIDS_PATH)
    log_path = os.path.join(path, source, 'logs')

    triplet_str = manipulation.triplet_string(batches=batches, source=source)
    run_str = f'{n0}-{nruns}'
    job_str = f'c{con_ver}_{source[:2]}{batches[0]}'
    time_str = f'{time:02}:00:00'
    extensions = {'monarch':'.sh', 'icer':'.qsub'}

    for cluster in ['monarch', 'icer']:
        print(f'Writing submission script: C{con_ver},  {cluster}')
        extension = extensions[cluster]
        filename = f'{cluster}_con{con_ver}_{triplet_str}_{run_str}{extension}'
        filepath = os.path.join(log_path, filename)

        script_str = get_submission_str(job_str=job_str, run_str=run_str,
                source=source,  batch0=batches[0], threads=threads,
                qos=qos, time_str=time_str, con_ver=con_ver, cluster=cluster)

        with open(filepath, 'w') as f:
            f.write(script_str)


def get_submission_str(job_str, run_str, source, batch0,
                    threads, qos, time_str, con_ver, cluster):
    """========================================================
    Returns string of submission script for given cluster
    ========================================================
    cluster  =  str  : cluster type, one of [monarch, icer]
    ========================================================"""
    if cluster == 'monarch':
        return f"""#!/bin/bash
#SBATCH --job-name={job_str}
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array={run_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={threads}
#SBATCH --partition=batch,short,medium
#SBATCH --qos={qos}_qos
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
######################

# module load python/3.5.1-gcc
source $HOME/python/mypy-3.6.3/bin/activate

N=$SLURM_ARRAY_TASK_ID
cd /home/zacpetej/id43/python/concord/pytools
python3 run_concord.py {source} {batch0} $N {con_ver} {threads} no_restart"""

    elif cluster == 'icer':
        return f"""#!/bin/bash --login
#PBS -N {job_str}
#PBS -t {run_str}
#PBS -l nodes=1:ppn={threads}
#PBS -l walltime={time_str}
#PBS -l mem=2000mb
#PBS -l file=2000mb
#PBS -j oe
#PBS -m abe
#PBS -M zac.johnston@monash.edu

N=$PBS_ARRAYID
source /mnt/home/f0003004/mypy3.6/bin/activate
cd /mnt/home/f0003004/codes/concord/pytools
python3 run_concord.py {source} {batch0} $N {con_ver} {threads} no_restart
qstat -f $PBS_JOBID     # Print out final statistics about resource uses before job exits"""

    else:
        print('ERROR: Not a valid cluster type. Must be one of [monarch, icer]')
        sys.exit()
