import numpy as np
import os, sys

import manipulation

GRIDS_PATH = os.environ['KEPLER_GRIDS']
CONCORD_PATH = os.environ['CONCORD_PATH']

def write_new_submissions(last_batch, con_ver, **kwargs):
    """========================================================
    Writes entire set of submission scripts for newly-defined con_ver
    ========================================================"""
    batches = np.arange(12, last_batch+1, 3)
    batches = np.concatenate([[4,7,9], batches])

    for batch in batches:
        write_submission(batches=batch, con_ver=con_ver, **kwargs)



def write_submission(batches, con_ver, n0=1, source='gs1826',
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
    nruns = manipulation.get_nruns(batches[0])
    path = kwargs.get('path', GRIDS_PATH)
    log_path = os.path.join(path, source, 'logs')

    triplet_str = manipulation.triplet_string(batches=batches, source=source)
    run_str = '{n0}-{n1}'.format(n0=n0, n1=nruns)
    job_str = 'c{cv}_{src}{b1}'.format(cv=con_ver, src=source[:2], b1=batches[0])
    time_str = '{hr:02}:00:00'.format(hr=time)
    extensions = {'monarch':'.sh', 'icer':'.qsub'}
    batch_list = ''

    for b in batches:
        batch_list += '{b} '.format(b=b)


    for cluster in ['monarch', 'icer']:
        print('Writing submission script for cluster:', cluster)
        extension = extensions[cluster]
        filename = '{cluster}_con{cv}_{triplet}_{runs}{ext}'.format(cluster=cluster,
                        cv=con_ver, triplet=triplet_str, runs=run_str, ext=extension)
        filepath = os.path.join(log_path, filename)

        script_str = get_submission_str(job_str=job_str, run_str=run_str,
                source=source,  batch_list=batch_list, threads=threads,
                qos=qos, time_str=time_str, con_ver=con_ver, cluster=cluster)

        with open(filepath, 'w') as f:
            f.write(script_str)


def get_submission_str(job_str, run_str, source, batch_list,
                    threads, qos, time_str, con_ver, cluster):
    """========================================================
    Returns string of submission script for given cluster
    ========================================================
    cluster  =  str  : cluster type, one of [monarch, icer]
    ========================================================"""
    if cluster == 'monarch':
        return """#!/bin/bash
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
python3 run_concord.py {source} {batch_list} $N {con_ver} {threads} no_restart""".format(job_str=job_str,
        run_str=run_str, source=source, batch_list=batch_list, threads=threads,
        qos=qos, time_str=time_str, con_ver=con_ver)

    elif cluster == 'icer':
        return """#!/bin/bash --login
#PBS -N {job_str}
#PBS -t {run_str}
#PBS -l nodes=1:ppn={threads}
#PBS -l walltime={time_str}
#PBS -l mem=1000mb
#PBS -l file=1gb
#PBS -j oe
#PBS -m abe
#PBS -M zac.johnston@monash.edu

N=$PBS_ARRAYID
source /mnt/home/f0003004/mypy/bin/activate
cd /mnt/home/f0003004/codes/concord
python3 run_concord.py {source} {batch_list} $N {con_ver} {threads} no_restart
qstat -f $PBS_JOBID     # Print out final statistics about resource uses before job exits""".format(job_str=job_str,
            run_str=run_str, source=source, batch_list=batch_list, threads=threads,
            time_str=time_str, con_ver=con_ver)

    else:
        print('ERROR: Not a valid cluster type. Must be one of [monarch, icer]')
        sys.exit()
