import ctools
import numpy as np
import pickle
import sys
import os

# =============================================================================
# Usage:
#       python run_concord.py [source] [batch1] [batch2] [batch3] [run] [restart] (step)
# =============================================================================

GRIDS_PATH = os.environ['KEPLER_GRIDS']

batches = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
#batches = [1]   #!!!
run = int(sys.argv[5])
#run = int(sys.argv[2])  #!!!

print('Loading observed and model data: Run {r} from batches [{b1}, {b2}, {b3}]'.format(r=run, b1=batches[0], b2=batches[1], b3=batches[2]))
#print('Loading observed and model data: Run {r} from batches [{b1}]'.format(r=run, b1=batches[0]))  #!!!

source = sys.argv[1]
obs = ctools.load_obs(source)
#obs = ctools.load_obs(source)[1] # !!!


models = ctools.load_models(runs=[run,run,run], batches=batches)
#models = ctools.load_models(runs=[run], source=source, batches=batches)[0] #!!!

pos = ctools.setup_positions(obs)
sampler = ctools.setup_sampler(obs=obs, models=models, nwalkers=200, threads=4)

batch_str = '{src}_{b1}-{b2}-{b3}_R{r}'.format(src=source, b1=batches[0], b2=batches[1], b3=batches[2], r=run)
#batch_str = '{src}_B{b1}_R{r}'.format(src=source, b1=batches[0], r=run)  #!!!

chain_path = os.path.join(GRIDS_PATH, source, 'concord')

# TODO: restarting needs testing/debugging (also tracking step labels correctly)
if sys.argv[3] == 'restart':
    load_step = sys.argv[7]
    chain_str0 = 'chain_{batch}'.format(batch=batch_str)
    pname = '{chain}_S{step}.npy'.format(chain=chain_str, step=load_step)
    pfile = os.path.join(chain_path, pname)
    pos = np.load(pfile)[:,-1,:]
    restart = True
    start = int(sys.argv[3])
else:
    restart = False
    start=0

total_steps = 1500
net_steps = total_steps - start
nsteps = 100      # No. of steps to do between savedumps
iters = round(net_steps/nsteps)
n0 = round(start/nsteps)

for i in range(n0, n0+iters):
    step = i*nsteps
    print('Doing steps: {0} - {1}'.format(step, (i+1)*nsteps))

    pos, lnprob, rstate = ctools.run_sampler(sampler, pos=pos, nsteps=nsteps, restart=restart)
    restart = False

    # === save chain state ===
    step_str = '{batch}_S{step}'.format(batch=batch_str, step=step+nsteps)

    chain_filepath = os.path.join(chain_path, 'chain_' + step_str)
    lnprob_filepath = os.path.join(chain_path, 'lnprob_' + step_str)
    rstate_filepath = os.path.join(chain_path, 'rstate_' + step_str + '.pkl')

    np.save(chain_filepath, sampler.chain)
    np.save(lnprob_filepath, np.array(lnprob))

    with open(rstate_filepath, 'wb') as output:
        pickle.dump(rstate, output, -1)

print('Done!')
