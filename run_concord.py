import ctools
import numpy as np
import pickle
import sys
import os

# =============================================================================
# Usage:
#       python run_concord.py [source] [batch1] [batch2] [batch3] [run] [concord_version] [restart] (step)
# =============================================================================

GRIDS_PATH = os.environ['KEPLER_GRIDS']

if len(sys.argv) != 9:
    print("""Must provide 8 parameters:
                1. source
                2. batch1
                3. batch2
                4. batch3
                5. run
                6. con_ver
                7. threads
                8. restart""")
    sys.exit()

#batches = [1]   #!!!
#run = int(sys.argv[2])  #!!!
source = sys.argv[1]
batches = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
run = int(sys.argv[5])
con_ver = int(sys.argv[6])
threads = int(sys.argv[7])
restart = sys.argv[8]

print('Loading observed and model data: Run {r} from batches [{b1}, {b2}, {b3}]'.format(r=run, b1=batches[0], b2=batches[1], b3=batches[2]))
#print('Loading observed and model data: Run {r} from batches [{b1}]'.format(r=run, b1=batches[0]))  #!!!

obs = ctools.load_obs(source)
models = ctools.load_models(runs=[run,run,run], batches=batches)

tdelwts = {1:1., 2:2.5e3, 3:100.}   # tdel weights for fitting (for different con_ver)
tdelwt = tdelwts[con_ver]
weights = {'tdelwt':tdelwt, 'fluxwt':1.}

pos = ctools.setup_positions(obs)
sampler = ctools.setup_sampler(obs=obs, models=models, nwalkers=200, threads=threads, weights=weights)

batch_str = '{src}_{b1}-{b2}-{b3}_R{r}'.format(src=source, b1=batches[0], b2=batches[1], b3=batches[2], r=run)
chain_path = os.path.join(GRIDS_PATH, source, 'concord')

# TODO: restarting needs testing/debugging
if restart == 'restart':
    load_step = sys.argv[7]
    chain_str0 = 'chain_{batch}'.format(batch=batch_str)
    pname = '{chain}_S{step}.npy'.format(chain=chain_str, step=load_step)
    pfile = os.path.join(chain_path, pname)
    pos = np.load(pfile)[:,-1,:]
    restart = True
    start = int(sys.argv[8])
else:
    restart = False
    start=0

total_steps = 2000
net_steps = total_steps - start
nsteps = 2000      # No. of steps to do between savedumps
iters = round(net_steps/nsteps)
n0 = round(start/nsteps)

for i in range(n0, n0+iters):
    step = i*nsteps
    print('Doing steps: {0} - {1}'.format(step, (i+1)*nsteps))

    pos, lnprob, rstate = ctools.run_sampler(sampler, pos=pos, nsteps=nsteps, restart=restart)
    restart = False

    # === save chain state ===
    step_str = '{batch}_S{step}_C{cv:02}'.format(batch=batch_str, step=step+nsteps, cv=con_ver)

    chain_filepath = os.path.join(chain_path, 'chain_' + step_str)
    # lnprob_filepath = os.path.join(chain_path, 'lnprob_' + step_str)
    # rstate_filepath = os.path.join(chain_path, 'rstate_' + step_str + '.pkl')

    np.save(chain_filepath, sampler.chain)
    # np.save(lnprob_filepath, np.array(lnprob))

    # with open(rstate_filepath, 'wb') as output:
    #     pickle.dump(rstate, output, -1)

print('Done!')
