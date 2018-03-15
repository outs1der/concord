import ctools
import numpy as np
import pickle
import sys
import os

# concord
import con_versions
import define_sources
import manipulation
# =============================================================================
# Usage:
#       python run_concord.py [source] [batch0] [run] [concord_version] [restart] (step)
# =============================================================================

GRIDS_PATH = os.environ['KEPLER_GRIDS']

if len(sys.argv) != 7:
    print("""Must provide 6 parameters:
                1. source
                2. batch1
                3. run
                4. con_ver
                5. threads
                6. restart""")
    sys.exit()

source = sys.argv[1]
# batches = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
batch0 = int(sys.argv[2])
run = int(sys.argv[3])
con_ver = int(sys.argv[4])
threads = int(sys.argv[5])
restart = sys.argv[6]

batches = manipulation.expand_batches(batches=batch0, source=source)
runs = np.full_like(batches, run)

print(f"""Loading observed and model data:
        Run {run} from batches {batches}""")

obs = ctools.load_obs(source=source)
models = ctools.load_models(runs=runs, batches=batches, source=source)
nwalkers = 200

# === Con_ver parameters ===
weights = con_versions.get_weights(con_ver)
disc_model = con_versions.get_disc_model(con_ver)

pos = ctools.setup_positions(source=source, nwalkers=nwalkers)
sampler = ctools.setup_sampler(obs=obs, source=source, models=models,
                    nwalkers=nwalkers, threads=threads, weights=weights,
                    disc_model=disc_model)

# batch_str = f'{source}_{batches[0]}-{batches[1]}-{batches[2]}_R{run}'
chain_path = os.path.join(GRIDS_PATH, 'sources', source, 'concord')

# TODO: restarting needs testing/debugging
# if restart == 'restart':
#     load_step = sys.argv[7]
#     chain_str0 = f'chain_{batch_str}'
#     pname = f'{chain_str}_S{load_step}.npy'
#     pfile = os.path.join(chain_path, pname)
#     pos = np.load(pfile)[:,-1,:]
#     restart = True
#     start = int(sys.argv[8])
# else:
restart = False
start=0

total_steps = 2000
net_steps = total_steps - start
nsteps = 2000      # No. of steps to do between savedumps
iters = round(net_steps/nsteps)
n0 = round(start/nsteps)

for i in range(n0, n0+iters):
    step0 = i * nsteps
    step1 = (i+1) * nsteps
    print(f'Doing steps: {step0} - {step1}')

    pos, lnprob, rstate = ctools.run_sampler(sampler, pos=pos, nsteps=nsteps,
                                                restart=restart)
    restart = False

    # === save chain state ===
    # step_str = f'{batch_str}_S{step1}_C{con_ver:02}'
    step_str = manipulation.full_string(batches=batches, run=run, source=source,
                                        step=step1, con_ver=con_ver)

    chain_filepath = os.path.join(chain_path, f'chain_{step_str}')
    np.save(chain_filepath, sampler.chain)

print('Done!')
