# Script to perform burst model comparisons, and run multinest
# Duncan.Galloway@monash.edu, 2018 Mar
#
# Initialise the environment

from burstclass import *
import pymultinest

# Read in an observed burst, from the reference sample of Galloway et al.
# (2017)

b = ObservedBurst('gs1826-24_4.177h.dat',path='../../burst/reference/')

# Now also define a model burst. Lampe et al. models are deprecated;

## c = ModelBurst('a005',path='../../reference')
# c = KeplerBurst(run_id='a028',path='../../burst/reference')

# Low-metallicity run supplied by Z. Johnston, 2017 April
# accretion rate, fuel composition and recurrence time were all provided by Zac;
# xi, g and R_NS are guesses

c=KeplerBurst(filename='mean_run930.dat',lAcc=0.108,Z=0.005,H=0.7,
              tdel=4.5/1.259,tdel_err=0.2/1.259,
              g = 1.86552e+14*u.cm/u.s**2, R_NS=11.2*u.km)

# ------- --------- --------- --------- --------- --------- --------- ---------

def mnPrior(cube, ndim, nparams):
    '''
    Prior function for use with pymultinest. Should transform the unit 
    cube into the parameter cube.
    Parameter order (equivalent to dimensions of cube) is: 
    distance (kpc), inclination (degrees), redshift, and offset time 
    (seconds)'''

# Distance is in the range 1-16 kpc
    cube[0] = 1. + cube[0]*15.
    
# inclination in the range 0-90; could be more restrictive for a non-dipper
    cube[1] = np.arcsin(cube[1])*180./pi

# redshift is in the range 1-2
    cube[2] = 1. + cube[2]
    
# one or more time offsets are in the range -20,+20
    for i in range(ndim-3):
        cube[i+3] = (cube[i+3]-0.5)*40.

# ------- --------- --------- --------- --------- --------- --------- ---------

def Loglike(cube, ndim, nparams):
    '''
    Function to return the likelihood, takes as input the parameter cube
    and should return the logarithm of the likelihood.
    Don't know how to pass additional parameters to this function
    (ans: seems we don't need to, we can just access them as global 
    variables)
    Although the cube parameter name implies multiple parameter points to
    evaluate, practically it seems to only have one
    Compares obs to models, with weights weight, and disk model disk
    '''
    
    param = np.array([cube[0],cube[1],cube[2],cube[3]])
#    print (param)
    loglikelihood = lhoodClass(param, 
                        ll_obs, ll_model, ll_weights, ll_disk)

    return loglikelihood

# ------- --------- --------- --------- --------- --------- --------- ---------

# Here are the parameters we're going to fit. Accessing these as global 
# variables in the function above

ll_obs = b
ll_model = c
ll_weights = {'fluxwt':1.0, 'tdelwt':2.5e3}
ll_disk = 'he16_a'

# Here we initialise the parameter starting point and get going
# I don't think this is actually necessary for MultiNest

params = [6.9,60.,1.25,-7.]
# args=(b,c,weights,'he16_a')

# Total no. of parameters, should be equal to ndims in most cases but if 
# you need to store some additional parameters with the actual parameters 
# then you need to pass them through the likelihood routine.

n_params = len(params)

# Not clear how to pass the walker positions, nor the inputs to lhoodClass
# to multinest

print ("Now running multinest...")
pymultinest.run(Loglike, mnPrior, n_params, outputfiles_basename='out/',
    resume = False, verbose = True)
print ("...done")
