# Various utilities moved from burstclass.py
# Augmented 2019 Aug with routines from Inferring\ composition.ipynb
#
# def value_to_dist(_num, nsamp=NSAMP_DEF, unit=None):
# def homogenize_params(theta, nsamp=None):
# def len_dist(d):
# def asym_norm(m, sigm=None, sigp=None, nsamp=NSAMP_DEF, positive=False, model=1):
# def intvl_to_errors(perc):
# def g(M,R,Newt=False,units='cm/s^2'):
# def redshift(M,R):
# def check_M_R_opz(M, R, opz):
# def calc_mr(g,opz):
# def solve_radius(M,R_Newt,eta=1e-6):
# def decode_LaTeX(string):
# def Q_nuc(Xbar, quadratic=False, old_relation=False, coeff=False)
# def X_0(xbar, zcno, tdel, opz=1.259, debug=False, old_relation=False):
# def alpha(_tdel, _fluen, _fper, _c_bol=1.0, nsamp=NSAMP_DEF):
# def hfrac(alpha, tdel, opz=1.259, zcno=0.02, old_relation=False, ...)
# def iso_dist(nsamp=1000, imin=0., imax=IMAX_NDIP)
# def dist(F_pk, F_pk_err, nsamp=10000, X=0.0, empirical=False, ...)
# def luminosity(F_X, F_X_err=0.0, dist=8*u.kpc, dist_err=None, nsamp=NSAMP_DEF, ...)
# def mdot(F_per, F_per_err, dist, nsamp=10000, M=M_NS, R=R_NS, ...)
# def yign(E_b, E_b_err, dist, R_NS=10., Xbar=0.7, opz=1.259, ...)
# def L_Edd(F_pk, F_pk_err, dist, dist_err=0.0, nsamp=10000, dip=False, ...)

import astropy.units as u
import astropy.uncertainty as unc
from astropy.visualization import quantity_support
from scipy.stats import poisson

# Some defaults

CONF=68.        # default 68% confidence intervals
NSAMP_DEF=1000  # default 1000 samples
IMAX_NDIP=75.   # [degrees] maximum possible inclination angle for non-dipper
M_NS=1.4 * u.M_sun  # default neutron-star mass
R_NS=11.2 * u.km    # default neutron-star radius
OPZ=1.259       # default redshift

ETA=1.e-6       # generic fraction to detect deviation from default values

# This parameter sets the prefactor for the time to burn all H via hot-CNO; see
# Lampe et al. (2016, ApJ 819, 46)

TPREF_CNO = 9.8 * u.hr
TPREF_CNO_OLD = 11.0 * u.hr

# Parameters for working with MINBAR

MINBAR_PERFLUX_UNIT = 1e-6*u.erg/u.cm**2/u.s
MINBAR_FLUX_UNIT = 1e-9*u.erg/u.cm**2/u.s
MINBAR_FLUEN_UNIT = 1e-6*u.erg/u.cm**2

import numpy as np
from math import *
from .diskmodel import *
import astropy.constants as const
from scipy.special import erfinv
import re

import logging

def create_logger():
    """
    Create a logger instance where messages are sent.
    See https://docs.python.org/3/library/logging.html

    :return: loggerinstance
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers: # Check if created before, otherwise a reload will add handlers
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s:%(funcName)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = create_logger()

# ------- --------- --------- --------- --------- --------- --------- ---------

def value_to_dist(_num, nsamp=NSAMP_DEF, unit=None, verbose=False):
    """
    This method converts a measurement to a distribution, to allow flexibility
    in how values are implemented in the various routines. Primarily used
    by :py:meth:`concord.utils.homogenize_distributions`.

    Intended to be very flexible in terms of the way the units are provided,
    but unfortunately np.shape for a tuple with elements having units, does not
    work! So have to pass the units on the entire object

    Also have an issue with combining distributions with and without
    units; "...the NdarrayDistribution will not combine with Quantity
    objects containing units"
    see https://docs.astropy.org/en/stable/uncertainty for more details

    :param num: scalar/array to convert to distribution
    :return: astropy distribution object

    Example usage:

    >>> y = cd.value_to_dist(3.) # scalars are kept as single values
    >>> z = cd.value_to_dist((3.,0.1),nsamp=10) # generate 10 samples from a normal distribution around 3.0 with st. dev 0.1
    >>> a = cd.value_to_dist((3.,0.5,0.1)) # generate default number of samples from an asymmetric Gaussian
    >>> b = cd.value_to_dist(a.distribution) # convert an array to a distribution

    with units:

    >>> z = cd.value_to_dist((3.,0.1)*u.hr,nsamp=10) # generate 10 samples from a normal distribution around 3.0 with st. dev 0.1
    """

    # Check here if it's already a distribution; don't want to run this twice

    if hasattr(_num, 'distribution'):
        if verbose:
            logger.warning('this quantity is already a distribution')
        return _num

    # Want to permit all these combinations:
    # val (if unit is omitted as for the first 3, assumed in MINBAR units)
    # (val, err)
    # (val, loerr, hierr)
    # val*unit
    # (val, err)*unit
    # (val, loerr, hierr)*unit
    num_unit = u.dimensionless_unscaled
    if hasattr(_num, 'unit'):
        # strip off the unit here to make the code which follows, work
        num_unit = _num.unit
        if hasattr(unit, 'unit'):
            assert (unit.unit == num_unit) # supplied unit might have prefactor
        else:
            assert (unit is None) or (unit == num_unit)  # supplied unit might have prefactor
        num = _num.value
    else:
        num = _num
        if unit is not None:
            num_unit = unit

    if np.shape(num) == () or np.shape(num) == (1,):
        # scalar, with no errors; just replicate that value
        # return unc.Distribution(np.full(nsamp, num) * num_unit)
        # since we can combine distributions with scalars, no need to make a
        # distribution out of this scalar
        return num * num_unit
    elif np.shape(num) == (2,):
        if num[1] is None:
            # return unc.Distribution(np.full(nsamp, num[0]) * num_unit)
            return num[0] * num_unit
        # value and error
        return unc.normal(num[0]*num_unit, std=num[1]*num_unit, n_samples=nsamp)
    elif np.shape(num) == (3,):
        # value with asymmetric error; convention is value, err_lo, err_hi
        return unc.Distribution(asym_norm(num[0], num[1], num[2], nsamp)*num_unit)
    else:
        # more than two values indicates a distribution, so just return that
        if len(num) < NSAMP_DEF:
            logger.warning('assuming a distribution despite only {} samples'.format(len(num)))
        return unc.Distribution(num*num_unit)

# ------- --------- --------- --------- --------- --------- --------- ---------

def homogenize_params(theta, nsamp=None):
    """
    This method is used by the MC routines, to ensure consistency of the
    parameter set. We want to make sure that each parameter (if a distribution)
    has the same length as any other parameter (unless it's a scalar).
    Each parameter is labeled and provided with a value (and possibly also
    error), and units. Inclination is also provided with the isotropy
    flag, and the angle limits.

    If none of the input parameters are already distributions, the
    provided number of samples nsamp will be used as the dimensions of the
    output distributions (this may be redundant, since we now also do the
    inclination distributions here)

    Usage:

    (par1, par2, ... nsamp ) = homogenize_params( dictionary_of_input_params_and_units, nsamp )

    Example usage:

    >>> F_pk, _nsamp = homogenize_params( {'fpeak': ((33., 2.), cd.MINBAR_FLUX_UNIT)}, 100)

    >>> F_pk, _inclination, _nsamp = cd.homogenize_params( {'fpeak': ((33., 2.), cd.MINBAR_FLUX_UNIT),
                                   'incl': (None, u.deg, isotropic, 0.0, 75.)}, 100)

    :param theta: dictionary with parameters and units
    :param nsamp: desired size of distribution objects, if not already set by
                  one or more of the parameters
    :return: tuple with all the parameters in the same order they're
        passed, plus the actual number of samples per parameter
    """

    # here we maintain a list of "standard" parameters for use in concord
    # you don't have to use these, but down the track we might do different things
    # based on the parameter type (e.g. for incl)

    params = ['tdel','fluen','fper','c_bol','alpha','fpeak','flux','lum','dist','incl']

    # this bit will allow us to replace the inclination treatement in the individual
    # functions

    isotropic = True
    if 'incl' in theta.keys():
        if len(theta['incl']) < 5:
            logger.error('with the inclination you must pass the isotropy flag and limits')
            # return a tuple of None's here, otherwise we get a confusing "cannot unpack non-iterable
            # NoneType object" error in addition to the logger error
            return (None,)*(len(theta)+1)
        isotropic = theta['incl'][2]

    # first loop we have to get the number of samples
    _nsamp = None
    mismatch = False
    l = [] # array for parameter lengths
    for par in theta.keys():
        if par not in params:
            logger.warning('parameter {} is not in my list')
        l.append( len_dist(theta[par][0]) )
        if (l[-1] > 3):
            # we have a distribution of some kind
            if (_nsamp is None):
                _nsamp = l[-1]
            mismatch = (l[-1] != _nsamp) | mismatch

    if mismatch:
        logger.error("mismatch in distribution sizes")
        # return a tuple of None's here, otherwise we get a confusing "cannot unpack non-iterable
        # NoneType object" error in addition to the logger error
        return (None,)*(len(theta)+1)

    # Set the size for the distributions
    # If any of the inputs are distributions (or we have the isotropy flag unset) we *must*
    # return distributions

    scalar = (max(l) == 1)
    if (not scalar) | (not isotropic):
        if (_nsamp is None):
            if (nsamp is None):
                nsamp = NSAMP_DEF
        else:
            if (nsamp is not None) & (_nsamp != nsamp):
                logger.warning('passed nsamp overridden by size of one or more parameter distribution')
            nsamp = _nsamp
    # print (scalar, _nsamp, nsamp)

    # Now assemble the output tuple
    theta_hom = []
    for i, par in enumerate(theta.keys()):

        # if scalar:
        if l[i] == 1:
            # copy scalars through to the output array, with a unit if not already present
            if (not hasattr(theta[par][0], 'unit')) & (theta[par][0] is not None) & (theta[par][1] is not None):
                theta_hom.append( theta[par][0] * theta[par][1] )
            else:
                theta_hom.append( theta[par][0] )
        else:
            if (par == 'incl'):
                # special here for the inclination
                if (not isotropic) & (theta[par][0] is None):
                    theta_hom.append( iso_dist(nsamp, imin=theta[par][3], imax=theta[par][4]) )
                else:
                    theta_hom.append( theta[par][0] )
            elif (par == 'c_bol') & (l[i] == 0):
                # special here for no supplied bolometric correction
                logger.warning('no bolometric correction applied')
                theta_hom.append( 1.0 )
            else:
                theta_hom.append( value_to_dist(theta[par][0], nsamp=nsamp, unit=theta[par][1]) )

    if scalar:
        theta_hom.append( 1 )
    else:
        theta_hom.append( nsamp )

    return tuple(theta_hom)

# ------- --------- --------- --------- --------- --------- --------- ---------

def len_dist(d):
    """
    Utility routine to replace the len function for distributions, and
    also return sensible values for scalars rather than throwing an error

    :param d:
    :return: length of the array OR distribution
    """

    if d is None:
        return 0
    elif hasattr(d,'distribution'):
        return len(d.distribution)
    elif np.shape(d) != ():
        return np.shape(d)[0]
    else:
        return 1

# ------- --------- --------- --------- --------- --------- --------- ---------

def asym_norm(m, sigm=None, sigp=None, nsamp=NSAMP_DEF, positive=False, model=1):
    '''
    Draw samples from an asymmetric error distribution characterised by a
    mean and upper and lower 68% confidence intervals (sigp and sigm,
    respectively). Used primarily to generate distributions from
    quantities with asymmetric errors, by :py:meth:`concord.utils.value_to_dist`.

    Follows the treatment of `Barlow (2003)
    <https://ui.adsabs.harvard.edu/abs/2003sppp.conf..250B>`_

    With the positive flag set, it will continue drawing samples until all
    are > 0. Note that the resulting distribution may not quite have the
    required shape

    :param m: mean (central) value for the distribution
    :param sigm: lower 68th-percentile uncertainty
    :param sigp: upper 68th-percentile uncertainty
    :param nsamp: number of samples required
    :param positive: ensure all the samples are positive (may affect the
        distribution)
    :param model: implementation of the distribution function; only one
        option is currently implemented
    :return: distribution with ``nsamp`` values having the desired shape
    '''

    if (sigm is None) or (sigp is None):
        # can get the boundaries from a tuple instead
        if len(m) != 3:
            logger.error('in the absence of separate sigma-values supply as a 3-element tuple')
            return None
        sigm = m[1]
        sigp = m[2]
        m = m[0]

    if model !=1:
        logger.error("other types of asymmetric distributions not yet implemented")
        return None

    x = np.random.uniform(size=nsamp)
    bd = abs(sigm) / (sigp + abs(sigm))

    _l = x < bd
    x[_l] = m + erfinv(x[_l] / bd - 1) * 2 ** 0.5 * abs(sigm)
    x[~_l] = m + erfinv((x[~_l] - bd) / (1 - bd)) * 2 ** 0.5 * sigp

    while positive & (np.any(x <= 0.)):
        _l = x <= 0.
        x[_l] = asym_norm(m, sigm, sigp, len(np.where(_l)[0]), model=model)

    return x

# ------- --------- --------- --------- --------- --------- --------- ---------

def intvl_to_errors(perc):
    """
    This function converts a confidence interval defined by a 3-element array
    to the more useful version with errors .

    :param perc: the array of percentiles (central, lower limit, upper limit)
    :return: 3-element array with (central value, lower uncertainty, upper uncertainty)
    """

    perc[1] = perc[0] - perc[1]
    perc[2] = perc[2] - perc[0]

    return perc

# ------- --------- --------- --------- --------- --------- --------- ---------

def g(M,R,Newt=False,units='cm/s^2'):
    '''
    This function calculates the surface gravity given a mass M and radius R,
    using the Newtonian expression if the flag Newt is set to True

    The result is returned in units of cm/s^2 by default

    :param M: neutron-star mass
    :param R: neutron-star radius
    :param Newt: default ``False`` returns the GR value, set to True to
        calculate the Newtonian value
    :param units: the required units
    :return: the surface gravity, in the required units
    '''

    if Newt:
        return (const.G*M/R**2).to(units)
    else:
        #        return const.G*M/(R**2*sqrt(1.-2.*const.G*M/(const.c**2*R))).to(units)
        return redshift(M,R)*g(M,R,Newt=True).to(units)

# ------- --------- --------- --------- --------- --------- --------- ---------

def redshift(M,R):
    '''
    This function calculates the gravitational redshift 1+z

    :param M: neutron-star mass
    :param R: neutron-star radius
    :return: 1+z
    '''

    return 1./sqrt(1.-2.*(const.G*M/(const.c**2*R)).decompose())

# ------- --------- --------- --------- --------- --------- --------- ---------

def check_M_R_opz(M, R, opz):
    """
    Utility routine to check the consistency of the mass, radius and redshift
    passed to various routines (e.g. mdot)

    :param M: neutron star mass, or None
    :param R: neutron star radius, or None
    :param opz: 1+z where z is the surface gravitational redshift, or None
    :return: boolean, ``True`` if the values are consistent, ``False`` otherwise
    """

    # define booleans here for whether or not we've got non-default values
    # might change this down the track and give the default value as None
    M_passed = M is not None
    R_passed = R is not None
    opz_passed = opz is not None
    if ((M_passed & ~R_passed) or (~M_passed & R_passed)):
        if opz_passed:
            logger.warning('possible inconsistency with both redshift and one of (M,R) supplied')
        else:
            logger.error('can''t calculate redshift in the absence of both (M, R)')

    # check here your mass, radius and redshift are consistent
    if M_passed & R_passed:
        opz_check = redshift(M, R)
        if opz_passed and (abs(opz / opz_check - 1) > ETA):
            logger.warning('provided redshift not consistent with M, R ({:.3f} != {:.3f}); overriding'.format(opz,
                                                                                                                   opz_check))
            return False

    return True

# ------- --------- --------- --------- --------- --------- --------- ---------

def calc_mr(g,opz):
    ''''
    Calculates neutron star mass and radius given a surface gravity and
    redshift. This routine is required when (for example) comparing a
    model computed at a fixed gravity, with observations requiring a
    particular value of redshift (to stretch the model lightcurve to match
    the observed one). The combination of surface gravity and redshift
    implies unique mass and radius, and the constraints on 1+z can be
    translated to constraints on M, R

    :param g: surface gravity, in units equivalent to cm/s**2
    :param opz: surface redshift 1+z
    :return: neutron star mass in g, radius in cm
    '''

    # First some checks

    if hasattr(opz,'unit'):
        assert (opz.unit == '')
    try:
        test = g.to('cm / s2')
    except ValueError:
        logger.error("incorrect units for surface gravity")
        return -1, -1

    # Now calculate the mass and radius and convert to cgs units

    R = (const.c**2*(opz**2-1)/(2.*g*opz)).to(u.cm)
    M = (g*R**2/(const.G*opz)).to(u.g)

    return M, R

# ------- --------- --------- --------- --------- --------- --------- ---------

def solve_radius(M,R_Newt,eta=1e-6):
    '''
    This routine determines the GR radius given the NS mass and Newtonian
    radius, assuming the GR and Newtonian masses are identical

    Solving is tricky so we just use an iterative approach

    :param M: neutron-star mass
    :param R_Newt: Newtonian neutron-star radius
    :param eta: tolerance for convergence
    :return: neutron-star radius R in GR
    '''

    R = R_Newt	# trial
    while (abs(g(M,R)-g(M,R_Newt,Newt=True))/g(M,R) > eta):
        R = R_Newt*sqrt(redshift(M,R))

    return R

# ------- --------- --------- --------- --------- --------- --------- ---------

def decode_LaTeX(string, delim='pm'):
    '''
    This function converts a LaTeX numerical value (with error) to one
    or more floating point values. It is expected that the number is formatted
    as ``$3.1\pm1.2$``, and the default delimiter assumes this scenario

    Will not work on asymmetric errors, expressed (for example) as
    ``$1.4_{-0.1}^{+0.2}$``

    Example usage:

    >>> cd.decode_LaTeX('$1.4\pm0.3')
    (1.4, 0.3)

    :param string: string with LaTeX numerical value and error
    :param delim: separator of number and uncertainty; defaults to ``\pm``
    :return: tuple giving the value and error(s)
    '''

    assert (type(string) == str) or (type(string) == np.str_)

    # Look for the start of a LaTeX numerical expression
    # We no longer explicitly look for the $ sign, as this pattern can match a
    # general numeric string. We also include the mantissa as optional

    val_match = re.search('([0-9]+(\.[0-9]+)?)',string)

    # If not found, presumably you just have a numerical expression as string

    if val_match is None:
        return float(string), None

    # Otherwise, convert the first part to a float, and look for the error

    val = float(val_match.group(1))
    # err_match = re.search('pm *([0-9]+\.[0-9]+)',string)
    err_match = re.search('([0-9]+(\.[0-9]+)?)',string[val_match.span()[1]:])

    if err_match == None:
        return val, None
    else:
        return val, float(err_match.group(1))

# ------- --------- --------- --------- --------- --------- --------- ---------

def tdel_dist(nburst, exp, nsamp=NSAMP_DEF):
    """
    Function to generate a synthetic distribution of recurrence times, given
    nburst observed events over a total exposure of exp. The resulting
    distribution may approximate the PDF of the underlying burst rate, but
    does assume that the bursts are independent, which is generally not the
    case

    :param nburst: number of events detected
    :param exp: total exposure time
    :param nsamp: number of samples to generate
    :return: distribution giving the PDF of the average recurrence time
    """

    if exp.decompose().unit != u.s:
        logger.error('exposure time needs a unit')
        return None

    # Generate a CDF of the underlying event rate, given nburst events detected

    mu = np.arange(15000) / 500.  # [0-30]
    prob = 1 - poisson.cdf(nburst, mu)

    # Need to make sure the range of mu is sufficiently wide; you might need
    # to tweak this for nburst much larger than 5 or so

    assert (1. - max(prob)) < 1e-6

    x = np.random.random(nsamp)
    y = [mu[np.argmin(abs(prob - _x))] for _x in x]

    return unc.Distribution( exp / y )

# ------- --------- --------- --------- --------- --------- --------- ---------

def Q_nuc(Xbar, quadratic=True, old_relation=False, coeff=False):
    '''
    This function implements the approximation to the nuclear energy
    generation rate Q_nuc given the mean hydrogen fraction in the fuel
    layer, Xbar, determined by `Goodwin et al. 2019 (ApJ 870, #64)
    <https://ui.adsabs.harvard.edu/abs/2019ApJ...870...64G>`_. The
    value returned is in units of MeV/nucleon:

    >>> q = Q_nuc(0.73)
    5.35517578

    There are three versions of the approximation, which can be selected
    by using the quadratic and old_relation flags; by default the most
    precise (and recent) version is used:

    >>> q = Q_nuc(0.73, quadratic=False)
    5.758715
    >>> q = Q_nuc(0.73, old_relation=True)
    4.52

    If you want the coefficients, rather than the actual Q_nuc value, you
    can use the coeff flag (in which case the first argument is ignored):

    >>> q_0, q_1 = Q_nuc(0.0,quadratic=False,coeff=True)
    [1.3455, 6.0455]

    :param Xbar: average hydrogen mass fraction of the burst column
    :param quadratic: set to ``True`` to use the (more accurate) quadratic approximation
    :param old_relation: set to ``True`` to use the earlier approximation
    :param coeff: set to ``True`` to return the coefficients used, in which
                  case ``Xbar`` is ignored
    :return: Q_nuc value, in MeV/nucleon, or the coefficients if ``coeff=True``
    '''

    q = [1.3050, 6.9511, -1.9218]

    if not quadratic:
        q = [1.3455, 6.0455]

    if old_relation:
        q = [1.6, 4.0]

    if coeff:
        return q

    return sum([q[i] * Xbar ** i for i in range(len(q))])

# ------- --------- --------- --------- --------- --------- --------- ---------

def X_0(Xbar, zcno, tdel, opz=OPZ, debug=False, old_relation=False):
    '''
    Routine (extracted from :py:meth:`concord.utils.hfrac`) to determine
    the accreted H fraction X_0 given the average H-fraction at ignition,
    Xbar, the CNO metallicity zcno, and the burst recurrence time, tdel.

    The time to exhaust the accreted hydrogen is calculated according to
    the formula derived by `Lampe et al. (2016, ApJ 819, #46)
    <http://adsabs.harvard.edu/abs/2016ApJ...819...46L>`_; if the
    ``old_relation=True`` flag is set, the alternative prefactor quoted by
    `Galloway et al. (2004 ApJ 601, 466)
    <https://ui.adsabs.harvard.edu/abs/2004ApJ...601..466G>`_ is used instead

    :param Xbar: average hydrogen mass fraction of the burst column
    :param zcno: CNO mass fraction in the burst fuel
    :param tdel: burst recurrence time
    :param opz: NS surface redshift, 1+z
    :param old_relation: use the older expression for the time to exhaust the accreted hydrogen
    :param debug: display debugging messages

    :return: inferred hydrogen mass fraction X_0 of the accreted fuel
    '''

    if max([len_dist(Xbar), len_dist(zcno), len_dist(tdel), len_dist(opz)]) > 1:
        logger.error('scalar arguments only')

    tpref = TPREF_CNO

    # Here we allow an option to use the old version if need be; this is already passed to the
    # Q_nuc function, so we'll get the old parameters from that if need be

    if old_relation:
        tpref = TPREF_CNO_OLD

    # Also here now we distinguish between the two possible cases, the first where there
    # remains hydrogen at the base, and the second where it is exhausted
    # This division is quantified via the f_burn fraction, which initially at least we
    # must estimate

    t_CNO = tpref * (Xbar / 0.7) / (zcno / 0.02)
    f_burn = tdel / (opz * t_CNO)
    f_burn_prev = 1. / f_burn  # dummy value first off to ensure the loop gets called
    if debug:
        print ('X_0: trial: t_CNO = {:.2f}, f_burn = {:.4f}'.format(t_CNO, f_burn))

    # Loop here to make sure we have a consistent choice
    # There are different formulae for the two different cases: where H is
    # exhausted at the base, and where some remains. You only need to run the
    # loop twice if we switch

    while (1. - f_burn) * (1. - f_burn_prev) < 0.:

        if (f_burn <= 1.):
            # still hydrogen at the base
            x_0 = Xbar + 0.35 * (tdel / (opz * tpref)) * (zcno / 0.02)
        else:
            # hydrogen exhausted at the base
            x_0 = np.sqrt(1.4 * Xbar * tdel / (opz * tpref) * (zcno / 0.02))

        f_burn_prev = f_burn
        t_CNO = tpref * (x_0 / 0.7) / (zcno / 0.02)
        f_burn = tdel / (opz * t_CNO)

        if debug:
            print('X_0: Xbar = {:.4f}, t_CNO = {:.4f}, x_0 = {:.4f}'.format(
                Xbar, t_CNO, x_0))#, xi_b, xi_p)
            print('flipping f_burn {} -> {}'.format(f_burn_prev, f_burn))

    #        print (i,xbar,t_CNO,f_burn,x_0,xi_b,xi_p)
    return x_0

# ------- --------- --------- --------- --------- --------- --------- ---------

def alpha(_tdel, _fluen, _fper, c_bol=None, nsamp=NSAMP_DEF, conf=CONF, fulldist=False):
    """
    Routine to calculate alpha, the observed ratio of persistent to burst
    energy (integrated over the recurrence time, tdel) from the input
    measurables

    Usage:

    >>> alpha = cd.alpha(2.681, 0.381, 3.72, 1.45)
    >>> print (alpha)
    136.64233701

    >>> alpha = cd.alpha((2.681, 0.007), (0.381, 0.003), (3.72, 0.18), (1.45, 0.09))
    >>> print (alpha)
    [136.23913536   9.88340505  11.21502717]

    :param tdel: burst recurrence time
    :param fluen: burst fluence
    :param fper: persistent flux
    :param c_bol: bolometric correction on persistent flux
    :return: alpha-values, either a scalar (if all the inputs are also
	scalars); a central value and confidence range; or (if
        ``fulldist=True``) a dictionary with the distributions of the
	alpha value and the input or adopted distributions of the
        intermediate values
    """

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    tdel, fluen, fper, _c_bol, _nsamp = homogenize_params( {'tdel': (_tdel, u.hr),
                                                            'fluen': (_fluen, MINBAR_FLUEN_UNIT),
                                                            'fper': (_fper, MINBAR_FLUX_UNIT),
                                                            'c_bol': (c_bol, None)}, nsamp )

    _alpha = (fper * _c_bol * tdel / fluen).decompose()

    if len_dist(_alpha) == 1:
        return _alpha

    if fulldist:
        return {'alpha': _alpha, 'tdel': tdel, 'fluen': fluen, 'fper': fper, 'c_bol': _c_bol}

    ac = np.percentile(_alpha.distribution, [50, 50 - conf / 2, 50 + conf / 2])

    return intvl_to_errors(ac)

# ------- --------- --------- --------- --------- --------- --------- ---------
def _i(obj, ind):
    """
    Utility function to permit slicing of arbitrary objects, including scalars
    (in which case the scalar is returned). For use with hfrac

    :param obj: array or scalar
    :param ind: index of value to return
    :return: slice of array or the scalar
    """

    if hasattr(obj,'distribution'):
        return obj.distribution[ind]
    try:
        return obj[ind]
    except:
        return obj

# ------- --------- --------- --------- --------- --------- --------- ---------

def hfrac(_tdel, _alpha=None, fper=None, fluen=None, c_bol=1.0,
          opz=OPZ, zcno=0.02, old_relation=False,
          isotropic=False, inclination=None, imin=0.0, imax=IMAX_NDIP,
          model='he16_a', conf=CONF, fulldist=False, nsamp=None, debug=False):
    '''
    Estimates the H-fraction at burst ignition, based on the burst properties
    In the absence of the alpha-value(s), you need to supply the
    persistent flux and burst fluence, along with the recurrence time, so
    that alpha can be calculated

    There is also a mode for calculating a single value of the parameters,
    based on a single value of alpha, tdel and the inclination; this mode
    is used by the function itself, within a loop over the inclination
    values.

    Example usage:

    >>> import astropy.units as u
    >>> cd.hfrac(2.5*u.hr, 140., inclination=30.)
    ** WARNING ** assuming inclination in degrees
    (<Quantity 0.19564785>, <Quantity 0.26656582>, 30.0)

    >>> cd.hfrac((2.681, 0.007), fluen=(0.381, 0.003),
            fper=(3.72, 0.18), c_bol=(1.45, 0.09),nsamp=100,fulldist=True)
    {'xbar': NdarrayDistribution([ 0.15246606,  0.20964612,  0.14169592,  0.11998812,  0.14293726,
        0.13757633, -0.0850957 ,  0.27077167,  0.01620781,  0.13673547,
       -0.00821256,  0.10193499, -0.08151998, -0.0370918 ,  0.26400253,
        0.1813988 , -0.04371571,  0.14432454,  0.28422351, -0.04962202,
		.
		.
		.
     'X_0': NdarrayDistribution([ 0.22852126,  0.28567535,  0.21799341,  0.19609461,  0.21895133,
		.
		.
		.
     'i': <QuantityDistribution [45.58518673, 15.61321579, 36.30632308, 48.18062101, 45.46515182,
		.
		.
		.

    :param tdel: burst recurrence time
    :param alpha: burst alpha, which (if not supplied) can be calculated
        instead from the fluence, fper, and c_bol
    :param fper: persistent flux
    :param fluen: burst fluence
    :param c_bol: bolometric correction on persistent flux
    :param opz: neutron-star surface redshift
    :param zcno: CNO mass fraction in the burst fuel
    :param old_relation: flag to use the old (incorrect) relation for Q_nuc
    :param isotropic: set to True to assume isotropic emission
    :param inclination: inclination value (or distribution)
    :param imin: minimum of allowed inclination range
    :param imax: maximum of allowed inclination range
    :param conf: confidence interval for output limits
    :param model: model string of He & Keek (2016), defaults to "A"
    :param conf: confidence % level for limits, ignored if fulldist=True
    :param fulldist: set to True to return the distributions of each parameter
    :param debug: display debugging messages
    :return: a tuple of values of Xbar, X_0 and inclination; or a
        dictionary including distributions of each of the values, as well
        as the distributions adopted for the observables
    '''

    if _alpha is None:
        # Now have the option of providing all the parameters for alpha, instead of the
        # values. But in that case you need to calculate alpha

        if (fper is None) or (fluen is None):
            logger.error('need to supply persistent flux & fluence in absence of alpha')
            return None
        if c_bol == 1.0:
            logger.warning('no bolometric correction applied in alpha calculation')

    # Flag to check for the "single" mode of operation, used by this routine

    scalar = (len_dist(_tdel) == 1) & ((len_dist(inclination) == 1) or isotropic) & \
              ((len_dist(_alpha) == 1) or ((_alpha is None) \
              & len_dist(fper) == len_dist(fluen) == len_dist(c_bol) == 1))
    # mode_single = ( (_alpha is not None) & (inclination is not None) &
    #               (np.shape(_alpha) == () ) & (not hasattr(_alpha,'distribution')) )
    if debug:
        print ('hfrac: mode_single = ',scalar)

    # The following parameters define the relation for Q_nuc = q_0 + q_1*hbar, where hbar is the
    # average hydrogen fraction in the fuel column at ignition

    q_0, q_1 = Q_nuc(0.0, old_relation=old_relation, quadratic=False, coeff=True)

    # Does this need to change if the Q_nuc coefficients also change? YES
    # alpha_0 = const.c**2/(6.6*u.MeV/const.m_p).to("m**2/s**2")
    alpha_0 = const.c ** 2 / (q_1 * u.MeV / const.m_p).to("m**2/s**2")

    # This parameter sets the prefactor for the time to burn all H via hot-CNO; see
    # Lampe et al. (2016, ApJ 819, 46)

    tpref = TPREF_CNO

    # Here we allow an option to use the old version if need be; this is already passed to the
    # Q_nuc function, so we'll get the old parameters from that if need be

    if old_relation:
        tpref = TPREF_CNO_OLD

    # convert the tdel parameter provided, to a distribution (if required)
    # and check that the other parameters have consistent sizes
    if scalar:
        tdel = _tdel
        if _alpha is None:
            alpha_dist = alpha(tdel, fluen, fper, c_bol)
        else:
            alpha_dist = _alpha
    else:
        if _alpha is None:
            # If you've not passed alpha, then we need to calculate it here, and copy all
            # the other parameters to make sure we save them for the output dict
            # The check for consistent array (distribution) sizes is done by alpha (but
            # excludes the inclination)

            _alpha = alpha(_tdel, fluen, fper, c_bol, nsamp=nsamp, fulldist=True)
            alpha_dist = _alpha['alpha']
            tdel = _alpha['tdel']
            _fper = _alpha['fper']
            _fluen = _alpha['fluen']
            _nsamp = len_dist(alpha_dist)
        else:
            # On the other hand if you pass alpha, we just need to make sure the arrays
            # (distributions) have consistent sizes

            tdel, alpha_dist, _nsamp = homogenize_params( {'tdel': (_tdel, u.hr),
                                                           'alpha': (_alpha, None)}, nsamp)
            _fper, _fluen = fper, fluen

        # if no sample size has been passed, but we still need to generate samples
        # to account for the emission anisotropy, determine the array size here
        if (nsamp is None) | (not isotropic):
            nsamp = NSAMP_DEF
            if (_nsamp > 3):
                nsamp = _nsamp

    # Set the inclination parameters, by default for isotropy
    xi_b = 1.
    xi_p = 1.
    if not isotropic:
        if inclination is None:
            # With no inclination provided, we set up a uniform distribution of values
            # up to some maximum, and calculate the composition values for each inclination
            inclination = iso_dist(nsamp, imin=imin, imax=imax)

        # Calculate the anisotropy factors for the inclination
        xi_b, xi_p = anisotropy(inclination, model=model)

    xmax = 0.77  # no longer used

    if scalar:
        # I think this part only gets called for scalar values, but you should check here

        # this will fail for isotropic=True
        # assert len_dist(tdel) == len_dist(alpha_dist) == len_dist(inclination) == 1
        assert len_dist(tdel) == len_dist(alpha_dist) == len_dist(xi_b) == len_dist(xi_p) == 1

        # And finally calculate the hydrogen fraction(s)
        # Early on I was limiting this by xmax, but we want to do that in the
        # calling routine, to make it clear where we exceed the likely maximum

        xbar = min([xmax,(opz-1.)*alpha_0/alpha_dist*(xi_b/xi_p)-q_0/q_1])
        x_0 = -1  # dummy value

        #    x_0 = min([xmax,xbar+0.35*(tdel/(opz*9.8*u.hr))*(zcno/0.02)])
        #    for i in range(10):
        if xbar > 0.0:
        # split this off as a separate routine
            x_0 = X_0(xbar, zcno, tdel, opz, debug=debug, old_relation=old_relation)

        # Although it's probably a mistake, return a tuple of values for the scalar version
        # in contrast to the fulldist one, which returns a dict
        return xbar, x_0, inclination
        # return {'xbar': xbar, 'X_0': x_0, 'i': inclination, 'alpha': alpha_dist,
        #         'fluen': fluen, 'fper': fper, 'model': model}

    else:

        # We loop over each of the inclination values and calculate the corresponding
        # properties
        # Original version of this had tdel, alpha as distributions, but all the other
        # parameters not. This makes it a bit hard to mix single values and distributions,
        # not least because you can't index floats (or single element quantities)
        # Now we work around this using the _i utility function, which returns a slice
        # of an array OR a scalar
        xbar = np.zeros(nsamp)
        x_0 = np.zeros(nsamp)
        for j, i in enumerate(inclination.distribution):

            # print ('{}/{}: tdel={}, alpha={}, opz={}, zcon={}\n'.format(
            #     j, len_dist(inclination),_i(tdel,j), _i(alpha_dist,j), _i(opz, j), _i(zcno, j)))
            xbar[j], x_0[j], dummy = hfrac( _i(tdel,j), _i(alpha_dist,j), _i(opz, j), _i(zcno, j),
                isotropic=isotropic, old_relation=old_relation, inclination=i,
                debug=debug)

        if fulldist:
            return {'xbar': unc.Distribution(xbar), 'X_0': unc.Distribution(x_0), 'i': inclination,
                    'alpha': alpha_dist, 'fluen': _fluen, 'fper': _fper, 'model': model}

        cval = [50, 50 - conf / 2, 50 + conf / 2]
        return { 'xbar': intvl_to_errors(np.percentile(xbar, cval)),
                 'X_0': intvl_to_errors(np.percentile(x_0[x_0 >= 0.], cval)),
                 'i': intvl_to_errors(np.percentile(inclination[x_0 >= 0.], cval)),
                 'alpha': alpha_dist, 'fluen': _fluen, 'fper': _fper, 'model': model}

# ------- --------- --------- --------- --------- --------- --------- ---------

def iso_dist(nsamp=NSAMP_DEF, imin=0., imax=IMAX_NDIP):
    '''
    Routine to generate an isotropic distribution of inclinations (angle
    from the system rotation axis to the line of sight) from imin up to
    some maximum value, defaulting to 75 degrees. This value corresponds
    to the likely maximum possible for a non-dipping source.

    :param nsamp: number of samples required
    :param imin: lower limit of distribution in degrees (default is zero)
    :param imax: upper limit of distribution in degrees (default is IMAX_NDIP)
    :return: distribution of inclinations between ``imin``, ``imax``
    '''

    if imin < 0. or imin > imax or imax > 90.:
        logger.error("imin, imax must be in the range [0,90] degrees")
        return None

    # Convert to radians
    cos_imin = np.cos(imin * np.pi / 180.)
    cos_imax = np.cos(imax * np.pi / 180.)
    # print(cos_imin, cos_imax)
    # uniform in cos(i) up to i=imax

    # return np.arccos(cos_imax + np.random.random(nsamp) * (cos_imin - cos_imax)) * 180. / np.pi * u.deg
    return unc.Distribution(np.arccos(cos_imax
                + np.random.random(nsamp) * (cos_imin - cos_imax))*u.radian).to('deg')

# ------- --------- --------- --------- --------- --------- --------- ---------

def dist(_F_pk, nsamp=None, dip=False,
         empirical=False, X=0.0, M=M_NS, opz=OPZ, T_e=0.0,
         isotropic=False, inclination=None, imin=0., imax=IMAX_NDIP,
         model='he16_a', conf=CONF, fulldist=False, plot=False):
    """
    This routine estimates the distance to the source, based on the
    measured peak flux of a radius-expansion burst. Two main options are
    available;
    ``empirical=False`` (default), in which case the Eddington luminosity is
    calculated as equation 7 from `Galloway et al. (2008, ApJS 179, 360) 
    <http://adsabs.harvard.edu/abs/2008ApJS..179..360G>`_; or
    ``empirical=True``, in which case the estimate of `Kuulkers et al. 2003
    (A&A 399, 663) <http://adsabs.harvard.edu/abs/2003A%26A...399..663K>`_
    is used. The default ``isotropic=False`` option also takes into account
    the likely effect of anisotropic burst emission, based on the models
    provided by concord

    Example usage:

    >>> cd.dist((30., 3.), isotropic=True, empirical=True)
    <Quantity [10.27267949,  0.50629732,  0.59722829] kpc>
    >>> cd.dist((30., 2., 5.), isotropic=False, empirical=True)
    <Quantity [10.75035916,  1.39721634,  1.461003  ] kpc>
    >>> cd.dist((30., 3.), isotropic=False, empirical=True, fulldist=True, nsamp=100)
    {'dist': <QuantityDistribution [11.12852694, 11.4435149 ,  9.79809775,  9.82146105,  9.34573615,
            10.4330567 , 10.99421133, 10.66816313,  8.83667458, 12.83249538,
            14.15062276, 13.26883293,  9.80702496,  8.72915705, 12.04546987,
                .
                .
                .

    :param _F_pk: peak burst flux
    :param nsamp: number of samples required for the distributions
    :param dip: set to True if the source is a "dipper"
    :param empirical: flag to use the empirical Eddington luminosity
    :param X: hydrogen mass fraction in the atmosphere
    :param M: neutron-star mass
    :param opz: neutron-star surface redshift
    :param T_e: temperature factor used in the theoretical L_Edd expression
    :param isotropic: set to True to assume isotropic emission
    :param inclination: inclination value (or distribution)
    :param imin: minimum of allowed inclination range
    :param imax: maximum of allowed inclination range
    :param conf: confidence interval for output limits
    :param fulldist: set to True to output full distributions for all parameters
    :param plot: set to True to generate a simple plot
    :return: inferred distance values, either a scalar (if all the inputs
        are also scalars); a central value and confidence range; or (if
        ``fulldist=True``) a dictionary with the distributions of the
	distance and the input or adopted distributions of the
        other parameters
    """

    alpha_T = 2.2e-9  # K^-1

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    F_pk, _inclination, _nsamp = homogenize_params( {'fpeak': (_F_pk, MINBAR_FLUX_UNIT),
                                                      'incl': (inclination, u.deg, isotropic, imin, imax)}, nsamp)

    # if no sample size has been passed, but we still need to generate samples
    # to account for the emission anisotropy, determine the array size here
    # if (nsamp is None) | (not isotropic):
    #     nsamp = NSAMP_DEF
    #     if (_nsamp > 3):
    #         nsamp = _nsamp

    if isotropic:
        xi_b, xi_p = 1., 1.
        # print ('take into account the disk effect')
        if dip == True:
            logger.warning('isotropic distribution not correct for dipping sources')
    else:
        # set up the nclination array. If it's not been passed to the function,
        # and we're not calculating the isotropic value, you need to generate a
        # distribution. If the peak flux is already a distribution you need to match
        # it's size, but if it's a single value you can just use the default sample
        # size
        # if (len_dist(inclination) <= 1):
        #     inclination = iso_dist(nsamp, imin=imin, imax=imax)

        xi_b, xi_p = anisotropy(_inclination, model=model)

    # Choose the Eddington flux value to compare the distance against

    if empirical:

        # Kuulkers et al. 2003, A&A 399, 663
        # L_Edd = 3.79e38 * u.erg / u.s
        # L_Edd_err = 0.15e38 * u.erg / u.s
        L_Edd = unc.normal(3.79e38 * u.erg / u.s, std=0.15e38 * u.erg / u.s, n_samples=_nsamp)
    else:

        # Galloway et al. 2008, ApJS 179, 360
        L_Edd = 2.7e38 * ((M / (1.4 * u.M_sun)) * (1 + (alpha_T * T_e) ** 0.86) / (1 + X)
                          / (opz / 1.31)) * u.erg / u.s
        L_Edd_err = 0. * u.erg / u.s

    dist = np.sqrt(L_Edd / (4 * np.pi * F_pk * xi_b)).to('kpc')

    if len_dist(dist) == 1:

        return dist #, dist_iso_err

    pc = dist.pdf_percentiles([50, 50 - conf / 2, 50 + conf / 2])

    # we have a distribution, so calculate the percentiles and plot if required
    if plot:
        # Do a simple plot of the distance distribution

        with quantity_support():
            plt.hist(dist.distribution, bins=50, density=True)
        plt.xlabel('Distance (kpc)')
        plt.axvline(pc[0], color='g')
        plt.axvline(pc[1], color='g', ls='--')
        plt.axvline(pc[2], color='g', ls='--')
        # plt.show()
    # else:
    # print (plot)

    if fulldist:

        # Return a dictionary with all the parameters you'll need
        return {'dist': dist, 'peak_flux': F_pk, 'i': _inclination, 'xi_b': xi_b, 'model': model}
    else:

        # Return the median value and the lower and upper errors
        return intvl_to_errors(pc)
        # return (pc[0], pc[0]-pc[1], pc[2]-pc[0])

# ------- --------- --------- --------- --------- --------- --------- ---------

def luminosity(_F_X, dist=None, c_bol=None, nsamp=None, burst=True, dip=False,
               isotropic=False, inclination=None, imin=0.0, imax=IMAX_NDIP,
               model='he16_a', conf=CONF, fulldist=False, plot=False):
    """
    Calculate the inferred luminosity given a measured X-ray flux and distance
    The flux, error and distance can be single values, or arrays (in which case
    it's assumed that they're distributions)

    This is a more sophisticated version of the L_Edd routine (forgot I had that)

    Example usage:

    Calculate the isotropic luminosity corresponding to a flux
    of 3e-9 erg/cm^2/s at 7.3 kpc

    >>> import concord as cd
    cd.luminosity(3e-9,7.3,isotropic=True)

    Calculate the range of luminosities corresponding to a persistent
    flux of 3e-9 erg/cm^2/s at 7.3 kpc, and assuming isotropic
    inclination distribution (i < 75 deg)

    >>> cd.luminosity(3e-9,7.3,burst=False)

    Calculate the range of luminosities corresponding to a burst flux
    of 3e-8, with uncertainty 1e-9, and an inclination of 45-60 degrees,
    plot (and return) the resulting distribution

    >>> cd.luminosity((3e-8,1e-9),7.3,imin=45,imax=60,plot=True,fulldist=True)

    :param F_X: X-ray flux, MINBAR units assumed if not present
    :param dist: distance to the source
    :param c_bol: bolometric correction to apply
    :param nsamp: number of samples
    :param burst: set to True for burst emission, to select the model anisotropy for bursts
    :param dip: set to True for a dipping source (deprecated)
    :param isotropic: set to True if isotropic value required
    :param inclination: system inclination or distribution thereof
    :param imin: minimum inclination for generated distribution
    :param imax: maximum inclination for generated distribution
    :param model: model string of He & Keek (2016), defaults to "A"
    :param conf: confidence % level for limits, ignored if fulldist=True
    :param fulldist: set to True to return the distributions of each parameter
    :param plot: plots the resulting distributions
    :return:
    """

    # if a distance is not supplied, use a reasonable value, but flag it
    if dist is None:
        dist = 8.0*u.kpc
        logger.warning('assuming distance of {:3.1f}'.format(dist))

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    F_X, _dist, _inclination, _c_bol, _nsamp = homogenize_params( {'flux': (_F_X, MINBAR_FLUX_UNIT),
                                                                   'dist': (dist, u.kpc),
                                                                   'incl': (inclination, u.deg, isotropic, imin, imax),
                                                                   'c_bol': (c_bol, None)}, nsamp)

    # if no sample size has been passed, but we still need to generate samples
    # to account for the emission anisotropy, determine the array size here
    # if (nsamp is None) | (not isotropic):
    #     nsamp = NSAMP_DEF
    #     if (_nsamp > 3):
    #         nsamp = _nsamp,

    # set the nominal distance for the plot labels, dist0
    if hasattr(dist, 'distribution'):
        dist0 = dist.pdf_percentile(50)
    elif len_dist(dist) == 1:
        dist0 = dist
    else:
        dist0 = dist[0]
#        _dist = value_to_dist(_dist, nsamp=nsamp, unit=u.kpc)

    label = 'Isotropic luminosity @ {:.2f} (erg/s)'.format(dist0)

    if isotropic:
        xi_b, xi_p = 1., 1.
        # print ('take into account the disk effect')
        if dip == True:
            logger.warning('isotropic distribution not correct for dipping sources')
    else:
        # if (len_dist(inclination) <= 1):
        #     inclination = iso_dist(nsamp, imin=imin, imax=imax)

        xi_b, xi_p = anisotropy(_inclination, model=model)
        label = 'Luminosity @ {:.2f} (erg/s)'.format(dist0)

        if not burst:
            # If you're calculating the persistent flux/luminosity, better use the right \xi
            logger.info('adopting anisotropy model {} for persistent emission'.format(model))
            xi_b = xi_p

    lum = (4 * np.pi * F_X * _c_bol * _dist ** 2 * xi_b).to('erg s-1')

    if len_dist(lum) == 1:

        # Just return the value; we only keep F_X as scalar if there's no
        # uncertainty provided
        return lum

    lc = lum.pdf_percentiles([50, 50 - conf / 2, 50 + conf / 2])

    if plot:
        # Do a simple plot of the distance distribution

        plt.hist(lum.distribution / (u.erg/u.s), bins=50, density=True)
        plt.xlabel(label)
        plt.axvline(lc[0].value, color='g')
        plt.axvline(lc[1].value, color='g', ls='--')
        plt.axvline(lc[2].value, color='g', ls='--')
        # plt.show()
    # else:
    # print (plot)

    if fulldist:

        # Return a dictionary with all the parameters you'll need
        return {'lum': lum, 'flux': F_X, 'c_bol': _c_bol, 'dist': _dist, 'i': _inclination, 'xi': xi_b, 'model': model}#, 'conf': conf}

    else:

        # Return the median value and the (asymmetric) lower and upper errors

        # return np.median(lum), np.percentile(lum, (50-conf/2, 50+conf/2)) * (u.erg/u.s) - np.median(lum)
        return intvl_to_errors(lc)

# ------- --------- --------- --------- --------- --------- --------- ---------

def L_Edd(F_pk, dist=8 * u.kpc, nsamp=NSAMP_DEF,
	  isotropic = False, inclination = None, imin = 0.0, imax = IMAX_NDIP,
          dip = False, conf=CONF, fulldist=False):
    '''
    This routine estimates the Eddington luminosity of the source, based
    on the measured peak flux of a radius-expansion burst and the distance
    (or some distribution thereof.  The default isotropic=False option also
    takes into account the likely effect of anisotropic burst emission, based
    on the models provided by concord

    This routine has been supplanted by :py:meth:`concord.utils.luminosity`, which
    it now calls; the only thing not incorporated into the new routine is
    the "simple" estimate of the isotropic luminosity error,

    ``L_Edd_iso_err = L_Edd_iso * np.sqrt((F_pk_err / F_pk) ** 2 + (2. * dist_err / dist) ** 2)``

    Burst emission is assumed (``burst=True``); apart from the F_pk value,
    parameters are as for :py:meth:`concord.utils.luminosity`

    :param F_pk: peak burst flux, MINBAR units assumed if not present
    :return: dictionary including calculation results and assumed distributions (as for luminosity)
    '''

    return luminosity(F_pk, dist, nsamp, isotropic, burst=True,
                   imin=imin, imax=imax, model='he16_a', conf=conf, fulldist=fulldist)

# ------- --------- --------- --------- --------- --------- --------- ---------


def mdot(_F_per, _dist, c_bol=None, M=None, R=None, opz=None,
         isotropic=False, inclination=None, imin=0.0, imax=IMAX_NDIP, dip=False,
         model='he16_a', nsamp=None, conf=CONF, fulldist=False):
    '''
    Routine to estimate the mdot given a (bolometric) persistent flux,
    distance, and inclination. This calculation was adapted initially from
    equation 2 of `Galloway et al. (2008, ApJS 179, 360) 
    <http://adsabs.harvard.edu/abs/2008ApJS..179..360G>`_, and uses an
    approximation to ``Q_grav = c**2*z/(1+z) \approx GM_NS/R_NS``
    which is good to about 10% for a typical neutron star

    Usage:

    >>> import concord as cd
    # calculate the mdot corresponding to a flux of 1e-9 at 10kpc, for a 10km
    # 1.4 M_sun neutron star (this is the prefactor for equation 2 in Galloway
    # et al. 2008)
    >>> import astropy.constants as c
    >>> cd.mdot(1., 10., M=1.4*c.M_sun, R=10.*u.km, isotropic=True)
    WARNING:homogenize_params:no bolometric correction applied
    <Quantity 6691.30392224 g / (cm2 s)>

    :param F_per: persistent flux, MINBAR units assumed if not present
    :param dist: distance to the source
    :param c_bol: bolometric correction to apply
    :param M: neutron-star mass
    :param R: neutron-star radius
    :param opz: 1+z where z is the surface gravitational redshift, or None
    :param isotropic: set to True if isotropic value required
    :param inclination: system inclination or distribution thereof
    :param imin: minimum inclination for generated distribution
    :param imax: maximum inclination for generated distribution
    :param dip: set to True for a dipping source (deprecated)
    :param model: model string of He & Keek (2016), defaults to "A"
    :param nsamp: number of samples to generate
    :param conf: confidence % level for limits, ignored if fulldist=True
    :param fulldist: set to True to return the distributions of each parameter
    :return: mdot-values, either a scalar (if all the inputs are also
	scalars); a central value and confidence range; or (if
        ``fulldist=True``) a dictionary with the distributions of the
	mdot value and the input or adopted distributions of the
        intermediate values
    '''

    mdot_unit = 'g cm-2 s-1'

    if not check_M_R_opz(M, R, opz):
        opz = redshift(M, R)
    if M is None:
        M = M_NS
    if R is None:
        R = R_NS
    if opz is None:
        opz = redshift(M, R)

    if hasattr(_F_per,'unit'):
        flux_unit = _F_per.unit
    else:
        flux_unit = MINBAR_FLUX_UNIT

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    F_per, dist, _c_bol, _inclination, nsamp = homogenize_params( {'fper': (_F_per, flux_unit),
                                                      'dist': (_dist, u.kpc),
                                                      'c_bol': (c_bol, None),
                                                      'incl': (inclination, u.deg, isotropic, imin, imax)},
                                                      nsamp )
#    dist = _dist
#    if not hasattr(dist, 'unit'):
#        print('** WARNING ** assuming units of {} for distance'.format(u.kpc))
#        dist *= u.kpc

#    scalar = (len_dist(_F_per) == 1) & (len_dist(_dist) == 1) & (len_dist(c_bol) == 1) \
#             & (isotropic or (len_dist(inclination) == 1))

#    if scalar:
#        F_per = _F_per
#        if not hasattr(F_per, 'unit'):
#            print('** WARNING ** assuming units of {} for flux'.format(flux_unit))
#        F_per *= flux_unit
#        _c_bol = c_bol
#    else:
#        F_per = value_to_dist(_F_per, nsamp=nsamp, unit=flux_unit)
#        dist = value_to_dist(dist, nsamp=nsamp, unit=u.kpc)
#        _c_bol = value_to_dist(c_bol, nsamp=nsamp)
#        # mdot_unit = u.g / u.cm ** 2 / u.s

    # if (nsamp > 1) and (not isotropic) and (len_dist(inclination) <= 1):
    #     inclination = iso_dist(nsamp, imin=imin, imax=imax)

    if isotropic:
        xi_b, xi_p = 1., 1.
        # print ('take into account the disk effect')
        if dip == True:
            logger.warning('isotropic distribution not correct for dipping sources')
    else:
        xi_b, xi_p = anisotropy(_inclination, model=model)

    # scale factors for opz and R are kept at the old values, even though the
    # best current estimates for these quantities have changed
    # prefactor here can be calculated as
    # (1e-9 * u.erg / u.cm ** 2 / u.s * (10 * u.kpc) ** 2 * 1.31 /
    #   (10 * u.km * c.G * 1.4 * c.M_sun)).to('g cm-2 s-1')
    # < Quantity 6713.24839048 g / (cm2 s) >
    # mdot_iso = (6.7e3 * (F_per / (1e-9 * flux_unit)) * (dist / (10 * u.kpc)) ** 2
    #             / (M / (1.4 * u.M_sun)) * (opz / 1.31) / (R / (10. * u.km))) * mdot_unit

    # mdot_iso_err = mdot_iso * np.sqrt((F_per_err / F_per) ** 2 + (2. * dist_err / dist) ** 2)
    # but now with full unit support, we just calculate in a more straightforward way

    Q_grav = const.G*M/R
    mdot = (F_per * _c_bol * dist * dist * opz * xi_p / (R**2 * Q_grav) ).to(mdot_unit)

    if isotropic or (nsamp == 1):

        return mdot #, mdot_iso_err

    else:

        if fulldist:

            # Return a dictionary with all the parameters you'll need

            return {'mdot': mdot, 'flux': F_per, 'c_bol': _c_bol, 'dist': dist, 'i': _inclination, 'xi_p': xi_p, 'model': model}
        else:

            # Return the median value and the (asymmetric) lower and upper errors

            mc = mdot.pdf_percentiles([50, 50-conf/2, 50+conf/2])
            return intvl_to_errors(mc)

# ------- --------- --------- --------- --------- --------- --------- ---------

def yign(_E_b, dist=None, nsamp=None, R=R_NS, opz=OPZ, Xbar=0.7, quadratic=False, old_relation=False,
         isotropic=False, inclination=None, imin=0.0, imax=IMAX_NDIP, dip=False,
         model='he16_a', conf=CONF, fulldist=False):
    """
    Calculate the burst column from the burst fluence, adapted initially
    from equation 4 of `Galloway et al. (2008, ApJS 179, 360) 
    <http://adsabs.harvard.edu/abs/2008ApJS..179..360G>`_

    :param _E_b: burst fluence
    :param _dist: distance ot bursting source
    :param R: neutron star radius
    :param opz: 1+z, surface redshift
    :param Xbar: mean H-fraction at ignition
    :param quadratic: flag to use the quadratic expression for Q_nuc
    :param old_relation: flag to use the old (incorrect) relation for Q_nuc
    :param isotropic: assume isotropic flux (or not)
    :param inclination: inclination value (or array; in degrees)
    :param imin: minimum inclination (degrees)
    :param imax: maximum inclination (degrees)
    :param dip: whether or not the source is a dipper
    :param model: model for anisotropy calculation
    :param nsamp: number of samples to generate
    :param conf: confidence interval for uncertainties; or
    :param fulldist: output the full distribution of calculated values
    :return: inferred ignition column values, either a scalar (if all the
	inputs are also scalars); a central value and confidence range; or
        (if ``fulldist=True``) a dictionary with the distributions of the
	ignition column and the input or adopted distributions of the
        other parameters
    """

    # yign_unit = u.g / u.cm ** 2
    yign_unit = 'g cm-2'

    # if a distance is not supplied, use a reasonable value, but flag it
    if dist is None:
        dist = 8.0*u.kpc
        logger.warning('assuming distance of {:3.1f}'.format(dist))

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    E_b, _dist, _inclination, _nsamp = homogenize_params( {'fluen': (_E_b, MINBAR_FLUEN_UNIT),
                                             'dist': (dist, u.kpc),
                                             'incl': (inclination, u.deg, isotropic, imin, imax)}, nsamp)

    # if no sample size has been passed, but we still need to generate samples
    # to account for the emission anisotropy, determine the array size here
    # if (nsamp is None) | (not isotropic):
    #     nsamp = NSAMP_DEF
    #     if (_nsamp > 3):
    #         nsamp = _nsamp

#    if hasattr(_E_b, 'unit'):
#        fluen_unit = _E_b.unit
#    else:
#        fluen_unit = MINBAR_FLUEN_UNIT

#    dist = _dist
#    if not hasattr(dist, 'unit'):
#        logger.warning('assuming units of {} for distance'.format(u.kpc))
#        dist *= u.kpc

#    scalar = (len_dist(_E_b) == 1) & (len_dist(_dist) == 1) \
#             & (isotropic or (len_dist(inclination) == 1))

#    if scalar:
#        E_b = _E_b
#        if not hasattr(E_b, 'unit'):
#            logger.warning('assuming units of {} for fluen'.format(fluen_unit))
#        E_b *= fluen_unit
#    else:
#        E_b = value_to_dist(_E_b, nsamp=nsamp, unit=fluen_unit)
#        dist = value_to_dist(dist, nsamp=nsamp, unit=u.kpc)

    if isotropic:
        xi_b, xi_p = 1., 1.
        # print ('take into account the disk effect')
        if dip == True:
            logger.warning('isotropic distribution not correct for dipping sources')
    else:
        # if (len_dist(inclination) <= 1):
        #     inclination = iso_dist(nsamp, imin=imin, imax=imax)

        xi_b, xi_p = anisotropy(_inclination, model=model)

    # old version with the prefactor, which can be calculated as
    # yign(1.,10.,R=10*u.km,opz=1.31, isotropic=True, Xbar=0.52)
    # the funny value for Xbar is to get a Q_nuc = 4.4 MeV/nucleon, but with the new
    # relation
    # yign_iso = (3e8 * (E_b / 1e-6 / fluen_unit) * (dist / 10. / u.kpc) ** 2
    #             / (Q_nuc(Xbar) / 4.4) * (opz / 1.31) / (R / (10.*u.km)) ** 2) * yign_unit
    # but now with full unit support, we just calculate in a more straightforward way

    qnuc = Q_nuc(Xbar, quadratic=quadratic, old_relation=old_relation) * u.MeV / const.m_p

    # print (len(E_b.distribution), len(dist.distribution), len(qnuc))
    # print (type(E_b), type(dist), type(xi_b), type(R), type(qnuc))
    yign = (E_b * dist**2 * opz * xi_b/ (R**2 * qnuc )).to(yign_unit)

    if len_dist(yign) == 1:
        return yign

    yc = yign.pdf_percentiles([50, 50 - conf / 2, 50 + conf / 2])

    if fulldist:

        # Return a dictionary with all the parameters you'll need

        return {'yign': yign, 'fluen': E_b, 'dist': dist, 'i': _inclination, 'xi_b': xi_b, 'model': model}
    else:

        # Return the median value and the (asymmetric) lower and upper errors

        return intvl_to_errors(yc)

# ------- --------- --------- --------- --------- --------- --------- ---------

def lum_to_flux(_lum, dist=None, c_bol=None, nsamp=None, burst=True, dip=False,
                   isotropic=False, inclination=None, imin=0.0, imax=IMAX_NDIP,
                   model='he16_a', conf=CONF, fulldist=False, plot=False):
    """
    This routine converts luminosity to flux, based on the provided distance
    Basically the inverse of the :py:meth:`concord.utils.luminosity` function

    :param lum: luminosity to convert to flux
    :param dist: distance to the source
    :param c_bol: bolometric correction to apply
    :param nsamp: number of samples
    :param burst: set to True for burst emission, to select the model anisotropy for bursts
    :param dip: set to True for a dipping source (deprecated)
    :param isotropic: set to True if isotropic value required
    :param inclination: system inclination or distribution thereof
    :param imin: minimum inclination for generated distribution
    :param imax: maximum inclination for generated distribution
    :param model: model string of He & Keek (2016), defaults to "A"
    :param conf: confidence % level for limits, ignored if fulldist=True
    :param fulldist: set to True to return the distributions of each parameter
    :param plot: plots the resulting distributions
    :return: inferred flux, either a scalar (if all the inputs are also
	scalars); a central value and confidence range; or (if
        ``fulldist=True``) a dictionary with the distributions of the
	flux and the input or adopted distributions of the intermediate
        values
    """

    # if a distance is not supplied, use a reasonable value, but flag it
    if dist is None:
        dist = 8.0*u.kpc
        logger.warning('assuming distance of {:3.1f}'.format(dist))

    # generate distributions where required, with correct units, and make sure
    # the lengths are consistent
    lum, _dist, _inclination, _c_bol, _nsamp = homogenize_params( {'lum': (_lum, u.erg/u.s),
                                                           'dist': (dist, u.kpc),
                                                           'incl': (inclination, u.deg, isotropic, imin, imax),
                                                           'c_bol': (c_bol, None)}, nsamp)

    if isotropic:
        xi_b, xi_p = 1., 1.
        # print ('take into account the disk effect')
        if dip == True:
            logger.warning('isotropic distribution not correct for dipping sources')
    else:
        xi_b, xi_p = anisotropy(_inclination, model=model)

        if not burst:
            # If you're calculating the persistent flux/luminosity, better use the right \xi
            logger.info('adopting anisotropy model {} for persistent emission'.format(model))
            xi_b = xi_p

    flux = (lum/(4 * np.pi * _dist ** 2 * xi_b * _c_bol)).to('erg cm-2 s-1')

    if len_dist(flux) == 1:

        # Just return the value; we only keep F_X as scalar if there's no
        # uncertainty provided
        return flux

    fc = flux.pdf_percentiles([50, 50 - conf / 2, 50 + conf / 2])

    if fulldist:

        # Return a dictionary with all the parameters you'll need
        return {'flux': flux, 'lum': _lum, 'c_bol': _c_bol, 'dist': _dist, 'i': _inclination, 'xi': xi_b, 'model': model}#, 'conf': conf}

    else:

        # Return the median value and the (asymmetric) lower and upper errors

        # return np.median(lum), np.percentile(lum, (50-conf/2, 50+conf/2)) * (u.erg/u.s) - np.median(lum)
        return intvl_to_errors(fc)
