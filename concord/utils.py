# Various utilities moved from burstclass.py
# Augmented 2019 Aug with routines from Inferring\ composition.ipynb
#
# def g(M,R,Newt=False,units='cm/s^2'):
# def opz(M,R):
# def calc_mr(g,opz):
# def solve_radius(M,R_Newt,eta=1e-6):
# def decode_LaTeX(string):
# def Q_nuc(Xbar, quadratic=False, old_relation=False, coeff=False)
# def hfrac(alpha, tdel, opz=1.259, zcno=0.02, old_relation=False, ...)
# def iso_dist(nsamp=1000, imin=0., imax=75.)
# def dist(F_pk, F_pk_err, nsamp=10000, X=0.0, empirical=False, ...)
# def mdot(F_per, F_per_err, dist, nsamp=10000, M_NS=1.4 * u.M_sun, R_NS=11.2 * u.km, ...)
# def yign(E_b, E_b_err, dist, R_NS=10., Xbar=0.7, opz=1.259, ...)
# def L_Edd(F_pk, F_pk_err, dist, dist_err=0.0, nsamp=10000, dip=False, ...)

import numpy as np
from math import *
from .diskmodel import *
import astropy.constants as const
import astropy.units as u
import re

# ------- --------- --------- --------- --------- --------- --------- ---------

def g(M,R,Newt=False,units='cm/s^2'):
    '''
    This function calculates the surface gravity given a mass M and radius R,
    using the Newtonian expression if the flag Newt is set to True

    The result is returned in units of cm/s^2 by default
    '''

    if Newt:
        return (const.G*M/R**2).to(units)
    else:
        #        return const.G*M/(R**2*sqrt(1.-2.*const.G*M/(const.c**2*R))).to(units)
        return opz(M,R)*g(M,R,Newt=True).to(units)

# ------- --------- --------- --------- --------- --------- --------- ---------

def opz(M,R):
    '''
    This function calculates the gravitational redshift 1+z
    '''

    return 1./sqrt(1.-2.*const.G*M/(const.c**2*R))

# ------- --------- --------- --------- --------- --------- --------- ---------

def calc_mr(g,opz):
    ''''
    this function calculates neutron star mass and radius given a surface
    gravity and redshift
    '''

    # First some checks

    if hasattr(opz,'unit'):
        assert (opz.unit == '')
    try:
        test = g.to('cm / s2')
    except ValueError:
        print ("Incorrect units for surface gravity")
        return -1, -1

    # Now calculate the mass and radius and convert to cgs units

    R_NS = (const.c**2*(opz**2-1)/(2.*g*opz)).to(u.cm)
    M_NS = (g*R_NS**2/(const.G*opz)).to(u.g)

    return M_NS, R_NS

# ------- --------- --------- --------- --------- --------- --------- ---------

def solve_radius(M,R_Newt,eta=1e-6):
    '''
    This routine determines the GR radius given the NS mass and Newtonian
    radius, assuming the GR and Newtonian masses are identical

    Solving is tricky so we just use an iterative approach
    '''

    R_NS = R_Newt	# trial
    while (abs(g(M,R_NS)-g(M,R_Newt,Newt=True))/g(M,R_NS) > eta):
        R_NS = R_Newt*sqrt(opz(M,R_NS))

    return R_NS

# ------- --------- --------- --------- --------- --------- --------- ---------

def decode_LaTeX(string):
    '''
    This function converts a LaTeX numerical value (with error) to one
    or more floating point values
    '''

    assert (type(string) == str) or (type(string) == np.str_)

    # Look for the start of a LaTeX numerical expression
    # We no longer explicitly look for the $ sign, as this pattern can match a
    # general numeric string. We also include the mantissa as optional

    val_match = re.search('([0-9]+(\.[0-9]+)?)',string)

    # If not found, presumably you just have a numerical expression as string

    if val_match == None:
        return float(string), None

    # Otherwise, convert the first part to a float, and look for the error

    val = float(val_match.group(1))
    err_match = re.search('pm *([0-9]+\.[0-9]+)',string)

    if err_match == None:
        return val, None
    else:
        return val, float(err_match.group(1))

# ------- --------- --------- --------- --------- --------- --------- ---------

def Q_nuc(Xbar, quadratic=False, old_relation=False, coeff=False):
    '''
    This function implements the approximation to the nuclear energy generation rate
    Q_nuc given the mean hydrogen fraction in the fuel layer, Xbar, determined by
    Goodwin et al. 2019 (ApJ 870, #64). The value returned is in units of MeV/nucleon:
    q = Q_nuc(0.73)
    5.35517578
    There are three versions of the approximation, which can be selected by using the
    quadratic and old_relation flags; by default the most precise (and recent) version
    is used:
    q = Q_nuc(0.73, quadratic=True)
    5.758715
    q = Q_nuc(0.73, old_relation=True)
    4.52
    If you want the coefficients, rather than the actual Q_nuc value, you can use the
    coeff flag (in which case the first argument is ignored):
    q_0, q_1 = Q_nuc(0.0,quadratic=True,coeff=True)
    [1.3455, 6.0455]
    '''

    q = [1.3050, 6.9511, -1.9218]

    if quadratic:
        q = [1.3455, 6.0455]

    if old_relation:
        q = [1.6, 4.0]

    if coeff:
        return q

    return sum([q[i] * Xbar ** i for i in range(len(q))])

# ------- --------- --------- --------- --------- --------- --------- ---------

def hfrac(alpha, tdel, opz=1.259, zcno=0.02, old_relation=False,
          imin=0.0, imax=75., nsamp=1000, isotropic=False, inclination=None,
          debug=False):
    '''
    This routine estimates the h-fraction at ignition, based on the burst properties
    There are three main modes based on the value of the isotropic flag:
    isotropic=True  - assumes both the burst and persistent emission is isotropic (xi_b=xi_p=1)
    isotropic=False - incorporates anisotropy, by drawing a set of inclination values and
                        calculating the H-fraction for each value. Inclinations are uniform on
                        the sphere (up to a maximum value of i=72 deg, based on the absence
                        of dips) and the value returned is the mean and 1-sigma error
    inclination=<i> - calculate for a specific value of inclination

    Usage:
    xbar, x_0 = hfrac(alpha,tdel[,opz=<opz>][,zcno=<zcno>] etc.)
    '''

    # some constants

    xmax = 0.77  # no longer used
    # imax = 72.*np.pi/180.
    # imax = 75. # [deg] maximum inclination for a non-dipping source

    # nsamp=1000 # number of samples for anisotropy distribution

    vector_alpha = not (np.shape(alpha) == ())
    #    print (vector_alpha,alpha,type(alpha))
    if vector_alpha:
        nsamp = len(alpha)
        if debug:
            print("hfrac: adopting {} samples to match size of alpha array".format(nsamp))

    if debug and (not vector_alpha):
        print('hfrac: alpha = {}, tdel= {}, opz = {}, zcno = {}, isotropic = {}'.format(
            alpha, tdel, opz, zcno, isotropic))

    # The following parameters define the relation for Q_nuc = q_0 + q_1*hbar, where hbar is the
    # average hydrogen fraction in the fuel column at ignition

    q_0, q_1 = Q_nuc(0.0, old_relation=old_relation, quadratic=True, coeff=True)

    # Does this need to change if the Q_nuc coefficients also change? YES
    # alpha_0 = const.c**2/(6.6*u.MeV/const.m_p).to("m**2/s**2")
    alpha_0 = const.c ** 2 / (q_1 * u.MeV / const.m_p).to("m**2/s**2")

    # This parameter sets the prefactor for the time to burn all H via hot-CNO; see
    # Lampe et al. (2016, ApJ 819, 46)

    tpref = 9.8 * u.hr

    # Here we allow an option to use the old version if need be; this is already passed to the
    # Q_nuc function, so we'll get the old parameters from that if need be

    if old_relation:
        tpref = 11. * u.hr

    # Set the inclination parameters, by default for isotropy

    xi_b = 1.
    xi_p = 1.
    #    print("isotropic, inclination", isotropic, inclination)
    if (not isotropic) and (inclination == None):

        # Here you can set up a uniform distribution of inclinations up to some maximum, and calculate
        # the statistical properties of the resulting values

        #        print ("Not yet implemented")

        # idist = np.arccos(np.cos(imax)+np.random.random(nsamp)*(1-np.cos(imax)))*180./np.pi*u.deg
        idist = iso_dist(nsamp, imax=imax)

        xbar = np.zeros(nsamp)
        x_0 = np.zeros(nsamp)
        #        print (vector_alpha)
        for j, i in enumerate(idist):
            if vector_alpha:
                #                print ('selecting alpha by element')
                xbar[j], x_0[j], dummy = hfrac(alpha[j], tdel[j], opz=opz, zcno=zcno, isotropic=isotropic,
                                               old_relation=old_relation, inclination=i, debug=debug)
            else:
                xbar[j], x_0[j], dummy = hfrac(alpha, tdel, opz=opz, zcno=zcno, isotropic=isotropic,
                                               old_relation=old_relation, inclination=i, debug=debug)

        return xbar, x_0, idist

    elif (not isotropic) and (inclination != None):
        xi_b, xi_p = anisotropy(inclination)

    # I think this part only gets called for scalar values, but you should check here

    assert np.shape(opz) == np.shape(alpha) == np.shape(inclination) == ()

    # And finally calculate the hydrogen fractions
    # Early on I was limiting this by xmax, but we want to do that in the
    # calling routine, to make it clear where we exceed the likely maximum

    #    xbar = min([xmax,(opz-1.)*alpha_0/alpha*(xi_b/xi_p)-q_0/q_1])
    xbar = (opz - 1.) * alpha_0 / alpha * (xi_b / xi_p) - q_0 / q_1
    x_0 = -1  # dummy value

    #    x_0 = min([xmax,xbar+0.35*(tdel/(opz*9.8*u.hr))*(zcno/0.02)])
    #    for i in range(10):
    if xbar > 0.0:

        # Also here now we distinguish between the two possible cases, the first where there
        # remains hydrogen at the base, and the second where it is exhausted
        # This division is quantified via the f_burn fraction, which initially at least we
        # must estimate

        t_CNO = tpref * (xbar / 0.7) / (zcno / 0.02)
        f_burn = tdel / (opz * t_CNO)
        f_burn_prev = 1. / f_burn  # dummy value first off to ensure the loop gets called

        # Loop here to make sure we have a consistent choice

        while (1. - f_burn) * (1. - f_burn_prev) < 0.:

            if (f_burn <= 1.):
                # still hydrogen at the base
                x_0 = xbar + 0.35 * (tdel / (opz * tpref)) * (zcno / 0.02)
            else:
                # hydrogen exhausted at the base
                x_0 = np.sqrt(1.4 * xbar * tdel / (opz * tpref) * (zcno / 0.02))

            f_burn_prev = f_burn
            t_CNO = tpref * (x_0 / 0.7) / (zcno / 0.02)
            f_burn = tdel / (opz * t_CNO)

            if debug:
                print(xbar, x_0, t_CNO, xi_b, xi_p)
                print('flipping f_burn {} -> {}'.format(f_burn_prev, f_burn))

    #        print (i,xbar,t_CNO,f_burn,x_0,xi_b,xi_p)

    #    return xbar, xbar+0.35*(tdel/(opz*tpref))*(zcno/0.02), inclination
    return xbar, x_0, inclination

# ------- --------- --------- --------- --------- --------- --------- ---------

def iso_dist(nsamp=1000, imin=0., imax=75.):
    '''
    Routine to generate an isotropic distribution of inclinations (angle from the system
    rotation axis to the line of sight) from imin up to some maximum value, defaulting to
    75 degrees. This value corresponds to the likely maximum possible for a non-dipping
    source.
    '''

    if imin < 0. or imin > imax or imax > 90.:
        print("** ERROR ** imin, imax must be in the range [0,90] degrees")
        return None

    # Convert to radians
    cos_imin = np.cos(imin * np.pi / 180.)
    cos_imax = np.cos(imax * np.pi / 180.)
    print(cos_imin, cos_imax)
    # uniform in cos(i) up to i=imax

    return np.arccos(cos_imax + np.random.random(nsamp) * (cos_imin - cos_imax)) * 180. / np.pi * u.deg

# ------- --------- --------- --------- --------- --------- --------- ---------

def dist(F_pk, F_pk_err, nsamp=10000, X=0.0, empirical=False,
         M_NS=1.4 * u.M_sun, opz=1.259, T_e=0.0, dip=False,
         isotropic=False, imin=0., imax=75., fulldist=False, plot=False):
    '''
    This routine estimates the distance to the source, based on the measured peak flux
    of a radius-expansion burst. Two main options are available;
    empirical=False (default), in which case the Eddington flux is calculated as equation
    7 from Galloway et al. (2008, ApJS 179, 360); or
    empirical=True, in which case the estimate of Kuulkers et al. 2003 (A&A 399, 663) is used
    The default isotropic=False option also takes into account the likely effect of
    anisotropic burst emission, based on the models provided by concord
    '''

    alpha_T = 2.2e-9  # K^-1

    flux_unit = F_pk.unit

    # Choose the Eddington flux value to compare the distance against

    if empirical:

        # Kuulkers et al. 2003, A&A 399, 663

        L_Edd = 3.79e38 * u.erg / u.s
        L_Edd_err = 0.15e38 * u.erg / u.s
    else:

        # Galloway et al. 2008, ApJS 179, 360

        L_Edd = 2.7e38 * ((M_NS / (1.4 * u.M_sun)) * (1 + (alpha_T * T_e) ** 0.86) / (1 + X)
                          / (opz / 1.31)) * u.erg / u.s
        L_Edd_err = 0. * u.erg / u.s

    # Treat the errors differently if we're doing the isotropic calculation

    if isotropic:
        dist_iso = np.sqrt(L_Edd / (4 * np.pi * F_pk)).to('kpc')

        # Simple estimate of the error

        dist_iso_err = dist_iso * 0.5 * np.sqrt((L_Edd_err / L_Edd) ** 2 + (F_pk_err / F_pk) ** 2)

        return dist_iso, dist_iso_err

    else:
        _F_pk = np.random.normal(0., 1., size=nsamp) * F_pk_err + F_pk
        dist_iso = np.sqrt(L_Edd / (4 * np.pi * _F_pk)).to('kpc')

        # print ('take into account the disk effect')
        if dip == True:
            print('** WARNING ** isotropic distribution not correct for dipping sources')
        idist = iso_dist(nsamp, imin=imin, imax=imax)
        xi_b, xi_p = anisotropy(idist)

        dist = dist_iso / np.sqrt(xi_b)

        if plot:
            # Do a simple plot of the distance distribution

            plt.hist(dist / u.kpc, bins=50, density=True)
            plt.xlabel('Distance (kpc)')
            plt.axvline(np.median(dist).value, color='g')
            plt.axvline(np.percentile(dist, 16), color='g', ls='--')
            plt.axvline(np.percentile(dist, 84), color='g', ls='--')
            # plt.show()
        # else:
        # print (plot)

        if fulldist:

            # Return a dictionary with all the parameters you'll need

            return {'dist': dist, 'i': idist, 'xi_b': xi_b}
        else:

            # Return the median value and the (asymmetric) lower and upper errors

            return np.median(dist), np.percentile(dist, (16, 84)) * u.kpc - np.median(dist)

# ------- --------- --------- --------- --------- --------- --------- ---------

def mdot(F_per, F_per_err, dist, nsamp=10000, M_NS=1.4 * u.M_sun, R_NS=11.2 * u.km, opz=1.259,
         isotropic=False, inclination=None, fulldist=False):
    '''
    Routine to estimate the mdot given a (bolometric) persistent flux, distance, and
    inclination
    '''

    flux_unit = F_per.unit
    dist_err = 0.0 * u.kpc
    mdot_unit = u.g / u.cm ** 2 / u.s

    vector_dist = not (np.shape(dist) == ())
    if vector_dist:
        nsamp = len(dist)

    if isotropic:

        # scale factors for opz and R_NS are kept at the old values, even though the
        # best current estimates for these quantities have changed
        mdot_iso = (6.7e3 * (F_per / 1e-9 / flux_unit) * (dist / 10 / u.kpc) ** 2
                    / (M_NS / (1.4 * u.M_sun)) * (opz / 1.31) / (R_NS / (10. * u.km))) * mdot_unit

        mdot_iso_err = mdot_iso * np.sqrt((F_per_err / F_per) ** 2 + (2. * dist_err / dist) ** 2)

        return mdot_iso, mdot_iso_err

    else:
        # Generate some random iterates for F_per consistent with the central value
        # and error
        _F_per = np.random.normal(0., 1., size=nsamp) * F_per_err + F_per

        mdot_iso = (6.7e3 * (_F_per / 1e-9 / flux_unit) * (dist / 10 / u.kpc) ** 2
                    / (M_NS / (1.4 * u.M_sun)) * (opz / 1.31) / (R_NS / (10. * u.km))) * mdot_unit

        # print ("taking into account anisotropy here...")
        if inclination is not None:
            if np.shape(dist) != np.shape(inclination):
                print('** ERROR ** length of distance and inclination arrays must agree')

        xi_b, xi_p = anisotropy(inclination)

        mdot = mdot_iso * xi_p

    if vector_dist:
        if fulldist:

            # Return a dictionary with all the parameters you'll need

            return {'mdot': mdot, 'dist': dist, 'i': inclination, 'xi_p': xi_p}
        else:

            # Return the median value and the (asymmetric) lower and upper errors

            return np.median(mdot), np.percentile(mdot, (16, 84)) * mdot_unit - np.median(mdot)

# ------- --------- --------- --------- --------- --------- --------- ---------

def yign(E_b, E_b_err, dist, R_NS=10., Xbar=0.7, opz=1.259,
         isotropic=False, inclination=None, fulldist=False):
    '''
    Calculate the burst column from the fluence
    '''

    fluen_unit = E_b.unit
    yign_unit = u.g / u.cm ** 2

    vector_dist = not (np.shape(dist) == ())
    if vector_dist:
        nsamp = len(dist)

    if isotropic:
        yign_iso = (3e8 * (E_b / 1e-6 / fluen_unit) * (dist / 10. / u.kpc) ** 2
                    / (Q_nuc(Xbar) / 4.4) * (opz / 1.31) / (R_NS / 10.) ** 2) * yign_unit

        yign_iso_err = yign_iso * np.sqrt((E_b_err / E_b) ** 2)  # +...

        return yign_iso, yign_iso_err

    else:
        # Generate some random iterates for E_b consistent with the central value
        # and error

        _E_b = np.random.normal(0., 1., size=nsamp) * E_b_err + E_b

        yign_iso = (3e8 * (_E_b / 1e-6 / fluen_unit) * (dist / 10. / u.kpc) ** 2
                    / (Q_nuc(Xbar) / 4.4) * (opz / 1.31) / (R_NS / 10.) ** 2) * yign_unit

        # print ("taking into account anisotropy here...")
        if inclination is not None:
            if np.shape(dist) != np.shape(inclination):
                print('** ERROR ** length of distance and inclination arrays must agree')

        xi_b, xi_p = anisotropy(inclination)

        yign = yign_iso * xi_b

    if vector_dist:
        if fulldist:

            # Return a dictionary with all the parameters you'll need

            return {'yign': yign, 'dist': dist, 'i': inclination, 'xi_b': xi_b}
        else:

            # Return the median value and the (asymmetric) lower and upper errors

            return np.median(yign), np.percentile(yign, (16, 84)) * yign_unit - np.median(yign)

# ------- --------- --------- --------- --------- --------- --------- ---------

def L_Edd(F_pk, F_pk_err, dist, dist_err=0.0, nsamp=10000, dip=False,
          isotropic=False, imin=0., imax=75., fulldist=False):
    '''
    This routine estimates the Eddington luminosity of the source, based on the measured
    peak flux of a radius-expansion burst and the distance (or some distribution thereof.
    The default isotropic=False option also takes into account the likely effect of
    anisotropic burst emission, based on the models provided by concord
    '''

    flux_unit = F_pk.unit

    vector_dist = not (np.shape(dist) == ())
    if vector_dist:
        nsamp = len(dist)

    if isotropic:
        L_Edd_iso = (4 * np.pi * dist ** 2 * F_pk).to('erg s-1')

        # Simple estimate of the error

        L_Edd_iso_err = L_Edd_iso * np.sqrt((F_pk_err / F_pk) ** 2 + (2. * dist_err / dist) ** 2)

        return L_Edd_iso, L_Edd_iso_err

    else:
        _F_pk = np.random.normal(0., 1., size=nsamp) * F_pk_err + F_pk
        if vector_dist:
            _dist = dist
        else:
            _dist = np.random.normal(0., 1., size=nsamp) * dist_err + dist

        L_Edd_iso = (4 * np.pi * _dist ** 2 * _F_pk).to('erg s-1')

        # print ('take into account the disk effect')
        if (dip == True) & (imax < 75.):
            print('** WARNING ** isotropic distribution not correct for dipping sources')

        idist = iso_dist(nsamp, imin=imin, imax=imax)
        xi_b, xi_p = anisotropy(idist)

        L_Edd = L_Edd_iso * xi_b

    if fulldist:

        # Return a dictionary with all the parameters you'll need

        return {'L_Edd': L_Edd, 'dist': _dist, 'i': idist, 'xi_b': xi_b}
    else:

        # Return the median value and the (asymmetric) lower and upper errors

        return np.median(L_Edd), np.percentile(L_Edd, (16, 84)) * u.erg / u.s - np.median(L_Edd)