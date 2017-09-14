# Burst class for MINBAR
#
# This code is a preliminary attempt to develop a burst class for MINBAR,
# including model-observation comparisons.
#
# Here are the classes and functions making up this module:
#
# class Lightcurve(object):
# class ObservedBurst(Lightcurve):
# class KeplerBurst(Lightcurve):
#
# def decode_LaTeX(string):
# def modelFunc(p,obs,model):
# def lnprior(theta):
# def apply_units(params,units = (u.kpc, u.degree, None, u.s)):
# def lhoodClass(params,obs,model):
# def plot_comparison(obs,models,param=None,sampler=None,ibest=None):

import numpy as np
from math import *

import astropy.units as u
import astropy.constants as const
import astropy.io.ascii as ascii
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import gridspec
import re
from scipy.interpolate import interp1d
from astroquery.vizier import Vizier
from chainconsumer import ChainConsumer

from anisotropy import *

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

#    val_match = re.search('\$ *([0-9]+\.[0-9]+)',string)
    val_match = re.search('([0-9]+(\.[0-9]+)?)',string)

# If not found, presumably you just have a numerical expression as string

    if val_match == None:
#        print (val_match)
        return float(string), None

# Otherwise, convert the first part to a float, and look for the error

    val = float(val_match.group(1))
    err_match = re.search('pm *([0-9]+\.[0-9]+)',string)

    if err_match == None:
        return val, None
    else:
        return val, float(err_match.group(1))

# ------- --------- --------- --------- --------- --------- --------- ---------

def modelFunc(p,obs,model):
    '''
    This function performs the stretching and rescaling of the (model
    predicted) burst lightcurve based on the input parameter set
    The calculations are based on those in Appendix B of Lampe et al.
    2016 (ApJ 819:46)
    Parameter array is (for now) a tuple with the appropriate units:
    ( distance, inclination, redshift, time offset )
    '''

    dist, inclination, opz, t_off = p

# Use the anisotropy function to calculate the anisotropy factors given
# the inclination

    xi_b, xi_p = anisotropy(inclination)

# First interpolate the model values onto the observed grid
# This step defines the interpolation function, using a shifted and rescaled
# model time to account for time dilation in the NS surface frame

#    fInterp = interp1d(model.time,model.lumin,bounds_error=False,
    fInterp = interp1d(opz*(model.time-t_off),model.lumin,bounds_error=False,
                      fill_value = min(model.lumin))

# Then we return the predicted model flux, interpolated onto the observed
# burst time bins. We shift the observed times to the middle of the bin, using
# the timepixr attribute.
# We also account for the distance, and the anisotropy parameter, using the
# convention of Fujimoto et al. 1988, i.e.
#   L_b = 4\pi d^2 \xi_b F_b

#    return p[1]*fInterp(p[2]*(obs.time-p[3]))*model.lumin.unit/(4.*pi*p[0].to('cm')**2)
    return ( (model.xi/opz)**2 * fInterp(obs.time+(0.5-obs.timepixr)*obs.dt)*model.lumin.unit
            / (4.*pi*dist.to('cm')**2) / xi_b )

# ------- --------- --------- --------- --------- --------- --------- ---------

# The basic object, which has attributes like time (array),
# flux/luminosity etc. and methods including plot

class Lightcurve(object):
    """Test class for a burst lightcurve"""

    def __init__(self, *args, **kwargs):
        """initialise the object by assigning named entities from the kwargs"""

# Initialise the various attributes, where present. We expect to have at
# least one of flux or lumin (and the related uncertainty)
# Units are handled by the parent classes ObservedBurst and ModelBurst

        self.time = kwargs.get('time',None)
        self.timepixr = kwargs.get('timepixr',0.0)
        self.dt = kwargs.get('dt',None)
        if kwargs.get('flux',None):
            self.flux = kwargs.get('flux',None)
            self.flux_err = kwargs.get('flux_err',None)

        self.lumin = kwargs.get('lumin',None)
        self.lumin_err = kwargs.get('lumin_err',None)

# Later we can make this plot method more elaborate
# where argument for step is appropriate for timepixr=0.0

    def plot(self):

#        plt.plot(self.time,self.flux)

        assert self.timepixr == 0.0
        plt.step(self.time,self.flux,where='post',
#		label=self.filename.replace('_',r'\_'))
		label=self.filename)
        plt.errorbar(self.time.value+(0.5-self.timepixr)*self.dt.value,
            self.flux.value, yerr=self.flux_err.value,fmt='b.')
        plt.xlabel("Time ({0.unit:latex_inline})".format(self.time))
        plt.ylabel("Flux ({0.unit:latex_inline})".format(self.flux))

# ------- --------- --------- --------- --------- --------- --------- ---------

# This class is for observed bursts; we define a compare method to match
# with models

class ObservedBurst(Lightcurve):
    '''
    Observed burst class. Apart from the lightcurve (which is
    defined with time, flux, and flux_err columns), the additional
    attributes set are:
    filename - source file name
    tdel, tdel_err - recurrence time and error (hr)
    comments - ASCII file header text
    table_file - LaTeX file of table 2 from Galloway et al. (2017)
    table - contents of table 2
    row - entry in the table corresponding to this burst
    fper, fper_err - persistent flux level
    cbol - bolometric correction
    mdot - accretion rate
    fluen - burst fluence
    F_pk - burst peak flux
    alpha - alpha value
    '''

    def __init__(self, filename, path=None, **kwargs):

# For now, this is restricted to the "reference" bursts, which have a
# format like this:

#	.
#	.
#	.
#
# Columns are:
# 'Time [s]' 'dt [s]' 'flux [10^-9 erg/cm^2/s]' 'flux error [10^-9
# erg/cm^2/s]' 'blackbody temperature kT [keV]' 'kT error [keV]'
# 'blackbody normalisation K_bb [(km/d_10kpc)^2]' 'K_bb error
# [(km/d_10kpc)^2]' chi-sq
#     -1.750  0.500   1.63    0.054   1.865  0.108   13.891   4.793  0.917
#     -1.250  0.500   2.88    1.005   1.862  0.246   22.220   4.443  1.034
#     -0.750  0.500   4.38    1.107   1.902  0.093   30.247   2.943  1.089
#     -0.250  0.500   6.57    0.463   1.909  0.080   46.936   6.969  0.849

        if path == None:
            path = '.'
        self.path = path
        self.filename = filename

        d=ascii.read(path+'/'+self.filename)

# Now we define a Lightcurve instance, using the columns from the file

        Lightcurve.__init__(self, time=d['col1']*u.s, dt=d['col2']*u.s,
                            flux=d['col3']*1e-9*u.erg/u.cm**2/u.s,
                            flux_err=d['col4']*1e-9*u.erg/u.cm**2/u.s)

# additional (presently redundant) quantities, held over from the older version
#        self.kT = d['col5']*u.keV
#        self.kT_err = d['col6']*u.keV
# This is not ideal, but gives the right behaviour, in principle
#        self.K_bb = d['col7']*u.km**2/(10.*u.parsec)**2
#        self.k_bb_err = d['col8']*u.km**2/(10.*u.parsec)**2
#        self.chisqr = d['col9']

# In principle we can parse a bunch of additional information from the headers

        self.comments = d.meta['comments']

# Here the recurrence time; looking for a string like
#   Average recurrence time is 3.350 +/- 0.04 hr
# (not present in all the files; only the 1826-24 ones)
# This has now been replaced by the more general table search below
#        for line in self.comments:
#            m = re.match('Average recurrence time is ([0-9]+\.[0-9]+) \+/- ([0-9]\.[0-9]+)',line)
#            if m != None:
#                self.tdel = float(m.group(1))*u.hr
#                self.tdel_err = float(m.group(2))*u.hr
##        print (tdel, tdel_err)

#        self.table_file = '/Users/duncan/burst/reference/doc/table2.tex'
        self.table_file = '/home/zacpetej/projects/codes/concord/table2.tex'
        self.table = Table.read(self.table_file)

# Below we associate each epoch with a file

        file=['gs1826-24_5.14h.dat',
              'gs1826-24_4.177h.dat',
              'gs1826-24_3.530h.dat',
              'saxj1808.4-3658_16.55h.dat',
              'saxj1808.4-3658_21.10h.dat',
              'saxj1808.4-3658_29.82h.dat',
              '4u1820-303_2.681h.dat',
              '4u1820-303_1.892h.dat',
              '4u1636-536_superburst.dat']
        self.table['file'] = file

# Find which of these (if any) are the one you're reading

        for i, lcfile in enumerate(self.table['file']):
            m = re.search(lcfile,self.filename)
            if m != None:
                self.row=i
#                print (i, lcfile, filename)

        if hasattr(self,'row'):

            tdel, tdel_err = decode_LaTeX(self.table['$\Delta t$ (hr)'][self.row])
            self.tdel = tdel*u.hr
            if tdel_err != None:
                self.tdel_err = tdel_err*u.hr
            else:

# If no tdel error is supplied by the file (e.g. for the later bursts from
# SAX J1808.4-3658), we set a nominal value corresponding to 1 s (typical
# RXTE time resolution) here

                self.tdel_err = 1./3600.*u.hr

# Decode the other table parameters

            label = ['fper','cbol','mdot','fluen','F_pk','alpha']
            unit = [1e-9*u.erg/u.cm**2/u.s, 1.,1.75e-8*const.M_sun/u.yr,
                    1e-6*u.erg/u.cm**2/u.s,
                    1e-9*u.erg/u.cm**2/u.s,1.]
            for i, column in enumerate(self.table.columns[4:10]):
#            print (i, column, label[i], self.table[column][row], type(self.table[column][row]))
                if ((type(self.table[column][self.row]) == np.str_)):
#    or (type(self.table[column][row]) == np.str_)):

# Here we convert the table entry to a value. We have a couple of options
# here: raw value, range (separated by "--"), or LaTeX expression

                    range_match = re.search('([0-9]+\.[0-9]+)--([0-9]+\.[0-9]+)',
			    self.table[column][self.row])
                    if range_match:

                        lo = float(range_match.group(1))
                        hi = float(range_match.group(2))
                        val = 0.5*(lo+hi)
                        val_err = 0.5*abs(hi-lo)

                    else:
                        val, val_err = decode_LaTeX(self.table[column][self.row])

# Now set the appropriate attribute

                    setattr(self,label[i],val*unit[i])
                    if val_err != None:
#                    print (column, label[i]+'_err',val_err)
                        setattr(self,label[i]+'_err',val_err*unit[i])
                else:
                    setattr(self,label[i],self.table[column][self.row]*unit[i])

# End block for adding attributes from the file. Below you can use the
# additional arguments on init to set or override attributes

        if kwargs != None:

            for key in kwargs:
#                print (key, kwargs[key])
                if (key == 'tdel') | (key == 'tdel_err'):
                    setattr(self,key,float(kwargs[key])*u.hr)
                else:
                    setattr(self,key,kwargs[key])

# This is the key method for running the mcmc; it can be used to plot the
# observations with the models rescaled by the appropriate parameters, and
# also returns a likelihood value

    def compare(self, mburst, param = [6.1*u.kpc,60.*u.degree,1.,+8.*u.s],
		breakdown = False, plot = False, subplot = True,
                debug = False):

        dist, inclination, opz, t_off = param

        if inclination > 90.*u.degree or inclination < 0.*u.degree:
            return 0.


        xi_b, xi_p = anisotropy(inclination)

# Here we calculate the equivalent mass and radius given the redshift and
# radius. Since we allow the redshift to vary, but the model is calculated
# at a fixed surface gravity, we need to adjust one (or both) of M_NS and
# R_NS to obtain a self-consistent set of parameters.
# Because many equations of state have roughly constant radius over a
# range of masses, we choose to keep R_NS constant and to vary M_NS

#        _t = (mburst.g.to(u.cm/u.s**2)*mburst.R_NS.to(u.cm)
#		/const.c.to(u.cm/u.s)**2)
#        M_NS = (mburst.g*mburst.R_NS**2/const.G * (-_t + sqrt(_t+1)))
        M_NS = mburst.g*mburst.R_NS**2/(const.G*opz)
        M_NS = M_NS.to(u.kg)
        if debug:
            print ('Inferred mass = {:.4f} M_sun'.format(M_NS/const.M_sun))
        Q_grav = const.G*M_NS/mburst.R_NS

# These parameters give the relative weight to the tdel and persistent
# flux for the likelihood. Since you have many more points in the
# lightcurve, you may want to weight these greater than one so that the
# MCMC code will try to match those preferentially

#        tdelwt=2.5e3
        tdelwt=1.0
        fluxwt=1.0

# Calculate the rescaled model flux with the passed parameters

        model = modelFunc(param,self,mburst)
        assert model.unit == self.flux.unit == self.flux_err.unit

# can check here if the object to compare is actually a model burst
#        print (type(mburst))

# Plot the observed burst

        if plot:

# Now do a more complex plot with a subplot
# See http://matplotlib.org/users/gridspec.html for documentation

            fig = plt.figure()
            gs = gridspec.GridSpec(4, 3)
            ax1 = fig.add_subplot(gs[0:3,:])

            self.plot()

# overplot the rescaled model burst

##        plt.plot(mburst.time, mburst.flux(dist))
#            plt.plot(self.time+(0.5-self.timepixr)*self.dt, model,'r-',
## This removes the problems with the underscore, but now introduces an
## extra backslash... argh
##		label=mburst.filename.replace('_',r'\_'))
#		label=mburst.filename)

            ax1.plot(self.time+(0.5-self.timepixr)*self.dt, model,'r-',
		label=mburst.filename)

            if subplot:

# Show the recurrence time comparison in the subplot; see
# http://matplotlib.org/examples/pylab_examples/axes_demo.html for
# illustrative example

                a = plt.axes([.55, .5, .3, .3], facecolor='y')
#                print (self.tdel,self.tdel_err)
                a.errorbar([0.95], self.tdel.value,
                       yerr=self.tdel_err.value, fmt='o')
                a.errorbar([1.05], mburst.tdel.value*opz,
                       yerr=mburst.tdel_err.value*opz, fmt='ro')
                plt.ylabel('$\Delta t$ (hr)')

#            plt.plot(self.time,model,'.')
                plt.xlim(0.8,1.2)
                plt.xticks([])

            else:

# This bit doesn't work because it wants to parse the labels as LaTeX,
# which works for some strings (byt not for bytes). Don't know why the
# strings sometimes end up as byte arrays... grr.

                plt.legend()
#                print (mburst.filename)

# Residual panel

            ax2 = fig.add_subplot(gs[3,:])
#            ax2.plot(self.time+(0.5-self.timepixr)*self.dt, model-self.flux)
            plt.errorbar(self.time.value+(0.5-self.timepixr)*self.dt.value,
		self.flux.value-model.value,
		yerr=self.flux_err.value,fmt='b.')
            ax2.axhline(0.0, linestyle='--', color='k')
            gs.update(hspace=0.0)

# Here we assemble an array with the likelihood components from each
# parameter, for accounting purposes
# By convention we calculate the observed parameter (from the model) and
# then do the comparison
# Should probably incorporate these calculations into the class, so you
# can just refer to them as an attribute

        lhood_cpt = np.array([])

# persistent flux (see lampe16, eq. 8, noting that our mdot is averaged
# over the neutron star surface):

        fper_sig2 = 1.0/(self.fper_err.value**2)
        fper_pred = ( mburst.mdot*Q_grav/
               (4.*pi*opz*dist**2*xi_p*self.cbol) )
        fper_pred = fper_pred.to(u.erg/u.cm**2/u.s)
        lhood_cpt = np.append(lhood_cpt, -fluxwt*(
               (self.fper.value-fper_pred.value)**2*fper_sig2
               +np.log(2.*pi/fper_sig2) ) )

# recurrence time

        tdel_sig2 = 1.0/(self.tdel_err.value**2+(mburst.tdel_err.value*opz)**2)
        lhood_cpt = np.append(lhood_cpt, -tdelwt*(
               (self.tdel.value-mburst.tdel.value*opz)**2*tdel_sig2
               +np.log(2.*pi/tdel_sig2) ) )

# lightcurve

        inv_sigma2 = 1.0/(self.flux_err.value**2)
        lhood_cpt = np.append(lhood_cpt,
        	-0.5 * np.sum( (model.value-self.flux.value)**2*inv_sigma2
                +np.log(2.0*pi/inv_sigma2) ) )
# troubleshooting the likelihood comparison
#        print ( (model.value-self.flux.value)**2*inv_sigma2
#                - np.log(2.0*pi*inv_sigma2) )

# Printing the values to test

        if debug:
            cl=0.0
            for i in range(len(self.time)):
                _lhood = -0.5*((model[i].value-self.flux[i].value)**2*inv_sigma2[i]
                    +np.log(2.0*pi/inv_sigma2[i]))
                cl += _lhood
                print ('{:6.2f} {:.4g} {:.4g} {:.4g} {:8.3f} {:8.3f}'.format(self.time[i],self.flux[i].value,self.flux_err[i].value,
                    model[i].value,_lhood,cl))

# This is just a preliminary version of the likelihood calculation, that
# does not include the recurrence time (or other parameters)
# Also a possible minor error that the tdel weight doesn't apply to the
# entire likelihood component

#        return ( -0.5 * np.sum( (model.value-self.flux.value)**2*inv_sigma2
#                               - np.log(2.0*pi*inv_sigma2) )
#               -tdelwt*(self.tdel.value-mburst.tdel.value*opz)**2*tdel_sig2
#               -np.log(2.*pi*tdel_sig2))

        lhood_p = ( -0.5 * np.sum( (model.value-self.flux.value)**2*inv_sigma2
                               - np.log(2.0*pi*inv_sigma2) )
               -tdelwt*(self.tdel.value-mburst.tdel.value*opz)**2*tdel_sig2
               -np.log(2.*pi*tdel_sig2))

        if breakdown:
            print ("Likelihood component breakdown (fper, tdel, lightcurve): ",lhood_cpt)

# Finally we return the sum of the likelihoods

#        print (lhood_cpt,lhood_cpt.sum())
        return lhood_cpt.sum()
#        return lhood_p

# ------- --------- --------- --------- --------- --------- --------- ---------

# Here's an example of a simulated model class

class KeplerBurst(Lightcurve):
    '''
    Example simulated burst class. Apart from the lightcurve (which is
    defined with time, lumin, and lumin_err columns), the additional
    (minimal) attributes requred are:
    filename - source file name
    tdel, tdel_err - recurrence time and error (hr)
    Lacc - accretion luminosity (in units of Mdot_Edd)
    xi - radius factor between assumed (Newtonian) value and the GR equivalent
    R_NS - model-assumed radius of the neutron star (GR)
    g - surface gravity assumed for the run
    '''

    def __init__(self, filename=None, run_id=None, path=None, **kwargs):

        if run_id != None:

# For a KEPLER run, we use the convention for filename as follows:

            self.filename = 'kepler_'+run_id+'_mean.txt'

        elif filename!= None:

            self.filename = filename

# Don't add the path to the filename, because we want to use the latter as
# a plot label (for example)

        if path == None:
            path = '.'
        self.path = path

# Read in the file, and initialise the lightcurve

        d=ascii.read(path+'/'+self.filename)
#        print (d.columns,d.meta)

        if ('time' in d.columns):
            Lightcurve.__init__(self, time=d['time']*u.s,
                            lumin=d['luminosity']*u.erg/u.s, lumin_err=d['u_luminosity']*u.erg/u.s)
        else:
            Lightcurve.__init__(self, time=d['col1']*u.s,
                            lumin=d['col2']*u.erg/u.s, lumin_err=d['col3']*u.erg/u.s)

        if ('comments' in d.meta):
            self.comments = d.meta['comments']

        if run_id != None:

# For KEPLER models, read information from the burst table
# Lately we go directly to the online version of the table

            try:
                Vizier.columns = ['all']
                model_table = Vizier.get_catalogs('J/ApJ/819/46')
                self.data = model_table[0]

# Table columns are: 'model','N','Z','H','Lacc','bstLgth','e_bstLgth','pkLum','e_pkLum',
# 'psLum','e_psLum','Fluence','e_Fluence','tau','e_tau','tDel','e_tDel','conv','e_conv',
# 'r1090','e_r1090','r2590','e_r2590','alpha1','e_alpha1','tau1','e_tau1','alpha','e_alpha',
# 'Flag'

                self.row = np.flatnonzero(self.data['model'] == run_id.encode('ascii'))[0]
#            print (self.row, self.data['model'][self.row], self.data['e_tDel'][self.row])
                self.tdel = self.data['tDel'][self.row]*u.hr
#            print (self.tdel.units)
                self.tdel_err = self.data['e_tDel'][self.row]*u.hr

# print (model_table[0].columns)

            except:
                print ("** WARNING ** can't get Vizier table, using local file")
                self.table_file='/Users/duncan/Documents/2015/Nat new catalog/summ.csv'
                self.data = ascii.read(self.table_file)

# Local file columns are slightly different: 'model','num','acc','z','h','lAcc','pul','cyc','
# burstLength','uBurstLength','peakLum','uPeakLum','persLum','uPersLum','fluence','uFluence',
# 'tau','uTau','tDel','uTDel','conv','uConv','r1090','uR1090','r2590','uR2590','singAlpha',
# 'uSingAlpha','singDecay','uSingDecay','alpha','uAlpha','flag'

                self.row = np.flatnonzero(self.data['model'] == "'xrb"+run_id+"'")[0]

# Set the legacy tdel attribute, as well as the correct units for tdel_err

                self.tdel = self.data['tDel'][self.row]/3600.*u.hr
                self.tdel_err = self.data['uTDel'][self.row]/3600.*u.hr

# Additional parameter here gives the radius conversion between the assumed
# (Newtonian) value of 10 km and the GR equivalent for a neutron star of
# mass 1.4 M_sun, to achieve the same surface gravity.
# If your model-defined surface gravity changes, or the radius (or mass)
# of the NS, this quantity will also have to change

            self.xi = 1.12

# Define the neutron star mass, radius, and surface gravity

            self.M_NS = 1.4*const.M_sun
            self.R_Newt = 10.*u.km
            self.R_NS = self.R_Newt*self.xi
            self.opz = 1./sqrt(1.-2.*const.G*self.M_NS/(const.c**2*self.R_NS))
            self.g = const.G*self.M_NS/(self.R_NS**2/self.opz)

# Set all the remaining attributes

            for attr in self.data.columns:
                setattr(self,attr,self.data[attr][self.row])

        elif kwargs != None:

# For non-KEPLER models, you can use kwargs to populate the parameters

            for key in kwargs:
#                print (key, kwargs[key])
                if (key == 'tdel') | (key == 'tdel_err'):
                    setattr(self,key,float(kwargs[key])*u.hr)
                else:
                    setattr(self,key,kwargs[key])

        if ((not hasattr(self,'Lacc')) & hasattr(self,'lAcc')):
            self.Lacc = self.lAcc

# Set the mdot with the correct units

        if hasattr(self,'Lacc'):
            self.mdot = self.Lacc*1.75e-8*const.M_sun/u.yr

# The flux method is supposed to calculate the flux at a particular distance

    def flux(self,dist):
        if not hasattr(self,'dist'):
            self.dist = dist

        return self.lumin/(4.*pi*self.dist.to('cm')**2)

# ------- --------- --------- --------- --------- --------- --------- ---------

# Now define a new likelihood function, based on the old one, but which
# can handle multiple pairs of observed bursts

# First define the prior

def lnprior(theta):
    dist, inclination, opz, t_off = theta

# We have currently flat priors for everything but the inclination, which
# has a probability distribution proportional to cos(i)

    if (dist.value > 0.0 and 0.0 < inclination.value < 90.
        and 1. < opz < 2):
        return np.log(np.cos(inclination))

    return -np.inf

# ------- --------- --------- --------- --------- --------- --------- ---------

def apply_units(params,units = (u.kpc, u.degree, None, u.s)):

# When called from emcee, the parameters array might not have units. So
# apply them here, in a copy of params (uparams)

    ok = True
    uparams = []
    n_units = len(units)
    for i, param in enumerate(params):
#        print (i,param,units[i])

# Define iunit here so we apply the last unit in the list, to the 4th (and
# all subsequent) element of params
# This is to cover the variable number of offset values for the burst
# start, which depends upon the number of bursts matching simultaneously

        iunit = min(i,n_units-1)

        if units[iunit] != None:
            if hasattr(param,'unit') == False:
#                uparams.append(param*units[i])
                uparams.append(param*units[iunit])
#                print ("Applying unit to element ",i)
            else:
                uparams.append(param)
#                if param.unit != units[i]:
                if param.unit != units[iunit]:
                    ok = False
        else:
            uparams.append(param)
            if hasattr(param,'unit') == True:
                ok = False
#    print (params, uparams, ok)
    assert(ok == True)

    return uparams

# ------- --------- --------- --------- --------- --------- --------- ---------

def lhoodClass(params,obs,model):
    '''
    Calculate the likelihood related to one or more model-observation
    comparisons The corresponding call to emcee will (necessarily) look
    something like this:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lhoodClass, args=[obs, models ])
    '''

    uparams = apply_units(params)

# We can pass multiple bursts, in which case we just loop over the function

    alh = 0.0

    if type(obs) == tuple:
        n = len(obs)
        if (n != len(model)):
            print ("** ERROR ** number of observed and model bursts don't match")

        for i in range(n):

# Need to create a reduced parameter array here, keeping only the offset
# value appropriate for this burst

            _params = uparams[0:3]
            _params.append(uparams[3+i])
#            alh += lhoodClass(uparams,obs[i],model[i])
            alh += lhoodClass(_params,obs[i],model[i])
#            print (i,n,alh)

    else:

# Or if we have just one burst, here's what we do

        alh = obs.compare(model,uparams)

    return alh + lnprior(uparams[0:4])

# ------- --------- --------- --------- --------- --------- --------- ---------

def plot_comparison(obs,models,param=None,sampler=None,ibest=None):
    '''
    This routine will plot each observation vs. each model, for a given
    set of parameters. If you supply an emcee sampler object, it will find
    the highest-likelihood sample and use that
    '''

# Need to specify at least one of param, sampler

    assert ((param != None) | (sampler != None))

    if param == None:

# If no parameters are specified, try to get the best example
# from the sampler object

        if ibest == None:

# Identify the maximum probability set of parameters

            chain_shape = np.shape(sampler.lnprobability)
            imax = np.argmax(sampler.lnprobability)
            ibest = np.unravel_index(imax,chain_shape)

# print (imax,chain_shape,ibest)
        prob_max = sampler.lnprobability[ibest]
        _param_best = sampler.chain[ibest[0],ibest[1],:]

    else:

# Just use the supplied parameters

        _param_best = param

    param_best = apply_units(_param_best)
    print ('Got parameter set for plotting: ',param_best)

# Want to have a check here in case there are other than 3 models & obs
# to compare

    n = len(obs)
    assert (n == 3)

    b1, b2, b3 = obs
    m1, m2, m3 = models
    # b1 = obs[0]
    # m1 = models[0]

# Can't use the gridspec anymore, as this is used for the individual plots

#    fig = plt.figure()
#    gs = gridspec.GridSpec(3,2)

# plot the model comparisons. Should really do a loop here, but not sure
# exactly how

    _param_best = param_best[0:4]
#    ax1 = fig.add_subplot(gs[0,0])
    b1.compare(m1,_param_best,plot=True,subplot=False)
#    fig.set_size_inches(8,3)

    _param_best = param_best[0:3]
    _param_best.append(param_best[4])
# #    ax2 = fig.add_subplot(gs[0,1])
    b2.compare(m2,_param_best,plot=True,subplot=False)

    _param_best = param_best[0:3]
    _param_best.append(param_best[5])
# #    ax3 = fig.add_subplot(gs[1,0])
    b3.compare(m3,_param_best,plot=True,subplot=False)

# Now assemlbe the tdel values for plotting. This is a bit clumsy

    x = np.zeros(n)
    xerr = np.zeros(n)
    y = np.zeros(n)
    yerr = np.zeros(n)
    for i, burst in enumerate(obs):
        x[i] = burst.tdel.value
        xerr[i] = burst.tdel_err.value
        y[i] = models[i].tdel.value*param_best[2]
        yerr[i] = models[i].tdel_err.value*param_best[2]

#    print (x,xerr,y,yerr)
#    print (type(x))
#    print (type(b1.tdel),type(m1.tdel))

#    ax4 = fig.add_subplot(gs[1:,-1])
    fig = plt.figure()
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o')
    plt.plot([3,6],[3,6],'--')
    plt.xlabel('Observed $\Delta t$ (hr)')
    plt.ylabel('Predicted $(1+z)\Delta t$ (hr)')

#    fig.set_size_inches(10,6)

# ------- --------- --------- --------- --------- --------- --------- ---------

def plot_contours(sampler,parameters=[r"$d$",r"$i$",r"$1+z$"],
        ignore=10,plot_size=6):
    '''
    Simple routine to plot contours of the walkers, ignoring some initial
    fraction of the steps (the "burn-in" phase)

    Documentation is here https://samreay.github.io/ChainConsumer/index.html
    '''

    nwalkers, nsteps, ndim = np.shape(sampler.chain)

    #samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
    samples = np.load('temp/chain_200.npy').reshape((-1,ndim))
#    print (np.shape(samples))

# This to produce a much more beautiful plot

    c = ChainConsumer()
    c.add_chain(samples, parameters = parameters)#,r"$\Delta t$"])


    fig = c.plot()
    fig.set_size_inches(6,6)

    return c

# end of file burstclass.py
