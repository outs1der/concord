# Burst class for MINBAR
# 
# This code is a preliminary attempt to develop a burst class for MINBAR,
# including model-observation comparisons. 
# 
# The first type of implementation we want to include is the "reference"
# bursts, which have a format like this:
# 
#     #
#     # Columns are:
#     # 'Time [s]' 'dt [s]' 'flux [10^-9 erg/cm^2/s]' 'flux error [10^-9 erg/cm^2/s]' 'blackbody temperature kT [keV]' 'kT error [keV]' 'blackbody normalisation K_bb [(km/d_10kpc)^2]' 'K_bb error [(km/d_10kpc)^2]' chi-sq
#      -1.750  0.500   1.63    0.054   1.865  0.108   13.891   4.793  0.917
#      -1.250  0.500   2.88    1.005   1.862  0.246   22.220   4.443  1.034
#      -0.750  0.500   4.38    1.107   1.902  0.093   30.247   2.943  1.089
#      -0.250  0.500   6.57    0.463   1.909  0.080   46.936   6.969  0.849
# 
# We ultimately want to run the comparison step in mlfit, which is invoked
# with something like this:
# 
#     params, uparams, ms, chisq, goodParams = lcCompare(burstFile,obsfile,plot=True)

import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.io.ascii as ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d
from astroquery.vizier import Vizier

from math import *

# ------- --------- --------- --------- --------- --------- --------- ---------

def decode_LaTeX(string):
    '''
    This function converts a LaTeX numerical value (with error) to one
    or more floating point values
    '''
    
    assert (type(string) == str) or (type(string) == np.str_)
    
# Look for the start of a LaTeX numerical expression

    val_match = re.search('\$ *([0-9]+\.[0-9]+)',string)
    
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
    
# Additional parameter here gives the radius conversion between the assumed
# (Newtonian) value of 10 km and the GR equivalent for a neutron star of 
# mass 1.4 M_sun, to achieve the same surface gravity.
# If your model-defined surface gravity changes, or the radius (or mass)
# of the NS, this quantity will also have to change

    xi = 1.12
    
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
    return ( (xi/opz)**2 * fInterp(obs.time+(0.5-obs.timepixr)*obs.dt)*model.lumin.unit
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
        plt.step(self.time,self.flux,where='post')
        plt.xlabel("Time ({0.unit:latex_inline})".format(self.time))
        plt.ylabel("Flux ({0.unit:latex_inline})".format(self.flux))

# ------- --------- --------- --------- --------- --------- --------- ---------

# This class is for observed bursts; we define a compare method to match
# with models

class ObservedBurst(Lightcurve):
    
    def __init__(self, filename):
        d=ascii.read(filename)

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
        self.filename = filename
        
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
           
        self.table_file = '/Users/duncan/burst/reference/doc/table2.tex'
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
                print (i, lcfile, filename)

        if hasattr(self,'row'):

            tdel, tdel_err = decode_LaTeX(self.table['$\Delta t$ (hr)'][self.row])
            self.tdel = tdel*u.hr
            if tdel_err != None:
                self.tdel_err = tdel_err*u.hr

# Decode the other table parameters

        label = ['cbol','mdot','fluen','F_pk','alpha']
        unit = [1.,1.75e-8*const.M_sun/u.yr,1e-6*u.erg/u.cm**2/u.s,
                1e-9*u.erg/u.cm**2/u.s,1.]
        for i, column in enumerate(self.table.columns[5:10]):
#            print (i, column, label[i], self.table[column][row], type(self.table[column][row]))
            if ((type(self.table[column][self.row]) == np.str_)):
#    or (type(self.table[column][row]) == np.str_)):
                val, val_err = decode_LaTeX(self.table[column][self.row])
                setattr(self,label[i],val*unit[i])
                if val_err != None:
#                    print (column, label[i]+'_err',val_err)
                    setattr(self,label[i]+'_err',val_err*unit[i])
            else:
                setattr(self,label[i],self.table[column][self.row]*unit[i])
                
    def compare(self, mburst, param = (6.1*u.kpc,60.*u.degree,1.,+8.*u.s), 
		plot = False, subplot = True):

        dist, inclination, opz, t_off = param

# This parameter gives the relative weight to the tdel for the likelihood

        tdelwt=2.5e3
        
# Calculate the rescaled model flux with the passed parameters

        model = modelFunc(param,self,mburst)
        assert model.unit == self.flux.unit == self.flux_err.unit

# can check here if the object to compare is actually a model burst
#        print (type(mburst))

# Plot the observed burst

        if plot:
            self.plot()
        
# overplot the rescaled model burst
#        plt.plot(mburst.time, mburst.flux(dist))
            plt.plot(self.time+(0.5-self.timepixr)*self.dt, model,'r.')
    
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
    
# This is just a preliminary version of the likelihood calculation, that
# does not include the recurrence time (or other parameters) and may also
# not properly incorporate the normalisation

        inv_sigma2 = 1.0/(self.flux_err.value**2)
        tdel_sig2 = 1.0/(self.tdel_err.value**2+(mburst.tdel_err.value*opz)**2)
        return ( -0.5 * np.sum( (model.value-self.flux.value)**2*inv_sigma2 
                               - np.log(2.0*pi*inv_sigma2) ) 
               -tdelwt*(self.tdel.value-mburst.tdel.value*opz)**2*tdel_sig2 
               -np.log(2.*pi*tdel_sig2))
        
# ------- --------- --------- --------- --------- --------- --------- ---------

# Here's an example of a simulated model class

class KeplerBurst(Lightcurve):
    
    def __init__(self, run_id, path=None):
        self.filename = 'kepler_'+run_id+'_mean.txt'
        if path != None:
            self.filename = path+'/'+self.filename

        d=ascii.read(self.filename)

        Lightcurve.__init__(self, time=d['col1']*u.s,  
                            lumin=d['col2']*u.erg/u.s, lumin_err=d['col3']*u.erg/u.s)

        self.comments = d.meta['comments']
        
# Read information from the burst table
# Lately we go directly to the online version of the table

        try:
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
        
# Set all the remaiing attributes

        for attr in self.data.columns:
            setattr(self,attr,self.data[attr][self.row])
            
        if (not hasattr(self,'Lacc')):
            self.Lacc = self.lAcc
            
# Set the mdot with the correct units

        self.mdot = self.Lacc*1.75e-8*const.M_sun/u.yr

# The flux method is supposed to calculate the flux at a particular distance

    def flux(self,dist):
        if not hasattr(self,'dist'):
            self.dist = dist
            
        return self.lumin/(4.*pi*self.dist.to('cm')**2)

# end of file burstclass.py
