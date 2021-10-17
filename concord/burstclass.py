# Burst class for MINBAR
#
# This code is a preliminary attempt to develop a burst class for MINBAR,
# including model-observation comparisons. 
# 
# See also the repository on BitBucket:
#   https://bitbucket.org/minbar/concord
#
# Duncan.Galloway@monash.edu, 2018
#
# including model-observation comparisons.
#
# Here are the classes and functions making up this module:
#
# class Lightcurve(object):
# class ObservedBurst(Lightcurve):
# class KeplerBurst(Lightcurve):
#
# def fper(mburst, param, c_bol=1.0):
# def modelFunc(p,obs,model, disc_model):
# def lnprior(theta):
# def apply_units(params,units = (u.kpc, u.degree, None, u.s)):
# def lhoodClass(params, obs, model, weights, disc_model):
# def plot_comparison(obs,models,param=None,sampler=None,ibest=None):
# def plot_contours(sampler,parameters,ignore,plot_size):

import os
from .utils import *
import astropy.io.ascii as ascii
import csv
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d
from astroquery.vizier import Vizier
from chainconsumer import ChainConsumer
from datetime import datetime

from concord import diskmodel as dm

# Get the path for the concord files from the environment variable

import pkg_resources

CONCORD_PATH = pkg_resources.resource_filename('concord','data')
# CONCORD_PATH = os.environ['CONCORD_PATH']
ETA = 1.e-6
# default colours for plotting
OBS_COLOUR = 'b'
MODEL_COLOUR = 'g'

# ------- --------- --------- --------- --------- --------- --------- ---------

def fper(mburst, param, c_bol=1.0):
    '''
    Calculates the persistent flux, based on the supplied mdot, redshift
    etc. Earlier we passed the individual parameters, but easiest to just provide
    the model burst
    '''

    dist, inclination, _opz, t_off = param

    if hasattr(mburst, 'g') & hasattr(mburst, 'R_Newt'):

        # The combination of an input redshift an model gravity uniquely define
        # a mass-radius combination (as for modelFunc)

        _M, _R = calc_mr(mburst.g, _opz)

        # Here we calculate the value of xi (ratio of GR to Newtonian radii),
        # appropriate for the adopted value of (1+z). This is used instead of the
        # value attached to the model, because that's for a different redshift

        # xi = sqrt(_opz)
        xi = (_R / mburst.R_Newt).decompose()
        # print (xi)
        _mdot = mburst.mdot
    else:
        # if we're not passing a ModelBurst, we assume the input is an mdot
        # value (in g/s or equivalent, for consistency with the ModelBurst attribute),
        # and calculate with the additional parameters
        xi = sqrt(_opz)
        _mdot = mburst

    # The mdot measured by the distant observer now depends upon the assumed
    # redshift; see Keek & Heger (2011) eq. B16

    mdot_infty = xi**2 * _mdot / _opz

    # Next we calculate Q_grav, to calculate the inferred persistent flux that
    # we should see (based on the accretion rate); that is not given in Lampe
    # et al.  2016, but is given in gal03d
    # Note that (as for the M_NS) we're using the passed parameter opz here,
    # not the value that is associated with the model burst

#        Q_grav = const.G*M_NS/mburst.R_NS # approximate
    Q_grav = const.c**2*(_opz-1)/_opz

    # Could split this bit off into a separate flux function, which would do the
    # anisotropy and distance factors

    # persistent flux (see Keek & Heger 2011, eq. B20, noting that our mdot is averaged
    # over the neutron star surface):

    return lum_to_flux( mdot_infty*Q_grav/_opz, dist, c_bol, inclination=inclination ).to(u.erg/u.cm**2/u.s)

# ------- --------- --------- --------- --------- --------- --------- ---------

def modelFunc(p,obs,model, disc_model):
    '''
    This function performs the stretching and rescaling of the (model
    predicted) burst lightcurve based on the input parameter set
    The calculations are based on those in Appendix B of Keek & Heger
    2011 (ApJ 743:189)
    Parameter array is (for now) a tuple with the appropriate units:
    ( distance, inclination, redshift, time offset )
    '''

    dist, inclination, _opz, t_off = p

# Use the anisotropy function to calculate the anisotropy factors given
# the inclination

    xi_b, xi_p = dm.anisotropy(inclination)

    # The combination of an input redshift an model gravity uniquely define
    # a mass-radius combination

    _M, _R = calc_mr(model.g, _opz)

    # Here we calculate the value of xi (ratio of GR to Newtonian radii),
    # appropriate for the adopted value of (1+z). This is used instead of the
    # value attached to the model, because that's for a different redshift

    # xi = sqrt(_opz)
    xi = (_R / model.R_Newt).decompose()

    # First interpolate the model values onto the observed grid
    # This step defines the interpolation function, using a shifted and rescaled
    # model time to account for time dilation in the NS surface frame

    fInterp = interp1d(_opz*(model.time-t_off),model.lumin,bounds_error=False,
                      fill_value = min(model.lumin))

    # Then we return the predicted model flux, interpolated onto the observed
    # burst time bins. We shift the observed times to the middle of the bin, using
    # the timepixr attribute.
    # We also account for the distance, and the anisotropy parameter, using the
    # convention of Fujimoto et al. 1988, i.e.
    #   L_b = 4\pi d^2 \xi_b F_b
    # The scaling here is as per eq. B15 in Keek et al. 2011

    #    return ( (model.xi/opz)**2
    # since xi = sqrt(_opz) (see above), this first term just becomes 1/_opz
    #    return ( (xi/_opz)**2

    return ( (xi / _opz)**2
            * fInterp(obs.time+(0.5-obs.timepixr)*obs.dt)*model.lumin.unit
            / (4.*pi*dist.to('cm')**2) / xi_b )

# ======= ========= ========= ========= ========= ========= ========= =========

class Lightcurve(object):
    """
    The fundamental burst object, which has attributes like time (array),
    flux/luminosity etc. and methods including plot. Minimal attributes
    are:

    time, dt - array of times and time bin duration
    timepixr - set to 0.0 (default) if time is the start of the bin, 0.5 for
               midpoint
    flux, flux_err - flux and error (no units assumed)
    lumin, lumin_err - luminosity and error

    Normally only one of flux or luminosity would be supplied

    Example: simulated burst (giving time and luminosity)

    Lightcurve(self, time=d['time']*u.s,
        lumin=d['luminosity']*u.erg/u.s, lumin_err=d['u_luminosity']*u.erg/u.s)

    Example: observed burst giving flux

    Lightcurve(self, time=d['col1']*u.s, dt=d['col2']*u.s,
        flux=d['col3']*1e-9*u.erg/u.cm**2/u.s,
        flux_err=d['col4']*1e-9*u.erg/u.cm**2/u.s)
    """

# ------- --------- --------- --------- --------- --------- --------- ---------

    def __init__(self, *args, **kwargs):
        """initialise the object by assigning named entities from the kwargs"""

# Initialise the various attributes, where present. We expect to have at
# least one of flux or lumin (and the related uncertainty)
# Units are handled by the parent classes (e.g. ObservedBurst and KeplerBurst)

        self.filename = kwargs.get('filename',None)

        self.time = kwargs.get('time',None)
        self.timepixr = kwargs.get('timepixr',0.0)

        # Not good enough to not have a dt column
#        self.dt = kwargs.get('dt',None)
        if 'dt' in kwargs:
            self.dt = kwargs.get('dt')
        else:
            dt = self.time[1:]-self.time[:-1]
            self.dt = np.append(dt.value,dt[-1].value)*dt.unit

#        if kwargs.get('flux',None):

        self.lumin = kwargs.get('lumin',None)
        self.lumin_err = kwargs.get('lumin_err',None)

# Here we define the dt_nogap and good attributes, which are used in the
# fluence calculation (and elsewhere)

        self.dt_nogap = self.dt
        if 'flux' in kwargs:
            self.flux = kwargs.get('flux',None)
            self.flux_err = kwargs.get('flux_err',None)
            # print ('flux',self.flux)
            # print ('flux_err',self.flux_err)

            self.good = np.where(self.flux_err < self.flux)[0]
            if len(self.good) > 0:
                for i in range(len(self.good)-1):
                    self.dt_nogap[self.good[i]] = max(
    [self.dt[self.good[i]],self.time[self.good[i+1]]-self.time[self.good[i]]] )
        else:
            self.good = np.arange(len(self.lumin))

# ------- --------- --------- --------- --------- --------- --------- ---------

    def print(self):
        '''
        Print the basic parameters of the lightcurve. Can be called in turn
        by the info commands for KeplerBurst, ObservedBurst to display the
        properties of the parent class... once those are written
        :return:
        '''

        print ('Lightcurve properties')
        print (f'  filename = {self.filename}')
        print ('  time range = ({:.3f},{:.3f})'.format(min(self.time),max(self.time)))
#        if hasattr(self,'lumin'):
        if self.lumin is not None:
            print ('  luminosity range = ({:.3e},{:.3e})'.format(min(self.lumin),max(self.lumin)))
        else:
            print ('  flux range = ({:.3e},{:.3e})'.format(min(self.flux),max(self.flux)))

# ------- --------- --------- --------- --------- --------- --------- ---------

    def write(self,filename='lightcurve.csv',addhdr=None):
        '''
        Write the lightcurve to a file
        '''

        if not hasattr(self,'flux'):
            print ("Luminosity writing not yet implemented. Sorry")
            return

# this doesn't work:
#        print ("Test of inheriting attributes from parent: {}".format(self.tdel))

        flux_unit = 1e-9*self.flux.unit
        with open (filename,'w') as f:

# print header information

            f.write("# file {}\n".format(filename))
            f.write("# created {}\n".format(str(datetime.now())))
            f.write("#\n")
            f.write("# Lightcurve object written to file via write method\n")
            f.write("#\n")
            f.write("# timepixr = {}\n".format(self.timepixr))
            if addhdr != None:
                f.write("#\n")
                if type(addhdr) == str:
                    f.write("# {}\n".format(addhdr))
                else:
                    for line in addhdr:
                        f.write("# {}\n".format(line))
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("# time ({}), dt ({}), flux ({}), flux_err\n".format(self.time.unit,self.dt.unit,flux_unit))

            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.time.value,self.dt.value,
                                 self.flux/flux_unit,self.flux_err/flux_unit))

# ------- --------- --------- --------- --------- --------- --------- ---------

# Later we can make this plot method more elaborate
# where argument for step is appropriate for timepixr=0.0

    def plot(self, yerror=True, obs_color='b', model_color='g',**kwargs):
        """Plot the lightcurve, accommodating both flux and luminosities"""

        assert self.timepixr == 0.0

        # Want to ensure consistent colors between the steps and points, which
        # otherwise won't occur

        kwargs_passed = kwargs
        if not 'color' in kwargs:
            if type(self) == ObservedBurst:
                kwargs['color'] = OBS_COLOUR
            elif type(self) == KeplerBurst:
                kwargs['color'] = MODEL_COLOUR
            else:
                kwargs['color'] = 'r'

        luminosity = False
        if hasattr(self,'flux'):
            y = self.flux
            yerr = self.flux_err
            ylabel = "Flux ({0.unit:latex_inline})".format(self.flux)
        elif hasattr(self,'lumin'):
            luminosity = True
            y = self.lumin
            yerr = self.lumin_err
            ylabel = "Luminosity ({0.unit:latex_inline})".format(self.lumin)
        
        plt.step(self.time,y,where='post',label=self.filename, **kwargs)
#        print (type(self.dt), type(yerr))
#        print (yerror & (self.dt != None) & (yerr != None))
#        if (yerror & (self.dt != None) & (yerr != None)):
        if yerror:
            try:
                plt.errorbar(self.time.value+(0.5-self.timepixr)*self.dt.value,
                    y.value, yerr=yerr.value,fmt='.', **kwargs)
#            plt.plot(self.time,y,label=self.filename)
            except:
                pass
#            print ("** WARNING ** errors not present, can't plot errorbars")

        if luminosity:
            plt.plot(np.array([min(self.time.value),max(self.time.value)]),
                     np.ones(2)*self.L_Edd,'--', **kwargs)
        plt.xlabel("Time ({0.unit:latex_inline})".format(self.time))
        plt.ylabel(ylabel)

        # Restore the original kwargs array before exiting

        kwargs = kwargs_passed

# ------- --------- --------- --------- --------- --------- --------- ---------

    def observe(self, param = [6.1*u.kpc,60.*u.degree,1.26,-10.*u.s], obs=None,
        disc_model='he16_a',c_bol=1.0):
        """
        Convert a luminosity profile to a simulated observation, with
        plausible errors. See Keek & Heger (2011) for the background calculations

        This method might make more sense as part of the KeplerBurst class
        (or a parent SimulatedBurst class), but for now we include this method
        as part of the lightcurve class, so that future classes of model
        bursts can access it
        """

        if not hasattr(self,'lumin'):
            print ("concord.observe: ** ERROR ** need luminosity to simulate observation")
            return None

        if not hasattr(self,'g'):
            print ("concord.observe: ** ERROR** need model gravity to simulate observation")

        # First unpack the simulation parameters

        dist, inclination, _opz, t_off = param
        xi_b, xi_p = dm.anisotropy(inclination, model=disc_model)

        if (obs == None):

            # No observation is provided, so we have to make up a burst lightcurve
            # with some dummy values for the time bins and fluxes
            # For consistency, we need to include the units also

            dt = 0.25*u.s
            npts = ceil((max(self.time)-min(self.time))*_opz/dt)
            obs = Lightcurve(time=np.arange(npts)*dt+t_off,
                             dt=np.full(npts,dt)*u.s,
                             flux=np.zeros(npts)*u.erg/u.cm**2/u.s,
                             flux_err=np.zeros(npts)*u.erg/u.cm**2/u.s)
#            print (obs.flux_err)

        else:
            npts = len(obs.time)

        # modelFunc does the actual scaling

        model = modelFunc(param, obs, self, disc_model)

        if hasattr(obs,'flux_err'):

            # Add some errors based on the flux_err

            model += np.random.normal(size=npts)*obs.flux_err

        else:
            print ("concord.observe: ** WARNING ** can't add scatter without flux errors")

        # This is OK (modelFunc will deal with it)

        # if hasattr(self,'opz'):
        #     if abs(self.opz-_opz) > ETA:
        #         print ('observe: ** WARNING ** inconsistent simulation redshift for model value')
        #         print ('         model value: {:.4f}, simulation value: {:.4f}\n'.format(self.opz,_opz))

        # Calculate the equivalent persistent flux

        _fper = fper(self, param, c_bol=c_bol)

        # print ('obs.time:',obs.time)
        # print ('obs.dt:',obs.dt)
        # print ('flux:',model)
        # print ('flux_err:',obs.flux_err)

        # And return an ObservedBurst object with appropriate label
        # Have a problem here with the doubling up of units, as the ObservedBurst.__init__
        # method will apply it's own

        sim = ObservedBurst(time=obs.time.value,dt=obs.dt.value,
                          flux=model.value,flux_err=obs.flux_err.value,
                          tdel = self.tdel*_opz,tdel_err=self.tdel_err*_opz,
                          fper = _fper, filename="{} @ {}".format(self.filename,dist),
                          c_bol=c_bol,
        # Include the simulation parameters in the burst description
                          sim_dist=dist, sim_inclination=inclination, sim_opz=_opz,
                          sim_t_off=t_off, sim_xi_b=xi_b, sim_xi_p=xi_p, sim_disc_model=disc_model,
        # Should potentially also include the model burst parameters, where available
                          model_g = self.g
                            )

        return sim

#         return Lightcurve(time=obs.time,dt=obs.dt,
#                           flux=model,flux_err=obs.flux_err,
# #                          tdel = self.tdel*_opz,tdel_err=self.tdel_err*_opz,
# #                          fper = fper(self.mdot,_opz,dist,xi_p,c_bol=c_bol),
#                           filename="{} @ {}".format(self.filename,dist))

# ------- --------- --------- --------- --------- --------- --------- ---------

    def fluence(self, plot=False, warnings=True):
        """
        Calculate the fluence for a lightcurve, following the approach
        from get_burst_data
        This code relies on the presence of dt_nogap and good attributes,
        which should be calculated when the Lightcurve object is defined
        """

        minpts=4	# Fit to a minimum of this number of points in the tail
        fitfrac=0.1	# Fit to points over this fraction span in the flux
        fdiffmax=0.1	# Threshold difference for exponential decay constant
			#   between different fit instances, to gauge the
			#   convergence/reliability of the fit procedure

        if hasattr(self,'flux'):
            y = self.flux
            yerr = self.flux_err
        elif hasattr(self,'lumin'):
            y = self.lumin
            yerr = self.lumin_err

        imax = np.argmax(y[self.good])
        pflux = y[self.good[imax]]
#        self.pflux = y[self.good[imax]]
#        self.pfluxe = self.flux_err[self.good[imax]]

        if (max(self.dt_nogap/self.dt) > 2.) & warnings:
            print ('** WARNING ** excessive gap filling not yet implemented')

        fluen = sum(y[self.good]*self.dt_nogap[self.good])
        fluene_stat = np.sqrt(sum( (yerr[self.good]*self.dt_nogap[self.good])**2 ))

# Now extrapolate the flux beyond the extent of the data, if possible

        rchisq=0.0
        ng = len(self.good)
        ntail=ng-imax-1	# Number of points in the tail
        if ntail >= minpts:
            tail=np.where(self.time[self.good] > self.time[self.good[imax]])[0]

# Determine the minimum flux reached in the tail. This might be bigger than
# fitfrac*pflux, so we need to count from there as a zero point

        minflux=0.
        if len(tail) > 0:
            minflux=min(y[self.good[tail]])

# Now determine the points in the tail which are *excluded* from the fit;
# for the fit selection, we then calculate from the end of this interval

        tailexcl=np.where((self.time[self.good] > self.time[self.good[imax]])
                       & (y[self.good] > minflux+fitfrac*pflux))[0]

        if len(tailexcl) > 0:
            sel=self.good[min([ng-4,max(tailexcl)]):ng]
				# Last three points, at least
        else:
            sel=self.good[min([imax+1,ng-4]):ng]
				# ...or, if all the tail points are excluded,
				#   just take the last three

        # Now actually do the fitting (to the log of the flux)
        # Originally we used linfit, but this is a bit obscure; subsequently
        # replaced with np.polyfit (see
        # https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)

        # result, cvm, fit_info = linfit(
        #     self.time[sel]-self.time[sel[0]]+self.dt_nogap[sel]/2.,
        #     np.log(y[sel].value),
        #     sigmay=0.5*(np.log((y[sel]+yerr[sel]).value)
        #                 -np.log((y[sel]-yerr[sel]).value)),
        #     return_all=True)
        result = np.polyfit(
          self.time[sel]-self.time[sel[0]]+self.dt_nogap[sel]/2.,
          np.log(y[sel].value), 1,
          w=1./(0.5*(np.log((y[sel]+yerr[sel]).value)
                     -np.log((y[sel]-yerr[sel]).value))))

# Result is a 2-element array, slope and intercept (opposite order to IDL)

        f_int=0.
        if (result[0] > 0.0) & warnings:
            print ('** WARNING ** fit is rising, result is not trustworthy')
        else:
            tmax=max(self.time[sel]+self.dt_nogap[sel]-self.time[sel[0]])
            f_int=-fluen.unit/result[0]*exp(result[1])*exp(tmax.value*result[0])

        fluen+=f_int

# Show the integrated part against the lightcurve, if required
# This could be improved

        if plot:
            self.plot()
            xx=np.arange(20)/19.*(max(self.time[sel]+self.dt_nogap[sel])-self.time[sel[0]])*4
#            print ("xx: ",xx)#+self.time[sel[0]])
#            print ("yy:",np.exp(xx.value*result[0]))
            plt.plot(xx+self.time[sel[0]],
                np.exp(result[1])*np.exp(xx.value*result[0]))

        if f_int > fluene_stat:
            if warnings:
                print ('** WARNING ** extrapolated fluence > stat_error, replacing')
            return fluen, f_int

        return fluen, fluene_stat

# ======= ========= ========= ========= ========= ========= ========= =========

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
    fper, fper_err - persistent flux level (1e-9 erg/cm^2/s)
    c_bol - bolometric correction
    mdot - accretion rate (total, not per unit area)
    fluen_table - burst fluence
    F_pk - burst peak flux
    alpha - alpha value

    The units for selected parameters are all stored as astropy.units
    objects
    '''

    def __init__(self, time, dt, flux, flux_err, **kwargs):
        '''
        Now we define a Lightcurve instance, using the columns from the file
        Units are applied to the input arrays (and assumed to be correct!)
        :param time:
        :param dt:
        :param flux:
        :param flux_err:
        :param kwargs:
        '''

        Lightcurve.__init__(self, time = time*u.s, dt = dt*u.s,
                                flux = flux*u.erg/u.cm**2/u.s,
                                flux_err= flux_err*u.erg/u.cm**2/u.s)

# End block for adding attributes from the file. Below you can use the
# additional arguments on init to set or override attributes

        if kwargs != None:

            for key in kwargs:
#                print (key, kwargs[key])
                # I think these special cases below were for instances where the parameters might
                # be passed as strings, e.g. from the LaTeX table
                if ((key == 'tdel') | (key == 'tdel_err')) & (not hasattr(kwargs[key],'unit')):
                    setattr(self,key,float(kwargs[key])*u.hr)
                elif ((key == 'fper') | (key == 'fper_err')) & (not hasattr(kwargs[key],'unit')):
                    setattr(self,key,float(kwargs[key])*u.erg/u.cm**2/u.s)
                else:
                    setattr(self,key,kwargs[key])

    @classmethod
    def ref(cls, source, dt):
        '''
        Method to read in a reference burst and populate the relevant arrays to create
        an ObservedBurst
        Calling approach:
        obs = ObservedBurst.ref('GS 1826-24', 3.5)
        '''

        # First read in the table
        # Because we apply this at the class level, it's available to all instances
        # Other attributes have to be applied at the instance level

        cls.table_file = os.path.join(CONCORD_PATH, 'table2.tex')
        cls.table = Table.read(cls.table_file)

        # Below we associate each epoch with a file

        file = ['gs1826-24_5.14h.dat',
                'gs1826-24_4.177h.dat',
                'gs1826-24_3.530h.dat',
                'saxj1808.4-3658_16.55h.dat',
                'saxj1808.4-3658_21.10h.dat',
                'saxj1808.4-3658_29.82h.dat',
                '4u1820-303_2.681h.dat',
                '4u1820-303_1.892h.dat',
                '4u1636-536_superburst.dat']
        cls.table['file'] = file

        # Now find which one you mean. Want to assemble a key that will match the filename

        row = None
        key = '{}_{}'.format(source.lower().replace(" ",""),dt)
        # print (key)
        for i, lcfile in enumerate(cls.table['file']):
            if key in lcfile:
                row = i
        #        print (i, lcfile, cls.key)

        if row is None:
            print ('** ERROR ** no match for key {}'.format(key))
            return

        # print ('source = {}'.format(source))
        # print ('dt = {}'.format(dt))

        tdel, tdel_err = decode_LaTeX(cls.table['$\Delta t$ (hr)'][row])

        if tdel_err is None:

        # If no tdel error is supplied by the file (e.g. for the later bursts from
        # SAX J1808.4-3658), we set a nominal value corresponding to 1 s (typical
        # RXTE time resolution) here

            tdel_err = 1. / 3600.

        # Here is the dictionary to provide the keyword arguments

        rowparam = {'key': key, 'row': row, 'tdel': tdel, 'tdel_err': tdel_err, 'source': source}

        # Decode the other table parameters
        # The label below is what each column will become as an attribute

        label = ['fper', 'c_bol', 'mdot', 'fluen_table', 'F_pk', 'alpha']
        # fper units are applied at the ObservedBurst __init__ stage
        # unit = [1e-9 * u.erg / u.cm ** 2 / u.s, 1., 1.75e-8 * const.M_sun / u.yr,
        unit = [1e-9, 1., 1.75e-8 * const.M_sun / u.yr,
                            1e-6 * u.erg / u.cm ** 2,
                            1e-9 * u.erg / u.cm ** 2 / u.s, 1.]

        for i, column in enumerate(cls.table.columns[4:10]):
            # print (i, column, label[i], self.table[column][row], type(self.table[column][row]))

            if ((type(cls.table[column][row]) == np.str_)):
            # or (type(self.table[column][row]) == np.str_)):

                # Here we convert the table entry to a value. We have a couple of options
                # here: raw value, range (separated by "--"), or LaTeX expression

                range_match = re.search('([0-9]+\.[0-9]+)--([0-9]+\.[0-9]+)',
                    cls.table[column][row])

                if range_match:

                    lo = float(range_match.group(1))
                    hi = float(range_match.group(2))
                    val = 0.5 * (lo + hi)
                    val_err = 0.5 * abs(hi - lo)

                else:
                    val, val_err = decode_LaTeX(cls.table[column][row])

                # Now set the appropriate attribute

                # setattr(self, label[i], val * unit[i])
                rowparam[label[i]] = val*unit[i]

                if val_err != None:
                # print (column, label[i]+'_err',val_err)
                    # setattr(self, label[i] + '_err', val_err * unit[i])
                    rowparam[label[i]+'_err'] = val_err*unit[i]
            else:
                # setattr(self, label[i], self.table[column][self.row] * unit[i])
                rowparam[label[i]] = cls.table[column][row] * unit[i]

        return cls.from_file(cls.table['file'][row], CONCORD_PATH, **rowparam)

    @classmethod
    def from_file(cls, filename, path='.', **kwargs):
        '''
        Method to read in a file and populate the relevant arrays to create
        an ObservedBurst
        This routine will read in any ascii lightcurve file which matches the
        format of the "reference" bursts:

        'Time [s]' 'dt [s]' 'flux [10^-9 erg/cm^2/s]' 'flux error [10^-9
        erg/cm^2/s]' 'blackbody temperature kT [keV]' 'kT error [keV]'
        'blackbody normalisation K_bb [(km/d_10kpc)^2]' 'K_bb error
        [(km/d_10kpc)^2]' chi-sq
            -1.750  0.500   1.63    0.054   1.865  0.108   13.891   4.793  0.917
            -1.250  0.500   2.88    1.005   1.862  0.246   22.220   4.443  1.034
            -0.750  0.500   4.38    1.107   1.902  0.093   30.247   2.943  1.089
            -0.250  0.500   6.57    0.463   1.909  0.080   46.936   6.969  0.849

        Calling approach:
        obs = ObservedBurst.from_file(filename)
        obs = ObservedBurst.from_file('gs1826-24_3.530h.dat','example_data')
        '''

        d = ascii.read(path+'/'+filename)

        return cls(d['col1'], d['col2'], d['col3']*1e-9, d['col4']*1e-9,
                   path=path, filename=filename, comments=d.meta['comments'],
                   **kwargs)

# ------- --------- --------- --------- --------- --------- --------- ---------

    def info(self):
        '''
        Display information about the ObservedBurst
        :return:
        '''

        print("\nObservedBurst parameters:")
        if hasattr(self,'tdel'):
            print ("  tdel = {:.4f}".format(self.tdel))
        fluen, fluen_err = self.fluence(warnings=False)
        print ("  Fluence = {:.3e} +/- {:.3e}".format(fluen.value, fluen_err))
        if hasattr(self,'fper') & hasattr(self,'fper_err'):
            print ("  F_per = {:.4e} +/- {:.4e}".format(self.fper,self.fper_err))
        # Simulated observed bursts won't have an fper_err attribute
        elif hasattr(self,'fper'):
            print ("  F_per = {:.4e} ".format(self.fper))
        if hasattr(self,'c_bol'):
            print ("  Bolometric correction = {}".format(self.c_bol))

        # Check if this is a simulated burst, and if so, list the parameters

        if hasattr(self,'sim_dist'):
          print ("this is a simulated burst, based on the following parameters:")
          print ("  distance = {:.4f}".format(self.sim_dist))
          print ("  inclination = {:.4f}".format(self.sim_inclination))
          print ("  disk model = {} giving xi_b = {:.4f}, xi_p = {:.4f}".format(self.sim_disc_model,
                 self.sim_xi_b, self.sim_xi_p))
          print ("  redshift = {:.4f}".format(self.sim_opz))

        # should also print the simulation lightcurve parameters, where available

        self.print()

# ------- --------- --------- --------- --------- --------- --------- ---------

    def compare(self, mburst, param = [6.1*u.kpc,60.*u.degree,1.26,-10.*u.s],
        		breakdown = False, plot = False, subplot = True,
                weights={'fluxwt':1.0, 'tdelwt':2.5e3},
                disc_model='he16_a', debug = False):
        '''
        This is the key method for running the mcmc; it can be used to plot the
        observations with the models rescaled by the appropriate parameters, and
        also returns a likelihood value

        'weights' give the relative weight to the tdel and persistent
        flux for the likelihood. Since you have many more points in the
        lightcurve, you may want to weight these greater than one so that the
        MCMC code will try to match those preferentially'''

        # Calculate the simulated burst using the observe method

        sim_burst = mburst.observe(param, obs=self, disc_model=disc_model, c_bol=self.c_bol)

        assert sim_burst.flux.unit == self.flux.unit == self.flux_err.unit

# Even though this is already in the prior, lhoodClass calls compare()
# before the prior, enabling an out-of-domain error in anisotropy
        dist, inclination, _opz, t_off = param
        if not 0. < inclination.value < 90.:
            return -np.inf

# can check here if the object to compare is actually a model burst
#        print (type(mburst))

# Here we assemble an array with the likelihood components from each
# parameter, for accounting purposes
# By convention we calculate the observed parameter (from the model) and
# then do the comparison
# Should probably incorporate these calculations into the class, so you
# can just refer to them as an attribute

        lhood_cpt = np.array([])

        # Persistent flux

        # fper_pred = fper(mburst.mdot,_opz,dist,xi_p,c_bol=self.cbol)
        fper_pred = sim_burst.fper

        if hasattr(self,'fper_err'):
            fper_sig2 = 1.0/(self.fper_err.value**2)
        else:
            print ('concord.compare: ** WARNING ** no uncertainty on F_per')
            fper_sig2 = 1.0
        lhood_cpt = np.append(lhood_cpt, -weights['fluxwt']*( 
               (self.fper.value-fper_pred.value)**2*fper_sig2 
               +np.log(2.*pi/fper_sig2) ) )

        # recurrence time

        # tdel_sig2 = 1.0/(self.tdel_err.value**2+(mburst.tdel_err.value*_opz)**2)
        # lhood_cpt = np.append(lhood_cpt, -weights['tdelwt']*(
        #        (self.tdel.value-mburst.tdel.value*_opz)**2*tdel_sig2
        #        +np.log(2.*pi/tdel_sig2) ) )
        tdel_sig2 = 1.0 / (self.tdel_err.value**2+sim_burst.tdel_err.value**2)
        lhood_cpt = np.append(lhood_cpt, -weights['tdelwt']*(
                (self.tdel.value-sim_burst.tdel.value)**2*tdel_sig2
                +np.log(2.*pi/tdel_sig2) ) )

        # lightcurve

        inv_sigma2 = 1.0/(self.flux_err.value**2)
        # lhood_cpt = np.append(lhood_cpt,
        # 	-0.5 * np.sum( (model.value-self.flux.value)**2*inv_sigma2
        #         +np.log(2.0*pi/inv_sigma2) ) )
        lhood_cpt = np.append(lhood_cpt,
         	-0.5 * np.sum( (sim_burst.flux.value-self.flux.value)**2*inv_sigma2
                 +np.log(2.0*pi/inv_sigma2) ) )

        if debug:
            cl=0.0
            for i in range(len(self.time)):
                _lhood = -0.5*((sim_burst.flux[i].value-self.flux[i].value)**2*inv_sigma2[i]
                    +np.log(2.0*pi/inv_sigma2[i]))
                cl += _lhood
                print ('{:6.2f} {:.4g} {:.4g} {:.4g} {:8.3f} {:8.3f}'.format(self.time[i],
                    self.flux[i].value, self.flux_err[i].value, sim_burst.flux[i].value,_lhood,cl))

        if breakdown:
            print ("Likelihood component breakdown (fper, tdel, lightcurve): ",lhood_cpt)

# Plot the observed burst

        if plot:

            # Now do a more complex plot with a subplot
            # See http://matplotlib.org/users/gridspec.html for documentation
            #
            # Want to maintain consistent colors; blue for observed, green for simulated?

            fig = plt.figure()
            gs = gridspec.GridSpec(4, 3)
            ax1 = fig.add_subplot(gs[0:3,:])

            self.plot(color='b')
        
# overplot the rescaled model burst

##        plt.plot(mburst.time, mburst.flux(dist))
#            plt.plot(self.time+(0.5-self.timepixr)*self.dt, model,'r-',
## This removes the problems with the underscore, but now introduces an
## extra backslash... argh
##		label=mburst.filename.replace('_',r'\_'))
#		label=mburst.filename)

            # ax1.plot(self.time+(0.5-self.timepixr)*self.dt, model,'r-',label=sim_burst.filename)
            sim_burst.plot(color='g')

            if subplot:

# Show the recurrence time comparison in the subplot; see 
# http://matplotlib.org/examples/pylab_examples/axes_demo.html for
# illustrative example

                # a = plt.axes([.55, .5, .3, .3], facecolor='y')
                a = plt.axes([.55, .5, .3, .3]) # yellow is kinda ugly

#                print (self.tdel,self.tdel_err)
                a.errorbar([0.95], self.tdel.value, 
                       yerr=self.tdel_err.value, fmt='o', color='b')
                a.errorbar([1.05], sim_burst.tdel.value,
                       yerr=sim_burst.tdel_err.value, fmt='o', color='g')
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
		self.flux.value-sim_burst.flux.value,
		yerr=self.flux_err.value,fmt='b.')
            ax2.axhline(0.0, linestyle='--', color='k')
            gs.update(hspace=0.0)
    
# Finally we return the sum of the likelihoods

        return lhood_cpt.sum()

# ======= ========= ========= ========= ========= ========= ========= =========

# Here's an example of a simulated model class

class KeplerBurst(Lightcurve):
    '''
    Example simulated burst class. Apart from the lightcurve (which is
    defined with time, lumin, and lumin_err columns), the additional
    (minimal) attributes requred are:

    filename - source file name
    tdel, tdel_err - recurrence time and error (hr)
    Lacc - accretion luminosity (in units of Mdot_Edd)
    R_NS - model-assumed radius of the neutron star (GR)
    g - surface gravity assumed for the run

    An example call is as follows:

    c_loZ=KeplerBurst(filename='mean1.data',path='kepler',
                  lAcc=0.1164,Z=0.005,H=0.7,
                  tdel=4.06/opz,tdel_err=0.17/opz,
                  g = 1.858e+14*u.cm/u.s**2, R_NS=11.2*u.km)

    If you provide a batch and run value, like so:

    c = KeplerBurst(batch=2,run=9)

    or (for the older results) a run_id value, like so:

    c = KeplerBurst(run_id='a028',path='../../burst/reference')

    the appropriate model lightcurve will be read in, and the model parameters
    will be populated from the table
    '''

    def __init__(self, filename=None, run_id=None, path=None, 
                 source='gs1826', grid_table=None, batch=None, run=None,
                 **kwargs):

        eta = 1e-6	# tolerance level for derived parameters

# This parameter sets the prefactor for the time to burn all H via hot-CNO

        tpref = 9.8*u.hr

# Setting the assumed Eddington limit here

        self.L_Edd = 3.53e38*u.erg/u.s

# Don't add the path to the filename, because we want to use the latter as
# a plot label (for example)

        if path == None:
            path = '.'
        self.path = path

# Different file specifications here

        if run_id is not None:

            # For a KEPLER run, we use the convention for filename as follows:

            self.filename = 'kepler_'+run_id+'_mean.txt'

        elif ((batch is not None) & (run is not None)):

            self.filename = source+"_{}_xrb{}_mean.data".format(batch,run)
            # temporarily hardwired the path
            # self.path = self.path+"/kepler_grids/sources/{}/mean_lightcurves/{}_{}".format(source,source,batch)
            self.path = "/Users/duncan/data/kepler_grids/sources/{}/mean_lightcurves/{}_{}".format(source,source,batch)

        elif filename is not None:

            self.filename = filename

        else:

            print ("** ERROR ** no valid model specification")
            return

        # Read in the file, and initialise the lightcurve

        d=ascii.read(self.path+'/'+self.filename)

        if ('time' in d.columns):
            Lightcurve.__init__(self, filename=self.filename, 
                            time=d['time']*u.s,
                            lumin=d['luminosity']*u.erg/u.s, lumin_err=d['u_luminosity']*u.erg/u.s)
        else:
            Lightcurve.__init__(self, filename=self.filename, 
                            time=d['col1']*u.s,
                            lumin=d['col2']*u.erg/u.s, lumin_err=d['col3']*u.erg/u.s)

        if ('comments' in d.meta):
            self.comments = d.meta['comments']

        if ((batch is not None) & (run is not None)) | (run_id is not None):

# ------- --------- --------- --------- --------- --------- --------- ---------
# For KEPLER models, read information from the burst table
# Need to set the recurrence time, NS mass

# Which table we use depends on the specification

            if ((batch != None) & (run != None)):

# For the newer models, read data only from the summary table, provided you
# can find it. Because this has moved around from version to version, here
# you can specify explicitly the name (assumed to be in the top-level
# "source" directory)

                if grid_table is not None:
                    self.summ_file = "../../"+grid_table
                else:
                    self.summ_file = "../../summ_{}.txt".format(source)

                self.data = ascii.read(self.path+"/"+self.summ_file)

                # Find the corresponding row

                self.row = np.where(np.logical_and(self.data['batch'] == batch,self.data['run'] == run))[0]

# Set some special parameters here (others are set with the kwargs later
# on). A couple of conventions for column names here

                self.tdel_colname = 'tDel'
                self.tdel_err_colname = 'uTDel'
                if not (self.tdel_colname in self.data.columns):
                    self.tdel_colname = 'dt'
                    self.tdel_err_colname = 'u_dt'

                self.tdel = self.data[self.tdel_colname][self.row][0]/3600.*u.hr
                self.tdel_err = self.data[self.tdel_err_colname][self.row][0]/3600.*u.hr

# For older models, you also need to read the companion parameter table
# Set the gravity and NS mass

                if not ('mass' in self.data.columns):
                    self.param = ascii.read(self.path+"/../../params_{}.txt".format(source))
                    self.row_p = np.where(np.logical_and(self.param['batch'] == batch,self.param['run'] == run))[0]

                    self.M_NS = self.param['mass'][self.row_p][0]*u.Msun
                else:
                    self.M_NS = self.data['mass'][self.row][0]*u.Msun

            else:

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

                except:

# But if that fails, we use the local version, which has a few differences
# (unfortunately)

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

# Set the parameters for the old Kepler models

# Additional parameter here gives the radius conversion between the assumed
# (Newtonian) value of 10 km and the GR equivalent for a neutron star of
# mass 1.4 M_sun, to achieve the same surface gravity.
# This is only true for the KEPLER models of Lampe et al. (2016);
# otherwise we calculate this parameter from the other inputs

                self.xi = 1.12

# Define the neutron star mass, radius, and surface gravity

                self.M_NS = 1.4*const.M_sun

# Parameters that are common to both generations of Kepler models

            self.R_Newt = 10.*u.km
#            self.g = const.G*self.M_NS/(self.R_NS**2/self.opz)
#            self.g = g(self.M_NS,self.R_NS)
            self.g = g(self.M_NS,self.R_Newt,Newt=True)
#            self.R_NS = self.R_Newt*self.xi
            self.R_NS = solve_radius(self.M_NS,self.R_Newt)
#            self.opz = 1./sqrt(1.-2.*const.G*self.M_NS/(const.c**2*self.R_NS))
            self.opz = redshift(self.M_NS,self.R_NS)
        
            # Set all the remaining attributes, with a few exceptions:
            # exclude the tDel, to avoid confusion with tdel (set earlier)
            # rename fluence here for consistency with ObservedBurst, and to avoid
            #   conflict with fluence method

            for attr in self.data.columns:
                if (attr == 'fluence'):
                    setattr(self,'fluen_table',self.data[attr][self.row][0])
                elif ((attr != 'tDel') & (attr != 'utDel')):
                    setattr(self,attr,self.data[attr][self.row][0])
            
# ------- --------- --------- --------- --------- --------- --------- ---------

        elif kwargs != None:

# For non-KEPLER models, you can use kwargs to populate the parameters

            for key in kwargs:
#                print (key, kwargs[key])
                if (key == 'tdel') | (key == 'tdel_err'):
                    setattr(self,key,float(kwargs[key])*u.hr)
                else:
                    setattr(self,key,kwargs[key])

# ------- --------- --------- --------- --------- --------- --------- ---------

# Should check here that you have all the required attributes
# Here we estimate the fraction of H burned before ignition, based on the
# expression from Lampe et al. 2016; this can be > 1.

        if (hasattr(self,'tdel') & hasattr(self,'z')):
            self.burn_frac = self.tdel/(self.opz*tpref)*(self.z/0.02)

# Make sure you can calculate g, and M_NS, and then you can calculate
# everything else from those, as well as checking for consistency with any
# passed values

        if (not hasattr(self,'g')):
            if (hasattr(self,'M_NS') & hasattr(self,'R_NS')): 
                self.g = g(self.M_NS,self.R_NS)
            elif (hasattr(self,'M_NS') & hasattr(self,'R_Newt')):
                self.g = g(self.M_NS,self.R_Newt,Newt=True)
            else:
                print ("** ERROR ** can't calculate g")

        if (not hasattr(self,'M_NS')):
            if (hasattr(self,'g') & hasattr(self,'R_NS')): 
# solving for M from the GR radius is a bit tricky...
                self.M_NS = (self.R_NS**3*self.g**2/(const.G*const.c**2)*
    (-1.+sqrt(1.+const.c**4/(self.R_NS*self.g)**2))).to(u.g)
            elif (hasattr(self,'g') & hasattr(self,'R_Newt')):
                self.M_NS = (self.g*R_Newt**2/const.G).to(u.g)
                if debug:
                    print ('Inferred mass = {:.4f} M_sun'.format(M_NS/const.M_sun))

# Here we calculate the redshift, and check for consistency with any
# already-set value

        if hasattr(self,'R_NS'):
            _opz = redshift(self.M_NS,self.R_NS)
            if hasattr(self,'opz'):
#                assert abs(_opz-self.opz)/_opz < eta, "Inconsistent value of opz, {} != {}".format(_opz,self.opz)
                if abs(_opz-self.opz)/_opz >= eta:
                  print ("Inconsistent value of opz, {:.4f} != {:.4f}".format(_opz,self.opz))
            else:
                self.opz = _opz

# ditto for xi (= ratio of R_NS/R_Newt)

        _xi = sqrt(self.opz)
        if hasattr(self,'xi'):
#            assert abs(_xi-self.xi)/_xi < eta, "Inconsistent value of xi, {} != {}".format(_xi,self.xi)
            if abs(_xi-self.xi)/_xi >= eta:
                print ("Inconsistent value of xi, {:.4f} != {:.4f}".format(_xi,self.xi))
        else:
            self.xi = _xi

# Compare method requires R_Newt, so calculate it here if it's not already present

        if (not hasattr(self,'R_Newt')):
            self.R_Newt = self.R_NS/self.xi

# Specifically, for the compare method, we need to know two of g, M_NS,
# R_Newt, R_NS

        if (not (hasattr(self,'g')) & hasattr(self,'M_NS') and 
            not (hasattr(self,'g')) & hasattr(self,'R_NS') and 
            not (hasattr(self,'g')) & hasattr(self,'R_Newt') and 
            not (hasattr(self,'M_NS')) & hasattr(self,'R_NS') and 
            not (hasattr(self,'M_NS')) & hasattr(self,'R_Newt')):
            print ("** WARNING ** insufficient parameters defined to convert to observed frame")

        # Some ambiguity with the attribute capitalisation for the accretion rate,
        # so try to fix that here. Also make this work with the more recent versions
        # of the tables

        if ((not hasattr(self,'Lacc')) & hasattr(self,'lAcc')):
            self.Lacc = self.lAcc
        if ((not hasattr(self,'Lacc')) & hasattr(self,'accrate')
            & hasattr(self,'acc_mult')):
            self.Lacc = self.accrate*self.acc_mult

# Set the mdot with the correct units

        if hasattr(self,'Lacc'):
            self.mdot = self.Lacc*1.75e-8*const.M_sun/u.yr

# Set the flag for super-Eddington bursts

        self.superEdd = (max(self.lumin) > self.L_Edd)

# - end of __init__ method -- --------- --------- --------- --------- ---------

    def info(self):
        '''
        Display information about the KeplerBurst
        :return:
        '''

        print("\nKeplerBurst parameters:")
        if hasattr(self,'tdel'):
            print ("  tdel = {:.3f}".format(self.tdel))
        if hasattr(self,'g'):
            print ("  g = {:.4e}".format(self.g))
        if hasattr(self,'R_Newt'):
            print ("  R_Newt = {:.3f}".format(self.R_Newt))
        if hasattr(self,'R_NS'):
            print ("  R_NS = {:.3f}".format(self.R_NS))
        if hasattr(self,'opz'):
            print ("  1+z = {:.3f}".format(self.opz))
        # This doesn't quite work yet
        # fluen, fluen_err = self.fluence(warnings=False)
        # print ("  Fluence = {:.3e} +/- {:.3e}".format(fluen.value, fluen_err))
        if hasattr(self,'fluen_table'):
            print ("  Fluence = {:.3e}".format(self.fluen_table))

        self.print()

# The flux method is supposed to calculate the flux at a particular distance
# Not used (and doesn't work)

#    def flux(self,dist):
#        if not hasattr(self,'dist'):
#            self.dist = dist
#
#        return self.lumin/(4.*pi*self.dist.to('cm')**2)

# ======= ========= ========= ========= ========= ========= ========= =========

# Now define a new likelihood function, based on the old one, but which
# can handle multiple pairs of observed bursts

# First define the prior

def lnprior(theta):
    dist, inclination, _opz, t_off = theta

# Have to enforce this here or the prior will come out wrong

    assert hasattr(inclination,'unit')
    assert inclination.unit == 'deg'

# We have currently flat priors for everything but the inclination, which
# has a probability distribution proportional to sin(i)

    if (dist.value > 0.0 and 0.0 < inclination.value < 90.
        and 1. < _opz < 2):
        return np.log(np.sin(inclination))
    else:
        return -np.inf

# ------- --------- --------- --------- --------- --------- --------- ---------

def apply_units(params,units = (u.kpc, u.degree, None, u.s)):

# When called from emcee, the parameters array might not have units. So
# apply them here, in a copy of params (uparams)

    ok = True
    uparams = []
    n_units = len(units)
    for i, param in enumerate(params):

# Define iunit here so we apply the last unit in the list, to the 4th (and
# all subsequent) element of params
# This is to cover the variable number of offset values for the burst
# start, which depends upon the number of bursts matching simultaneously

        iunit = min(i,n_units-1)

        if units[iunit] != None:
            if hasattr(param,'unit') == False:
                uparams.append(param*units[iunit])
            else:
                uparams.append(param)
                if param.unit != units[iunit]:
                    ok = False
        else:
            uparams.append(param)
            if hasattr(param,'unit') == True:
                ok = False
    assert(ok == True)

    return uparams

# ------- --------- --------- --------- --------- --------- --------- ---------

def lhoodClass(params, obs, model, **kwargs):
    '''
    Calculate the likelihood related to one or more model-observation
    comparisons The corresponding call to emcee will (necessarily) look
    something like this:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lhoodClass, args=[obs, models, weights])

    Use the kwargs construction to pass the additional parameters weights,
    disc_model to the compare method if required
    '''

    uparams = apply_units(params)
    # print ('uparams = {}'.format(uparams))

# We can pass multiple bursts, in which case we just loop over the function

    alh = 0.0

    if type(obs) == tuple:
        n = len(obs)
        if (n != len(model)):
            print ("** ERROR ** number of observed and model bursts don't match")

        for i in range(n):

# Need to create a reduced parameter array here, keeping only the offset
# value appropriate for this burst

            _params = uparams[:3]
            # print (i,uparams,_params)
            _params.append(uparams[3+i])
            alh += lhoodClass(_params, obs[i], model[i], **kwargs)

    else:

# Or if we have just one burst, here's what we do

        alh = obs.compare(mburst=model, param=uparams, **kwargs)

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
    #
    b1, b2, b3 = obs
    m1, m2, m3 = models

#    b1, b2 = obs
#    m1, m2 = models

# Can't use the gridspec anymore, as this is used for the individual plots

#    fig = plt.figure()
#    gs = gridspec.GridSpec(3,2)
    #TODO make this iteratable and generalised to different numbers of epoch
# plot the model comparisons. Should really do a loop here, but not sure
# exactly how

    _param_best = param_best[0:4]

    b1.compare(m1,_param_best,plot=True,subplot=False)
#    fig.set_size_inches(8,3)

    _param_best = param_best[0:3]
    _param_best.append(param_best[4])
    b2.compare(m2,_param_best,plot=True,subplot=False)

    _param_best = param_best[0:3]
    _param_best.append(param_best[5])
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

    fig = plt.figure()
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o')
    plt.plot([3,6],[3,6],'--')
    plt.xlabel('Observed $\Delta t$ (hr)')
    plt.ylabel('Predicted $(1+z)\Delta t$ (hr)')

# ------- --------- --------- --------- --------- --------- --------- ---------

def plot_contours(sampler,parameters=[r"$d$",r"$i$",r"$1+z$"],
        ignore=10,plot_size=6):
    '''
    Simple routine to plot contours of the walkers, ignoring some initial
    fraction of the steps (the "burn-in" phase)

    Documentation is here https://samreay.github.io/ChainConsumer/index.html
    '''

    nwalkers, nsteps, ndim = np.shape(sampler.chain)

    samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
#    samples = np.load('temp/chain_200.npy').reshape((-1,ndim))
#    print (np.shape(samples))

# This to produce a much more beautiful plot

    c = ChainConsumer()
    c.add_chain(samples, parameters = parameters)#,r"$\Delta t$"])

# reports,
# "This method is deprecated. Please use chainConsumer.plotter.plot instead"

    fig = c.plot()
    fig.set_size_inches(8,8)

    return c

# end of file burstclass.py
