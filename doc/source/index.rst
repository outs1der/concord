.. concord documentation master file, created by
   sphinx-quickstart on Tue Nov 16 11:59:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to concord's documentation!
===================================

``concord`` implements some commonly-used approaches for determining system
parameters for neutron stars from thermonuclear bursts,
and fully accounting for astrophysical uncertainties.
The code includes the capability for quantitative comparison of observed burst
properties with the predictions of numerical models, 
and the tools presented here are intended
to make comprehensive model-observation comparisons straightforward.

This code is under active development, but the `v1.0.0 release <https://bridges.monash.edu/articles/software/concord_release_1/21287616>`_ is
associated with a companion paper accepted by Astrophysical Journal
Supplements (see `Galloway et al. 2022 <https://iopscience.iop.org/article/10.3847/1538-4365/ac98c9>`_, also available at
`arXiv:2210.03598 <https://arxiv.org/abs/2210.03598>`_). A preprint of the
paper is available in the `doc` subdirectory of the repository

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Getting started
===============

Clone the repository and install using pip::

    python3 -m pip install

You can then import the repository and use the functions. Here's a very simple example, to find the peak luminosity of a
burst from 4U 0513+40 measured by RXTE, as part of the MINBAR sample. The
first part calculates the isotropic luminosity, neglecting the uncertainty
in both the peak flux and the distance::

    import concord as cd
    import astropy.units as u
    F_pk, e_F_pk = 21.72, 0.6 # 1E-9 erg/cm^2/s bolometric; MINBAR #3443
    d = (10.32, 0.24, 0.20) # asymmetric errors from Watkins et al. 2015
    l_iso = cd.luminosity( F_pk, dist=d[0], isotropic=True )
    print (l_iso)
    # 2.767771097997098e+38 erg / s

The second part takes into account both the uncertainties in the peak flux
and distance (including the asymmetric errors), and also includes the
model-predicted effect of the high system inclination (>80 degrees)::

    l_asym = cd.luminosity( (F_pk, e_F_pk), dist=d, burst=True, imin=80, imax=90, fulldist=True)
    lc = l_asym['lum'].pdf_percentiles([50, 50 - cd.CONF / 2, 50 + cd.CONF / 2]) 
    l_unit = 1e38*u.erg/u.s
    print ('''\nIsotropic luminosity is {:.2f}e38 erg/s
      Taking into account anisotropy, ({:.2f}-{:.2f}+{:.2f})e38 erg/s'''.format(l_iso/l_unit, lc[0]/l_unit, (lc[0]-lc[1])/l_unit, (lc[2]-lc[0])/l_unit))
    # Isotropic luminosity is 2.77e38 erg/s
    #   Taking into account anisotropy, (4.85-0.43+0.51)e38 erg/s

With the `fulldist=True` option, the function returns a dictionary with
the Monte-Carlo generated distribution of the result (key `lum`) and all
the intermediate quantities. 

Check the ``Inferring burster properties`` notebook, which includes the 
examples used in the paper, for additional demonstrations of usage.

Obtaining burst data
====================

You'll also need some burst measurements or data; at their simplest, most
of the functions operate on measurable quantities (with uncertainties),
like the peak burst flux, burst fluence, persistent flux etc.

You can access data from over 7000 bursts from 85 sources via MINBAR, by
also installing the Python repository from
https://github.com/outs1der/minbar.

You can also use your own time-resolved spectroscopic data to generate an 
:py:class:`concord.burstclass.ObservedBurst` object, and then run the
functions on that.  You might need to define routines to read in your own
data, depending on the format.

You can download some reference bursts from
http://burst.sci.monash.edu/reference, and put the data (including the
``.dat`` files and the ``table2.tex`` file) in the ``concord/data``
directory.

You can use the example :py:class:`concord.burstclass.KeplerBurst` class
to read in some model bursts from
http://burst.sci.monash.edu/kepler.

Alternatively you can use that class as an example to write your own
appropriate for your model results.

How the functions work
======================

.. copied from the paper

There are three principal components; the functions in ``utils.py``, the
anisotropy treatment in ``diskmodel.py`` and the observed and model
burst classes in ``burstclass.py``.

The functions in ``utils.py`` provide the basic functionality as
described in `Functions`_. In order to treat the uncertainties in
the observed quantities, we adopt a Monte-Carlo (MC) approach via the 
`astropy <https://www.astropy.org>`_ 
`Distribution <https://docs.astropy.org/en/stable/uncertainty>`_ package.
Input values (including measured
quantities) can be provided in four different ways: 

- scalar
- value with symmetric error, represented as a tuple ``(value, error)``
- value with asymmetric error, as a 3-element tuple ``(value, lo_error, hi_error)``
- an arbitrary array.

Quantities with error estimates will be converted to a ``Distribution``
object of user-defined size, and with a symmetric or asymmetric normal
distribution. In this way we can represent a wide range of probability
distribution functions (PDFs) for the input parameters. These 
``Distribution`` objects can then be used in calculations as for scalars,
hence providing uncertainty propagation at the cost of additional
computation. 

Input values can be provided with units, or if units are absent, will be
assumed to have the standard units for `MINBAR
<http://burst.sci.monash.edu/minbar>`_ quantities;
for flux, :math:`10^{-9}\ \rm{erg\,cm^{-2}\,s^{-1}}`,
burst fluence :math:`10^{-6}\ \rm{erg\,cm^{-2}}`,
recurrence time hr, and so on.

All functions can operate assuming isotropic distributions for the burst
or persistent emission (``isotropic=True``), but by default will include
the possible effects of anisotropic emission via the ``diskmodel.py``
routine. This routine incorporates the modelling of `He & Keek (2016, ApJ
819, #47) <http://adsabs.harvard.edu/abs/2016ApJ...819...47H>`_, via ASCII
tables provided with the code.  

There are three options for the treatment
of emission anisotropy:

* ``isotropic=True``  - assumes both the burst and persistent emission is isotropic (:math:`\xi_b=\xi_p=1`)
* ``isotropic=False`` - incorporates anisotropy, by drawing a set of inclination values and calculating the H-fraction for each value. Inclinations are uniform on the sphere (up to a maximum value of :math:`i=72^\circ`, based on the absence of dips) and the value returned is the mean and 1-sigma error, or the complete distributions (with the fulldist option)
* ``inclination=<i>`` - calculate for a specific value of inclination

Where observed or model burst lightcurves are available, the classes in
``burstclass.py`` provide a way to represent those observations and
perform various standard analyses, including observation-model
comparisons. The ``ObservedBurst`` class can be instantiated from an
ASCII file giving the burst flux as a function of time, or in a number of
other ways.

An example ``KeplerBurst`` model burst class is provided, which offers a
number of ways of reading in Kepler burst runs, and which can be
adapted to outputs of different codes.

The reference bursts provided by `Galloway et al. (2017, PASA 34, e109)
<http://adsabs.harvard.edu/abs/2017PASA...34...19G>`_ can be read
in directly, provided the data is included in the ``data`` directory.

Functions
=========

.. automodule:: concord.utils
   :members:

Classes
=======

.. automodule:: concord.burstclass
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
