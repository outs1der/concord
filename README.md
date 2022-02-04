# README #

This repository contains code intended to simplify the analysis of astronomical X-ray data of thermonuclear (type-I) bursts.

Full documentation can be found at https://burst.sci.monash.edu/concord/

This code is under development, and a companion paper is expected to be submitted in early 2022

To get started, look at the  `Inferring burster properties` jupyter notebook

### What is this repository for? ###

* Analysis of thermonuclear X-ray bursts
* Reading in and plotting thermonuclear burst data and models
* Performing model-observation comparisons

### How do I get set up? ###
 
Use the included `environment.yml` file to set up a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment with the required dependencies:

```
conda env create -f environment.yml
```

This will create an environment called `concord`, which you can activate with: 

```
conda activate concord
```

Then add concord to the local environment with:
```
python3 -m pip install .
```

You can then import the repository and use the functions. Here's a very simple example, to find the peak luminosity of a
burst from 4U 0513+40 measured by RXTE, as part of the MINBAR sample. The first part calculates the isotropic luminosity, neglecting the uncertainty in both the peak flux and the distance:
```
>>> import concord as cd
>>> import astropy.units as u
>>> F_pk, e_F_pk = 21.72, 0.6 # 1E-9 erg/cm^2/s bolometric; MINBAR #3443
>>> d = (10.32, 0.24, 0.20) # asymmetric errors from Watkins et al. 2015
>>> l_iso = cd.luminosity( F_pk, dist=d[0], isotropic=True )
WARNING:homogenize_params:no bolometric correction applied
>>> print (l_iso)
2.767771097997098e+38 erg / s
```
The second part takes into account both the uncertainties in the peak flux and distance (including the asymmetric errors), and also includes the model-predicted effect of the high system inclination (>80 degrees):
```
>>> l_asym = cd.luminosity( (F_pk, e_F_pk), dist=d, burst=True, imin=80, imax=90, fulldist=True)
WARNING:homogenize_params:no bolometric correction applied
>>> lc = l_asym['lum'].pdf_percentiles([50, 50 - cd.CONF / 2, 50 + cd.CONF / 2]) 
>>> l_unit = 1e38*u.erg/u.s
>>> print ('''\nIsotropic luminosity is {:.2f}e38 erg/s
...   Taking into account anisotropy, ({:.2f}-{:.2f}+{:.2f})e38 erg/s'''.format(l_iso/l_unit, 
...                                                                             lc[0]/l_unit, 
...                                                 (lc[0]-lc[1])/l_unit, (lc[2]-lc[0])/l_unit))
Isotropic luminosity is 2.77e38 erg/s
  Taking into account anisotropy, (4.85-0.43+0.51)e38 erg/s
```
With the `fulldist=True` option, the function returns a dictionary with the Monte-Carlo generated distribution of the result (key `lum`) and all the intermediate quantities. Check the `Inferring burster properties` notebook for additional demonstrations of usage

There are a number of sources for bursts to analyse:
* The Multi-INstrument Burst ARchive (MINBAR) contains more than 7000 events from 85 sources, and you can download the entire dataset and a Python repository to access them from http://burst.sci.monash.edu and links therein
* The reference burst sample including lightcurves and recurrence times from http://burst.sci.monash.edu/reference
* Kepler-predicted model bursts from http://burst.sci.monash.edu/kepler

Or, you can define routines to read in your own model predictions, for example based on the `KeplerBurst` class

### Who do I talk to? ###

* Duncan.Galloway@monash.edu

### Why concord? ###

Because we're trying to achieve a "concordance" fit to a wide range of burst data. Plus, concord was cool
