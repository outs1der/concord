# README #

This repository contains code intended to simplify the analysis of astronomical X-ray data of thermonuclear (type-I) bursts.

To get started, look at the accompanying paper (in the `doc/` folder), or the `Burst matching tutorial` jupyter notebook (in the `notebooks/` folder)

### What is this repository for? ###

* Analysis of thermonuclear X-ray bursts
* Reading in and plotting thermonuclear burst data and models
* Performing model-observation comparisons

### How do I get set up? ###

* Clone the repository and install using pip:
```
python3 -m pip install .
```
* import concord
* Check the Burst matching tutorial notebook for demonstrations of usage
* You'll also need to download some reference bursts from http://burst.sci.monash.edu/reference, and some model bursts from http://burst.sci.monash.edu/kepler. Or, you can define routines to read in your own data

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Duncan.Galloway@monash.edu

### Why concord? ###

Because we're trying to achieve a "concordance" fit to a wide range of burst data. Plus, concord was cool
