SA-GPR
======

This repository contains a Python code for carrying out Symmetry-Adapted Gaussian Process Regression (SA-GPR) for the machine-learning of tensors. For more information, see:

<https://arxiv.org/abs/1709.06757>

Requirements
============

The Python packages scipy and sympy are required to run the SA-GPR code.

Installation
============

To install the program, go to the src/ directory and run,

bash install.sh

This will compile the Fortran code for filling power spectra using f2py.

Workflow
========

There are two steps to applying SA-GPR to a physical problem:

1. Calculation of the similarities (kernels) between molecules/bulk systems.
2. Minimization of the prediction error and generation of the weights for prediction.

These two steps are applied by running two different Python scripts: sa-gpr-kernels.py computes the kernels (step 1) and sa-gpr-apply.py carried out the regression (step 2).

Examples
========

The example/ directory contains four sub-directories, with data for the dielectric tensors of water monomers, water dimers, the Zundel cation and boxes of 32 bulk water molecules. Using the water monomer, Zundel cation and bulk water directories, we illustrate three examples of how to use SA-GPR.

Before starting, source the environment settings file :code `env.sh` using :code `$ source env.sh`.

1. Water Monomer
----------------

Here, we learn the energy of the water monomer. The energy only has a scalar (L=0) component, and learning it is equivalent to the standard SOAP algorithm. We begin by computing the L=0 kernels between the 1000 molecules:

::

  $ cd example/water_monomers
  $ sa-gpr-kernels.py -lval 0 -f coords_1000.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8

This will create an L=0 kernel file, using the coordinates in coords_1000.in, with Gaussian width 0.3 Angstrom, an angular cutoff of l=6, a radial cutoff of 4 Angstrom, central atom weighting of 1.0, and with centering on oxygen atoms (atomic number 8). The kernel file, :code `kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt`, can now be used to perform the regression:

::

  $ sa-gpr-apply.py -r 0 -k 0 kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -t energy_1000.in -lm 0 1e-8

The regression is performed for a rank-0 tensor, using the kernel file we produced, with a training set containing 200 randomly selected configurations, of which all are used for training. The file :code `energy_1000.in` contains the energies of the 1000 coordinates, and we use a regularization of 1e-8. By varying the value of the :code `ftr` variable, it is possible to create a learning curve.

2. Zundel Cation
----------------

(a) Learning all components of the hyperpolarizability tensor at once

Here, we learn the hyperpolarizabilities of the Zundel cation. Because the calculation of the full kernel matrix is quite expensive, we will split this problem up into the calculation of several sub-blocks of the matrix. In example/zundel, run:

(b) Learning the components separately

3. Bulk water
-------------

(a) Learning of the polarizability

(b) Learning curves

Contact
=======

david.wilkins@epfl.ch
andrea.grisafi@epfl.ch
