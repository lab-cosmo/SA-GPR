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

Before starting, source the environment settings file :code:`env.sh` using :code:`$ source env.sh`.

1. Water Monomer
----------------

Here, we learn the energy of the water monomer. The energy only has a scalar (L=0) component, and learning it is equivalent to the standard SOAP algorithm. We begin by computing the L=0 kernels between the 1000 molecules:

::

  $ cd example/water_monomers
  $ sa-gpr-kernels.py -lval 0 -f coords_1000.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8

This will create an L=0 kernel file, using the coordinates in coords_1000.in, with Gaussian width 0.3 Angstrom, an angular cutoff of l=6, a radial cutoff of 4 Angstrom, central atom weighting of 1.0, and with centering the environment on oxygen atoms (atomic number 8). The kernel file, :code:`kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt`, can now be used to perform the regression:

::

  $ sa-gpr-apply.py -r 0 -k 0 kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -t energy_1000.in -lm 0 1e-8

The regression is performed for a rank-0 tensor, using the kernel file we produced, with a training set containing 200 randomly selected configurations, of which all are used for training. The file :code:`energy_1000.in` contains the energies of the 1000 coordinates, and we use a regularization parameter of 1e-8. By varying the value of the :code:`ftr` variable, from 0 to 1, it is possible to create a learning curve from the case of no-example to the full training data set.

2. Zundel Cation
----------------

**Learning all components of the hyperpolarizability tensor at once**

Here, we learn the hyperpolarizabilities of the Zundel cation. Because the calculation of the full kernel matrix is quite expensive, we will split this problem up into the calculation of several sub-blocks of the matrix. In :code:`example/water_zundel`, run:

::

  $ mkblocks_nocell.sh coords_1000.in 100

This will create 55 `Block` folders, each of which contains a subset of the coordinates. In each of these folders, run the commands:

::

  $ sa-gpr-kernels.py -lval 1 -f coords.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8
  $ sa-gpr-kernels.py -lval 3 -f coords.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8

This will create two kernel files in each folder, one for L=1 and one for L=3 (a symmetric, rank-3 hyperpolarizability tensor can be split up into these two components). Having created these sub-kernels, the next step is to put these back together into a full kernel tensor. To do this, run:

::

  $ rebuild_kernel.py -l 1 -ns 1000 -nb 10 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0
  $ rebuild_kernel.py -l 3 -ns 1000 -nb 10 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0

This will produce two files, :code:`kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt` and :code:`kernel3_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt`. The Block subfolders can now be deleted. These kernels can be used to perform the regression:

::

  $ sa-gpr-apply.py -r 3 -k 1 kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt 3 kernel3_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -t beta_1000.in -lm 1 1e-3 3 1e-3

This command is similar to the one used to perform the regression on the water monomer, except that now we specify a rank-3 tensor, and give as input two kernels (one with L=1 and one with L=3), and two regularization parameters.

**Learning the dipole moment**

The dipole moment is an L=1 tensor, and so the kernel we have already calculated allows us to learn this tensor "for free":

::

  $ sa-gpr-apply.py -r 1 -k 1 kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -t dipole_1000.in -lm 1 1e-3

Users are encouraged to experiment with the size of the training set and the regularization parameter. In the example on bulk water, we will show how to produce a learning curve.

**Learning the hyperpolarizability components separately**

Instead of learning the L=1 and L=3 components of the hyperpolarizability at the same time, we might want to learn them separately. For this, we first need to split the tensor into its spherical components:

::

  $ cartesian_to_spherical.py -f beta_1000.in

This will produce two files, :code:`beta_1000.in:L1` and :code:`beta_1000.in:L3`, which are the L=1 and L=3 (real) spherical components respectively. To perform regression on the L=1 component, run the command:

::

  $ regression.py -k kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -t beta_1000.in.L1 -rdm 200 -nc 5 -ftr 1.0 -lm 1e-6 -o output.out 

To perform regression on the L=3 component, run the command:

::

  $ regression.py -k kernel3_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -t beta_1000.in.L3 -rdm 200 -nc 5 -ftr 1.0 -lm 1e-6 -o output.out 

In these examples, we loop over 5 random selections of the training set. There will be 5 output files printed out, each of which gives the members of the training set for this selection, along with the regression errors and the SA-GPR weights.

3. Bulk water
-------------

Here we consider the case of liquid water as an example of a condansed-phase (infinite) system. First of all move to the example directory: 

::

  $ cd example/water_bulk/

The files :code:`coords_1000.in` and :code:`cell_1000.in` contain respectively the coordinates and the cell vectors of 1000 structures represented by 32 water molecules in a periodic box.
The example also contains two kind of properties associated to those structures, namely, the infinite-frequency static dielectric response tensors (:code:`epsilon_1000.in` ) and an effective representation of the molecular polarizabilities (:code:`alpha_1000.in`) obtained by applying the Clausius-Mossotti relationship of the dielectric tensors.

**Learning the dielectric tensor**

The dielectric response of the system is represented by a rank-2 tensor having an irreducible representation which involves the L=0 and L=2 spherical components. To compute the corresponding tensorial kernels we can follow almost the same procedure of the Zundel cation. First of all we need to split the data set in several tiny sub-blocks. To split into blocks of dimension 10, you can run:

::

  $ mkblocks_cell.sh coords_1000.in cell_1000.in 10

Then, in each of the `Block` folders generated, run the command for calculation of the kernels:

::

  $ sa-gpr-kernels.py -lval 0 -f coords.in -per -c cell.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8
  $ sa-gpr-kernels.py -lval 2 -f coords.in -per -c cell.in -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen 8

This time, :code:`-per` is needed to specify that you are dealing with a periodic system, together with the file of cell vectors. 

Finally, the kernel reconstruction and the regression procedure is identical to what explained in the case of the Zundel cation. 


**Learning curves**



Contact
=======

david.wilkins@epfl.ch

andrea.grisafi@epfl.ch
