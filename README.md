SA-GPR
------

This repository contains a Python code for carrying out Symmetry-Adapted Gaussian Process Regression (SA-GPR) for the machine-learning of tensors. For more information, see:

<<INSERT LINK TO ARXIV HERE>>

Requirements
------------

<<HERE WE SHOULD SAY WHAT THE REQUIREMENTS ARE FOR THIS CODE>>

Installation
------------

To install the program, go to the src/ directory and run,

bash install.sh

This will compile the Fortran code for filling power spectra using f2py.

Workflow
--------

There are two steps to applying SA-GPR to a physical problem:

1. Calculation of the similarities (kernels) between molecules/bulk systems.
2. Minimization of the prediction error and generation of the weights for prediction.

These two steps are applied by running two different Python scripts: sa-gpr-kernels.py computes the kernels (step 1) and sa-gpr-apply.py carried out the regression (step 2).

Examples
--------

The example/ directory contains four sub-directories, for machine-learning of the properties of water monomers, water dimers, the Zundel cation and boxes of 32 bulk water molecules. We use these directories to illustrate some examples of how to use SA-GPR.

<<MENTION SETTING ENVIRONMENT VARIABLES BEFORE RUNNING>>

1. Water Monomers

<<HERE HAVE A BASIC EXAMPLE FOR WATER DIMERS>>

2. Water Dimers

Here, we learn the polarizabilities of the water dimer. Because the calculation of the full kernel matrix is quite expensive, we will split this problem up into the calculation of several sub-blocks of the matrix. In example/water_dimer, run:

src/scripts/mkblocks_nocell.sh coords_1000.in 100

This will create 55 "Block" folders, each of which contains a subset of the full configurations. For each of these subfolders, the kernels are calculated using:

python src/sa-gpr-kernels.py -lval 0 -f coords.in -f coords.in -f coords.in -per 1 -cen 8 -rc 5.0
python src/sa-gpr-kernels.py -lval 2 -f coords.in -f coords.in -f coords.in -per 1 -cen 8 -rc 5.0

These commands create the L=0 and L=2 kernels, respectively. To form the full kernels from these files, we use the command:

python src/scripts/rebuild_kernel0.py <<COMMAND LINE ARGUMENTS>>
python src/scripts/rebuild_kernel2.py <<COMMAND LINE ARGUMENTS>>

This will create the files kernel0_1000_sigma0.3_lcut6_cutoff5.0_cweight1.0.txt and kernel2_1000_sigma0.3_lcut6_cutoff5.0_cweight1.0.txt. The final step is to use these kernels to perform the regression. For this, we use:

python src/sa-gpr-apply.py -k 0 kernel0_1000_sigma0.3_lcut6_cutoff5.0_cweight1.0.txt 2 kernel2_1000_sigma0.3_lcut6_cutoff5.0_cweight1.0.txt -sel 0 500 -ftr 1.0 -t alpha_1000.in -lm 0 1e-3 2 1e-3 -r 2

3. Zundel Cation

<<EXAMPLE FOR ZUNDEL CATION>>

4. Bulk Water

<<BULK WATER EXAMPLE>>

Contact
-------

<<PUT SOMEONE'S CONTACT DETAILS HERE>>
