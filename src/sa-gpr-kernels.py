#!/usr/bin/env python2

from __future__ import print_function
from builtins import range
import utils.kernels
import utils.kern_utils
import utils.parsing
import argparse
import sys
import numpy as np
from itertools import product

# THIS IS A WRAPPER THAT CALLS PYTHON SCRIPTS TO BUILD L-SOAP KERNELS.

# Get command-line arguments.
args = utils.parsing.add_command_line_arguments_tenskernel("Tensorial kernel")
[ftrs,npoints,lval,sg,lc,rcut,cweight,vrb,centers,nlist,atomic,extrap,ntest] = utils.parsing.set_variable_values_tenskernel(args)

print("""
NUMBER OF CONFIGURATIONS = {np}"
Building the symmetry adapted SOAP kernel for L = {lv}"
"
Kernel hyper-parameters:"
---------------------------"
Gaussian width = {sg}"
Angular cutoff = {lc}"
Environment cutoff = {rcut}"
Central atom weight = {cw}"
""".format(np=npoints, lv =lval, sg = sg, lc=lc, rcut=rcut, cw=cweight))

# Build kernels.
if extrap == False:

    [centers,atom_indexes,natmax,nat,kernels] = utils.kernels.build_kernels(lval,ftrs,npoints,sg,lc,rcut,cweight,vrb,centers,nlist)

    # Transformation matrices to real spherical harmonics.
    CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
    CC = np.conj(CR)
    CT = np.transpose(CR)

    if atomic:

        # Transform local kernels to real
        kloc = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1),dtype=float)
        for i in range(npoints):
            for j in range(npoints):
                for ii in range(nat[i]):
                   for jj in range(nat[j]):
                       kloc[i,j,ii,jj] = np.real(np.dot(np.dot(CC,kernels[0][i,j,ii,jj]),CT))

        # Get indexes for atoms of the same type
        iat = 0
        atom_idx = {}
        for k in centers:
            atom_idx[k] = []
            for il in atom_indexes[0][k]:
                atom_idx[k].append(iat)
                iat+=1

        # Build kernels of identical atoms
        katomic = {}
        natspe = {}
        ispe = 0
        for k in centers:
            natspe[ispe] = len(atom_idx[k])
            katomic[ispe] = np.zeros((natspe[ispe]*npoints,natspe[ispe]*npoints,2*lval+1,2*lval+1),float)
            irow = 0
            for i in range(npoints):
                for ii in atom_idx[k]:
                    icol = 0
                    for j in range(npoints):
                        for jj in atom_idx[k]:
                            katomic[ispe][irow,icol] = kloc[i,j,ii,jj]
                            icol+=1
                    irow+=1
            ispe += 1

        # Print out kernels
        envfile = []
        for k in centers:
            filename = "kernel{lv}_atom{k}_nconf{np}_sigma{sg}_lcut{lc}_cutoff{rc}_cweight{cw}.npy".format(np=npoints,k=k, lv =lval, sg = sg, lc=lc, rcut=rcut, cw=cweight)
            envfile.append(open(filename,"w"))
        nspecies = len(centers)
        for ispe in range(nspecies):
            np.save(envfile[ispe],katomic[ispe])
            #for i in range(natspe[ispe]*npoints):
            #    for j in range(natspe[ispe]*npoints):
            #        for iim,jjm in product(range(2*lval+1),range(2*lval+1)):
            #            print >> envfile[ispe], katomic[ispe][i,j,iim,jjm]

    else:
        for n in range(len(nlist)):
            # Transform kernel to real
            kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=float)
            for i,j in product(range(npoints),range(npoints)):
                kernel[i,j] = np.real(np.dot(np.dot(CC,kernels[1+n][i,j]),CT))
            # Save kernel.
            kernel_file = "kernel{lv}_{np}_sigma{sg}_lcut{lc}_cutoff{rcut}_cweight{cw}_n{n}.npy".format(lv=lval, np=npoints, sg = sg,lc = lc, rcut = rcut, cw = cweight, n =nlist[n])
            np.save(kernel_file,kernel)
            #kernfile = open(kernel_file,"w")
            #for i in range(npoints):
            #    for j in range(npoints):
            #        for iim,jjm in product(range(2*lval+1),range(2*lval+1)):
            #            print >> kernfile, kernel[i,j,iim,jjm]
            #kernfile.close()

else:

    [centers,atom_indexes,natmax,nat,kernels] = utils.extra_kernels.build_kernels(lval,ftrs,npoints,sg,lc,rcut,cweight,vrb,centers,nlist,ntest)

    # Transformation matrices to real spherical harmonics.
    CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
    CC = np.conj(CR)
    CT = np.transpose(CR)

    if atomic:

        # Transform local kernels to real
        kloc = np.zeros((ntest,npoints-ntest,natmax,natmax,2*lval+1,2*lval+1),dtype=float)
        for i in range(ntest):
            for j in range(npoints-ntest):
                for ii in range(nat[i]):
                   for jj in range(nat[j]):
                    kloc[i,j,ii,jj] = np.real(np.dot(np.dot(CC,kernels[0][i,j,ii,jj]),CT))

        # Get indexes for atoms of the same type

        # TESTING MOLECULES BEING AT THE HEAD OF THE LIST
        iat = 0
        atom_idx_test = {}
        for k in centers:
            atom_idx_test[k] = []
            for il in atom_indexes[0][k]:
                atom_idx_test[k].append(iat)
                iat+=1

        # TRAINING MOLECULES BEING AT THE TAIL OF THE LIST
        iat = 0
        atom_idx_train= {}
        for k in centers:
            atom_idx_train[k] = []
            for il in atom_indexes[-1][k]:
                atom_idx_train[k].append(iat)
                iat+=1

        # Build kernels of identical atoms
        katomic = {}
        natspe_train = {}
        natspe_test  = {}
        ispe = 0
        for k in centers:
            natspe_train[ispe] = len(atom_idx_train[k])
            natspe_test[ispe]  = len(atom_idx_test[k])
            katomic[ispe] = np.zeros((natspe_test[ispe]*ntest,natspe_train[ispe]*(npoints-ntest),2*lval+1,2*lval+1),float)
            irow = 0
            for i in range(ntest):
                for ii in atom_idx_test[k]:
                    icol = 0
                    for j in range(npoints-ntest):
                        for jj in atom_idx_train[k]:
                            katomic[ispe][irow,icol] = kloc[i,j,ii,jj]
                            icol+=1
                    irow+=1
            ispe += 1

        # Print out kernels
        envfile = []
        for k in centers:
            filename = "kernel{lv}_atom{k}_ntest{nt}_ntrain{ntr}_sigma{sg}_lcut{lc}_cutoff{rc}_cweight{cw}.npy".format(nt=ntest, ntr =(npoints-ntest), nk=k, lv =lval, sg = sg, lc=lc, rcut=rcut, cw=cweight)
            envfile.append(open(filename,"w"))
        nspecies = len(centers)
        for ispe in range(nspecies):
            np.save(envfile[ispe],katomic[ispe])
    #        for i in range(natspe[ispe]*npoints):
    #            for j in range(natspe[ispe]*npoints):
    #                for iim,jjm in product(range(2*lval+1),range(2*lval+1)):
    #                    print >> envfile[ispe], katomic[ispe][i,j,iim,jjm]

    else:
        for n in range(len(nlist)):
            # Transform kernel to real
            kernel = np.zeros((ntest,npoints-ntest,2*lval+1,2*lval+1),dtype=float)
            for i,j in product(range(ntest),range(npoints-ntest)):
                kernel[i,j] = np.real(np.dot(np.dot(CC,kernels[1+n][i,j]),CT))
            # Save kernel.
            kernel_file = "kernel{lv}_ntest{nt}_ntrain{ntr}_sigma{sg}_lcut{lc}_cutoff{rcut}_cweight{cw}.npy".format(lv=lval, nt=ntest, ntr =(npoints-ntest),  sg = sg,lc = lc, rcut = rcut, cw = cweight)
            np.save(kernel_file,kerne)
            #kernfile = open(kernel_file,"w")
            #for i in range(npoints):
            #    for j in range(npoints):
            #        for iim,jjm in product(range(2*lval+1),range(2*lval+1)):
            #            print >> kernfile, kernel[i,j,iim,jjm]
            #kernfile.close()
