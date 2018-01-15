#!/usr/bin/env python2

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
[ftrs,npoints,lval,sg,lc,rcut,cweight,vrb,centers,nlist,atomic] = utils.parsing.set_variable_values_tenskernel(args)

print ""
print "NUMBER OF CONFIGURATIONS =", npoints
print "Building the symmetry adapted SOAP kernel for L =", lval
print ""
print "Kernel hyper-parameters:" 
print "---------------------------"
print "Gaussian width =", sg 
print "Angular cutoff =", lc
print "Environment cutoff =", rcut
print "Central atom weight =", cweight 
print ""

# Build kernels.
[centers,atom_indexes,natmax,nat,kernels] = utils.kernels.build_kernels(lval,ftrs,npoints,sg,lc,rcut,cweight,vrb,centers,nlist)

# Transformation matrices to real spherical harmonics.
CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
CC = np.conj(CR)
CT = np.transpose(CR)

if atomic:

    # Transform local kernels to real
    kloc = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1),dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            for ii in xrange(nat[i]):
               for jj in xrange(nat[j]):
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
        filename = "kernel"+str(lval)+"_atom"+str(k)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".npy"
        envfile.append(open(filename,"w"))
    nspecies = len(centers)
    for ispe in xrange(nspecies):
        np.save(envfile[ispe],katomic[ispe])
#        for i in xrange(natspe[ispe]*npoints):
#            for j in xrange(natspe[ispe]*npoints):
#                for iim,jjm in product(xrange(2*lval+1),xrange(2*lval+1)):
#                    print >> envfile[ispe], katomic[ispe][i,j,iim,jjm]

else:
    for n in xrange(len(nlist)):
        # Transform kernel to real
        kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=float)
        for i,j in product(xrange(npoints),xrange(npoints)):
            kernel[i,j] = np.real(np.dot(np.dot(CC,kernels[1+n][i,j]),CT))
        # Save kernel.
        kernel_file = "kernel"+str(lval)+"_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+"_n"+str(nlist[n])+".npy"
        np.save(kernel_file,kernel)
        #kernfile = open(kernel_file,"w")
        #for i in xrange(npoints):
        #    for j in xrange(npoints):
        #        for iim,jjm in product(xrange(2*lval+1),xrange(2*lval+1)):
        #            print >> kernfile, kernel[i,j,iim,jjm]
        #kernfile.close()
