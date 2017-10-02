#!usr/bin/python

import utils.kernels
import utils.kern_utils
import utils.parsing
import argparse
import sys
import numpy as np
from itertools import product

# This is a wrapper that calls python scripts to build L-SOAP kernels.

# INPUT ARGUMENTS
args = utils.parsing.add_command_line_arguments_tenskernel("Tensorial kernel")
[ftrs,vcell,npoints,lval,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist] = utils.parsing.set_variable_values_tenskernel(args)

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
print "Filtering width =", fwidth
print ""

[kernels] = utils.kernels.build_kernels(lval,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)

CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
CC = np.conj(CR)
CT = np.transpose(CR)
#kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=float)
#for i,j in product(xrange(npoints),xrange(npoints)):
#    kernel[i,j] = np.real(np.dot(np.dot(CC,kernels[0][i,j]),CT))
## Save kernel.
#kernel_file = open("kernel"+str(lval)+"_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".txt","w")
#for i,j in product(xrange(npoints),xrange(npoints)):
#    for iim,jjm in product(xrange(2*lval+1),xrange(2*lval+1)):
#        print >> kernel_file,kernel[i,j,iim,jjm]
#kernel_file.close()

# Print product kernels.
#if (nlist != [0]):
for n in xrange(len(nlist)):
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=float)
    for i,j in product(xrange(npoints),xrange(npoints)):
        kernel[i,j] = np.real(np.dot(np.dot(CC,kernels[n+1][i,j]),CT))
    # Save kernel.
    kernel_file = open("kernel"+str(lval)+"_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+"_n"+str(nlist[n])+".txt","w")
    for i,j in product(xrange(npoints),xrange(npoints)):
        for iim,jjm in product(xrange(2*lval+1),xrange(2*lval+1)):
            print >> kernel_file,kernel[i,j,iim,jjm]
    kernel_file.close()
