#!usr/bin/python

import utils.kernels
import utils.parsing
import argparse
import sys
import numpy as np

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

if (lval==0):

#    [[kernel]] = utils.kernels.build_kernels(0,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)
    [kernels] = utils.kernels.build_kernels(0,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)

    # Save the kernel
    kernel_file = open("kernel0_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".txt","w")
    for i in xrange(npoints):
        for j in xrange(npoints):
            print >> kernel_file, kernels[0][i,j]
    kernel_file.close()

elif (lval==1):

    [[ckernel]] = utils.kernels.build_kernels(1,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)

    # From COMPLEX to REAL, M = -1,0,+1
    CR = np.array([[1.0j,0.0,1.0j],
                   [0.0,np.sqrt(2.0),0.0],
                   [1.0,0.0,-1.0]            ], dtype=complex) / np.sqrt(2.0)
    CC = np.conj(CR)
    CT = np.transpose(CR)
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1), dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            kernel[i,j] = np.real(np.dot(np.dot(CC,ckernel[i,j]),CT))

    # Save the kernel
    kernel_file = open("kernel1_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".txt","w")
    for i in xrange(npoints):
        for j in xrange(npoints):
            for iim in xrange(3):
                for iik in xrange(3):
                    print >> kernel_file, kernel[i,j,iim,iik]
    kernel_file.close()

elif (lval==2):

    [[ckernel]] = utils.kernels.build_kernels(2,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)

    # From COMPLEX to REAL M = -2,-1,0,+1,+2 
    CR = np.array([[1.0j,0.0,0.0,0.0,-1.0j],
                   [0.0,1.0j,0.0,1.0j,0.0],
                   [0.0,0.0,np.sqrt(2.0),0.0,0.0],
                   [0.0,1.0,0.0,-1.0,0.0],
                   [1.0,0.0,0.0,0.0,1.0]      ],dtype=complex) / np.sqrt(2.0)
    CC = np.conj(CR)
    CT = np.transpose(CR)
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1), dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            kernel[i,j] = np.real(np.dot(np.dot(CC,ckernel[i,j]),CT))

    # Save the kernel
    kernel_file = open("kernel2_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".txt","w")
    for i in xrange(npoints):
        for j in xrange(npoints):
            for iim in xrange(5):
                for iik in xrange(5):
                    print >> kernel_file, kernel[i,j,iim,iik]
    kernel_file.close()

elif (lval==3):

    [[ckernel]] = utils.kernels.build_kernels(3,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers,nlist)

    # From COMPLEX to REAL M = -3,-2,-1,0,+1,+2,+3
    CR =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,1.0j],
                  [0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0],
                  [0.0,0.0,1.0j,0.0,1.0j,0.0,0.0],
                  [0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0],
                  [0.0,0.0,1.0,0.0,-1.0,0.0,0.0],
                  [0.0,1.0,0.0,0.0,0.0,1.0,0.0],
                  [1.0,0.0,0.0,0.0,0.0,0.0,-1.0]   ], dtype=complex) / np.sqrt(2.0)
    CC = np.conj(CR)
    CT = np.transpose(CR)
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1), dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            kernel[i,j] = np.real(np.dot(np.dot(CC,ckernel[i,j]),CT))

    # Save the kernel
    kernel_file = open("kernel3_"+str(npoints)+"_sigma"+str(sg)+"_lcut"+str(lc)+"_cutoff"+str(rcut)+"_cweight"+str(cweight)+".txt","w")
    for i in xrange(npoints):
        for j in xrange(npoints):
            for iim in xrange(7):
                for iik in xrange(7):
                    print >> kernel_file, kernel[i,j,iim,iik]
    kernel_file.close()

else:

    print "The symmetry adapted kernel for this L value is not yet covered by this code!"
    sys.exit(0)
