#!/usr/bin/env python2

import argparse
import utils.kernels
import utils.parsing
import utils.kern_utils
import scipy.linalg
import sys
import numpy as np
from ase.io import read

# This script takes a spherical tensor component and carries out regression on it.

# Command-line arguments.
parser = argparse.ArgumentParser(description="SA-GPR Regression")
parser.add_argument("-lm",  "--lmda",     type=float, required=True,                   help="Lambda value for KRR calculation")
parser.add_argument("-ftr", "--ftrain",   type=float, default=1.0,                     help="Fraction of data points used for testing")
parser.add_argument("-f",   "--features", type=str,   required=True,                   help="File containing atomic coordinates")
parser.add_argument("-p",   "--property", type=str,   required=True,                   help="Property to be learned")
parser.add_argument("-l",   "--lval",     type=int,   required=True,                   help="l value of tensor")
parser.add_argument("-k",   "--kernel",   type=str,   required=True,                   help="File containing kernel")
parser.add_argument("-sel", "--select",   type=int,   default=[0,100],      nargs='+', help="Select maximum training partition")
parser.add_argument("-rdm", "--random",   type=int,   default=0,                       help="Number of random training points")
parser.add_argument("-nc",  "--ncycles",  type=int,   default=1,                       help="Number of cycles for regression with random selection")
parser.add_argument("-o",   "--ofile",    type=str,   default='output.out',            help="Output file for weights")
args = parser.parse_args()

# Parse command-line arguments
lm = args.lmda
fractrain = float(args.ftrain)
rdm = args.random
ncycles = args.ncycles
ofile = args.ofile
sel = args.select
if (len(sel)!=2):
    print "Beginning and end of selection must be specified!"
    sys.exit(0)

# Read in tensor
ftrs = read(args.features,':')
lval = args.lval

if lval == 0:
    tens = [str(ftrs[i].info[args.property]) for i in xrange(len(ftrs))]
elif lval == 2:
    tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in xrange(len(ftrs))]
else:
    tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in xrange(len(ftrs))]

all_data = np.array([i.split(' ') for i in tens]).astype(float)

print
print "Doing regression with L=%i"%lval
print

kernel_flatten = np.loadtxt(args.kernel,dtype=float)

# Check that the L value for the kernel is the same as that for the tensor component
if len(kernel_flatten) != len(all_data[0])**2 * len(all_data)**2:
    print "Kernel does not match data!"
    sys.exit(0)

print "Kernels loaded"
print

intrins_dev = 0.0
abs_error = 0.0

for nc in xrange(ncycles):

    # Shuffle data, if applicable
    ndata = len(tens)
    [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)

    # Put kernel into matrix form
    kernel = utils.kern_utils.unflatten_kernel(ndata,2*lval+1,kernel_flatten)

    # Partition properties and kernels for training and testing
    vtrain_part = np.array([all_data[i] for i in trrange])
    vtest_part = np.array([all_data[i] for i in terange])
    ktr = [[kernel[i,j] for j in trrange] for i in trrange]
    kte = [[kernel[i,j] for j in trrange] for i in terange]

    vtrain_part = np.concatenate(vtrain_part)
    vtest_part  = np.concatenate(vtest_part)

    # If L=0, subtract the mean from the data
    if (lval==0):
        vtrain_part  = np.real(vtrain_part).astype(float)
        meantrain    = np.mean(vtrain_part)
        vtrain_part -= meantrain
        vtest_part   = np.real(vtest_part).astype(float)

    # Build training kernels
    ktrain_all_pred = utils.kern_utils.build_training_kernel(nt,2*lval+1,ktr,lm)
    ktrain     = ktrain_all_pred[0]
    ktrainpred = ktrain_all_pred[1]

    # Invert training kernels
    invktrvec = scipy.linalg.solve(ktrain,vtrain_part)

    # Build testing kernels
    ktest = utils.kern_utils.build_testing_kernel(ns,nt,2*lval+1,kte)

    # Predict on test data set
    outvec = np.dot(ktest,invktrvec)
    if (lval==0):
        outvec += meantrain

    # Accumulate errors
    indev  = np.sqrt(np.std(vtest_part)**2)
    abserr = np.sqrt(np.sum((outvec - vtest_part)**2)/float( (2*lval+1)*ns))

    # Print out weights; each cycle will have its own output file with (1) members of the training set, (2) weights, (3) errors
    if (ofile != ''):
        print "Printing out cycle %i with intrinsic deviation %f and absolute error %f"%(nc,indev,abserr)
        print "Intrinsic error = %f"%float(100*np.sqrt(abserr**2/indev**2))
        outfile = open(ofile + "." + str(nc),"w")
        print >> outfile, "Regression for L = %i"%lval
        print >> outfile, "Tensor file:"
        print >> outfile, args.features
        print >> outfile, "Kernel file:"
        print >> outfile, args.kernel
        print >> outfile, "Training set:"
        print >> outfile, trrange
        print >> outfile, ""
        print >> outfile, "Intrinsic deviation = %f"%indev
        print >> outfile, "Absolute error = %f"%abserr
        print >> outfile, "Intrinsic error = %f"%float(100*np.sqrt(abserr**2/indev**2))
        print >> outfile, ""
        if (lval==0):
            print >> outfile, "Mean value:"
            print >> outfile, meantrain
            print >> outfile, ""
        print >> outfile, "Weights:"
        for i in xrange(len(invktrvec)):
            print >> outfile, invktrvec[i]
