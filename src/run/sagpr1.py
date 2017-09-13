#!/usr/bin/python

import sys
import numpy as np
import math
import scipy.linalg
import argparse 
import os
sys.path.insert(1,os.path.join(sys.path[0], '..'))
import utils.kern_utils

###############################################################################################################################

def do_sagpr1(lm1,fractrain,dips,kernel1_flatten,sel,rdm):

    # Initialize regression
    intrins_dev1 = 0.0
    abs_error1 = 0.0
    ncycles = 5

    print "Results averaged over "+str(ncycles)+" cycles"

    for ic in range(ncycles):

        ndata = len(dips)
        [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)

        # Build kernel matrix
        kernel1 = utils.kern_utils.unflatten_kernel(ndata,3,kernel1_flatten)

        # Partition properties and kernel for training and testing
        dipstrain = [dips[i] for i in trrange]
        dipstest = [dips[i] for i in terange]
        vtrain = np.array([i.split() for i in dipstrain]).astype(complex)
        vtest = np.array([i.split() for i in dipstest]).astype(complex)
        k1tr = [[kernel1[i,j] for j in trrange] for i in trrange]
        k1te = [[kernel1[i,j] for j in trrange] for i in terange]

        # Unitary transormation matrix from Cartesian to spherical (l=1,m=-1,0,+1), Condon-Shortley convention.
        CS  = np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0)
        # Transformation matrix from complex to real spherical harmonics (l=1,m=-1,0,+1).
        CR1 = np.array([[1.0j,0.0,1.0j],[0.0,np.sqrt(2.0),0.0],[1.0,0.0,-1.0]],dtype = complex) / np.sqrt(2.0)    # Get training and testing vectors for l=1.
        vtrain1 = np.concatenate(np.array([np.real( np.dot(CR1,np.dot(vtrain[i],CS))) for i in xrange(nt)])).astype(float)
        vtest1  = np.concatenate(np.array([np.real( np.dot(CR1,np.dot(vtest[i],CS)))  for i in xrange(ns)])).astype(float)

        # Build training kernel.
        ktrain1 = np.zeros((3*nt,3*nt),dtype=float)
        ktrainpred1 = np.zeros((3*nt,3*nt),dtype=float)
        for i in xrange(nt):
            for j in xrange(nt):
                k1rtr = k1tr[i][j]
                for al in xrange(3):
                    for be in xrange(3):
                        aval = 3*i + al
                        bval = 3*j + be
                        ktrain1[aval][bval] = k1rtr[al][be] + lm1*(aval==bval)
                        ktrainpred1[aval][bval] = k1rtr[al][be]

        # Invert training kernel.
        invktrvec1 = scipy.linalg.solve(ktrain1,vtrain1)

        # Build testing kernel.
        ktest1 = np.zeros((3*ns,3*nt),dtype=float)
        for i in xrange(ns):
            for j in xrange(nt):
                k1rte = k1te[i][j]
                for al in xrange(3):
                    for be in xrange(3):
                        aval = 3*i + al
                        bval = 3*j + be    
                        ktest1[aval][bval] = k1rte[al][be]    

        # Predict on test data set..
        outvec1 = np.dot(ktest1,invktrvec1)
        intrins_dev1 += np.std(vtest1)**2
        abs_error1 += np.sum((outvec1-vtest1)**2)/(3*ns)
        # Convert the predicted full tensor back to Cartesian coordinates.
        outvec1s = outvec1.reshape((ns,3))
        predcart = np.concatenate(np.array([np.real(np.dot(np.dot(np.conj(CR1).T,outvec1s[i]),np.conj(CS).T)) for i in xrange(ns)])).astype(float)
        testcart = np.real(np.concatenate(vtest)).astype(float)

    intrins_dev1 = np.sqrt(intrins_dev1/float(ncycles))
    abs_error1 = np.sqrt(abs_error1/float(ncycles))
    intrins_error1 = 100*np.sqrt(abs_error1**2/intrins_dev1**2)

    print ""
    print "testing data points: ", ns
    print "training data points: ", nt
    print "Results for lambda_1 = ", lm1
    print "--------------------------------"
    print " TEST STD  = %.6f"%intrins_dev1
    print " ABS  RMSE = %.6f"%abs_error1
    print " TEST RMSE = %.6f %%"%intrins_error1

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lm", "--lmda", nargs='+', help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",type=float, help="Fraction of data points used for testing")
    parser.add_argument("-t", "--tensors", help="File containing tensors")
    parser.add_argument("-k1", "--kernel1", help="File containing L=0 kernel")
    parser.add_argument("-sel", "--select",nargs='+', help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",type=int, help="Number of random training points")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):
    lm0=0.01
    lm1=0.01
    lm2=0.01
    lm3=0.01
    lm = [lm0,lm1,lm2,lm2]
    if args.lmda:
        lmlist = args.lmda
        # This list will either be separated by spaces or by commas (or will not be allowed).
        # We will be a little forgiving and allow a mixture of both.
        if sum([lmlist[i].count(',') for i in xrange(len(lmlist))]) > 0:
            for i in xrange(len(lmlist)):
                lmlist[i] = lmlist[i].split(',')
            lmlist = np.concatenate(lmlist)
        if (len(lmlist)%2 != 0):
            print "Error: list of lambdas must have the format n,lambda[n],m,lambda[m],..."
            sys.exit(0)
        for i in xrange(len(lmlist)/2):
            nval = int(lmlist[2*i])
            lmval = float(lmlist[2*i+1])
            lm[nval] = lmval

    ftrain=1
    if args.ftrain:
        ftr = args.ftrain 
    if args.tensors:
        tfile = args.tensors
    else:
        print "Features file must be specified!"
        sys.exit(0)
    # Read in features
    tens=[line.rstrip('\n') for line in open(tfile)]
    print "Loading kernel matrices..."
    if args.kernel1:
        kfile1 = args.kernel1
    else:
        print "Kernel 1 file must be specified!"
        sys.exit(0)
    kernel1 = np.loadtxt(kfile1,dtype=float)

    beg = 0
    end = int(len(tens)/2)
    sel = [beg,end]
    if args.select:
        sellist = args.select
        for i in xrange(len(sellist)):
            sel[0] = int(sellist[0])
            sel[1] = int(sellist[1])

    rdm = 0
    if args.random:
        rdm = args.random

    return [lm[1],ftr,tens,kernel1,sel,rdm]

###############################################################################################################################

if __name__ == '__main__':
    # Read in all arguments and call the main function.
    args = add_command_line_arguments_learn("SA-GPR for vectors")
    [lm1,fractrain,dips,kernel1_flatten,sel,rdm] = set_variable_values_learn(args)
    do_sagpr1(lm1,fractrain,dips,kernel1_flatten,sel,rdm)
