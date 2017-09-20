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

def do_sagpr1(lvals,lm,fractrain,tens,kernel_flatten,sel,rdm,rank,ncycles):

    # Initialize regression
#    ncycles = 1

#    lvals = [1]
    degen = [(2*l+1) for l in lvals]
    intrins_dev   = np.zeros(len(lvals),dtype=float)
    intrins_error = np.zeros(len(lvals),dtype=float)
    abs_error     = np.zeros(len(lvals),dtype=float)

    print "Results averaged over "+str(ncycles)+" cycles"

    for ic in range(ncycles):

        ndata = len(tens)
        [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)

        # Build kernel matrix
        kernel = [utils.kern_utils.unflatten_kernel(ndata,degen[i],kernel_flatten[i]) for i in xrange(len(lvals))]

        # Partition properties and kernel for training and testing
        [vtrain,vtest,ktr,kte] = utils.kern_utils.partition_kernels_properties(tens,kernel,trrange,terange)

        # Extract the 3 non-equivalent components x,y,z; include degeneracy.
        [tenstrain,tenstest,mask1,mask2] = utils.kern_utils.get_non_equivalent_components(vtrain,vtest)

        # Unitary transormation matrix from Cartesian to spherical (l=1,m=-1,0,+1), Condon-Shortley convention.
#        CS  = np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0)
#        for i in xrange(3):
#            CS[i] = CS[i] * mask1[i]
#
        CS = utils.kern_utils.get_CS_matrix(rank,mask1,mask2)

        # Transformation matrix from complex to real spherical harmonics (l=1,m=-1,0,+1).
        CR = utils.kern_utils.complex_to_real_transformation(degen)

        # Extract the real spherical components (l=1) of the dipoles.
        [ vtrain_part,vtest_part ] = utils.kern_utils.partition_spherical_components(tenstrain,tenstest,CS,CR,degen,ns,nt)

        meantrain = np.zeros(len(degen),dtype=float)
        for i in xrange(len(degen)):
            if degen[i]==1:
                vtrain_part[i]  = np.real(vtrain_part[i]).astype(float)
                meantrain[i]    = np.mean(vtrain_part[i])
                vtrain_part[i] -= meantrain[i]
                vtest_part[i]   = np.real(vtest_part[i]).astype(float)
        
        # Build training kernel.
        ktrain_all_pred = [utils.kern_utils.build_training_kernel(nt,degen[i],ktr[i],lm[i]) for i in xrange(len(degen))]
        ktrain     = [ktrain_all_pred[i][0] for i in xrange(len(degen))]
        ktrainpred = [ktrain_all_pred[i][1] for i in xrange(len(degen))]

        # Invert training kernel.
        invktrvec = [scipy.linalg.solve(ktrain[i],vtrain_part[i]) for i in xrange(len(degen))]

        # Build testing kernel.
        ktest = [utils.kern_utils.build_testing_kernel(ns,nt,degen[i],kte[i]) for i in xrange(len(degen))]

        # Predict on test data set.
        outvec = [np.dot(ktest[i],invktrvec[i]) for i in xrange(len(degen))]
        for i in xrange(len(degen)):
            if degen[i]==1:
                outvec[i] += meantrain[i]

        # Accumulate errors.
        for i in xrange(len(degen)):
            intrins_dev[i] += np.std(vtest_part[i])**2
            abs_error[i] += np.sum((outvec[i]-vtest_part[i])**2)/(degen[i]*ns)

        # Convert the predicted full tensor back to Cartesian coordinates.
        predcart = utils.kern_utils.spherical_to_cartesian(outvec,degen,ns,CR,CS,mask1,mask2)

        testcart = np.real(np.concatenate(vtest)).astype(float)

    for i in xrange(len(degen)):
        intrins_dev[i] = np.sqrt(intrins_dev[i]/float(ncycles))
        abs_error[i] = np.sqrt(abs_error[i]/float(ncycles))
        intrins_error[i] = 100*np.sqrt(abs_error[i]**2/intrins_dev[i]**2)

    print ""
    print "testing data points: ", ns
    print "training data points: ", nt
    for i in xrange(len(degen)):
        print "--------------------------------"
        print "RESULTS FOR L=%i MODULI (lambda=%f)"%(lvals[i],lm[i])
        print "-----------------------------------------------------"
        print "STD", intrins_dev[i]
        print "ABS RSME", abs_error[i]
        print "RMSE = %.4f %%"%intrins_error[i]

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
    [lm1,fractrain,tens,kernel1_flatten,sel,rdm] = set_variable_values_learn(args)
    do_sagpr1([lm1],fractrain,tens,[kernel1_flatten],sel,rdm,1)
