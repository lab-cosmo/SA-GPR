#!/usr/bin/env python2

import utils.kernels
import utils.parsing
import utils.kern_utils
import scipy.linalg
import sys
import numpy as np

###############################################################################################################################

def do_sagpr(lvals,lm,fractrain,tens,kernel_flatten,sel,rdm,rank,ncycles):

    # initialize regression
    degen = [(2*l+1) for l in lvals]
    intrins_dev   = np.zeros(len(lvals),dtype=float)
    intrins_error = np.zeros(len(lvals),dtype=float)
    abs_error     = np.zeros(len(lvals),dtype=float)

    print "Results averaged over "+str(ncycles)+" cycles"

    for ic in range(ncycles):

        ndata = len(tens)
        [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)
       
        # Build kernel matrices.
        kernel = [utils.kern_utils.unflatten_kernel(ndata,degen[i],kernel_flatten[i]) for i in xrange(len(lvals))]

        # Partition properties and kernel for training and testing
        [vtrain,vtest,ktr,kte] = utils.kern_utils.partition_kernels_properties(tens,kernel,trrange,terange)

        # Extract the non-equivalent tensor components; include degeneracy.
        [tenstrain,tenstest,mask1,mask2] = utils.kern_utils.get_non_equivalent_components(vtrain,vtest)
  
        # Unitary transormation matrix from Cartesian to spherical, Condon-Shortley convention.
        CS = utils.kern_utils.get_CS_matrix(rank,mask1,mask2)

        # Transformation matrix from complex to real spherical harmonics.
        CR = utils.kern_utils.complex_to_real_transformation(degen)

        # Extract the real spherical components of the tensors.
        [ vtrain_part,vtest_part ] = utils.kern_utils.partition_spherical_components(tenstrain,tenstest,CS,CR,degen,ns,nt)

        meantrain = np.zeros(len(degen),dtype=float)
        for i in xrange(len(degen)):
            if degen[i]==1:
                vtrain_part[i]  = np.real(vtrain_part[i]).astype(float)
                meantrain[i]    = np.mean(vtrain_part[i])
                vtrain_part[i] -= meantrain[i]
                vtest_part[i]   = np.real(vtest_part[i]).astype(float)

        # Build training kernels.
        ktrain_all_pred = [utils.kern_utils.build_training_kernel(nt,degen[i],ktr[i],lm[i]) for i in xrange(len(degen))]
        ktrain     = [ktrain_all_pred[i][0] for i in xrange(len(degen))]
        ktrainpred = [ktrain_all_pred[i][1] for i in xrange(len(degen))]
  
        # Invert training kernels.
        invktrvec = [scipy.linalg.solve(ktrain[i],vtrain_part[i]) for i in xrange(len(degen))]

        # Build testing kernels.
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

# This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.

# Parse input arguments
args = utils.parsing.add_command_line_arguments_learn("SA-GPR")
[lvals,lm,fractrain,tens,kernels,sel,rdm,rank,ncycles] = utils.parsing.set_variable_values_learn(args)

# Read-in kernels
print "Loading kernel matrices..."
#kernel = []
#for k in xrange(len(kernels)):
#    if (kernels[k]!=''):
#    kernel.append(np.loadtxt(kernels[k],dtype=float))
#    else:
#        kernel.append(0)
kernel = [np.loadtxt(kernels[k],dtype=float) for k in xrange(len(kernels))]
print "...Kernels loaded."

## Get list of l values
#lvals = []
#if (rank%2 == 0):
#    # Even L
#    lvals = [l for l in xrange(0,rank+1,2)]
#else:
#    # Odd L
#    lvals = [l for l in xrange(1,rank+1,2)]
#
#lms     = [lm[i]     for i in lvals]
#kernels = [kernel[i] for i in lvals]

do_sagpr(lvals,lm,fractrain,tens,kernel,sel,rdm,rank,ncycles)
