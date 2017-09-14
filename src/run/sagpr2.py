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

def do_sagpr2(lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm):

    # initialize regression
    intrins_dev0 = 0.0
    abs_error0 = 0.0
    intrins_dev2 = 0.0
    abs_error2 = 0.0
    ncycles = 5

    print "Results averaged over "+str(ncycles)+" cycles"

    for ic in range(ncycles):

        ndata = len(alps)
        [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)
       
        # Build kernel matrix
        kernel0 = utils.kern_utils.unflatten_kernel0(ndata,kernel0_flatten)
        kernel2 = utils.kern_utils.unflatten_kernel(ndata,5,kernel2_flatten)

        # Partition properties and kernel for training and testing
        alpstrain = [alps[i] for i in trrange]
        alpstest = [alps[i] for i in terange]
        vtrain = np.array([i.split() for i in alpstrain]).astype(complex)
        vtest = np.array([i.split() for i in alpstest]).astype(complex)
        k0tr = [[kernel0[i,j] for j in trrange] for i in trrange]
        k0te = [[kernel0[i,j] for j in trrange] for i in terange]
        k2tr = [[kernel2[i,j] for j in trrange] for i in trrange]
        k2te = [[kernel2[i,j] for j in trrange] for i in terange]

        # Extract the 6 non-equivalent components xx,xy,xz,yy,yz,zz; include degeneracy.
        [alptrain,alptest] = utils.kern_utils.get_non_equivalent_components(vtrain,vtest)
   
        # Unitary transormation matrix from Cartesian to spherical (l=0,m=0 | l=2,m=-2,-1,0,+1,+2), Condon-Shortley convention.
        CS = np.array([[-1.0/np.sqrt(3.0),0.5,0.0,-1.0/np.sqrt(6.0),0.0,0.5],[0.0,-0.5j,0.0,0.0,0.0,0.5j],[0.0,0.0,0.5,0.0,-0.5,0.0],[-1.0/np.sqrt(3.0),-0.5,0.0,-1.0/np.sqrt(6.0),0.0,-0.5],[0.0,0.0,-0.5j,0.0,-0.5j,0.0],[-1.0/np.sqrt(3.0),0.0,0.0,2.0/np.sqrt(6.0),0.0,0.0]],dtype = complex)
        degeneracy = [1.0,np.sqrt(2.0),np.sqrt(2.0),1.0,np.sqrt(2.0),1.0]
        for i in xrange(6):
            CS[i] = CS[i] * degeneracy[i]

        # Transformation matrix from complex to real spherical harmonics (l=2,m=-2,-1,0,+1,+2).
        [CR2] = utils.kern_utils.complex_to_real_transformation([5])

        # Extract the complex spherical components (l=0,l=2) of the polarizabilities.
        [ [vtrain0,vtrain2],[vtest0,vtest2] ] = utils.kern_utils.partition_spherical_components(alptrain,alptest,CS,[1,5],ns,nt)

        vtrain0 = np.real(vtrain0).astype(float)
        meantrain0 = np.mean(vtrain0)
        vtrain0 -= meantrain0        
        vtest0 = np.real(vtest0).astype(float)

        # For l=2, convert the complex spherical components into real spherical components.
        realvtrain2 = np.array([np.real(np.dot(CR2,vtrain2[i])) for i in xrange(nt)],dtype=float)
        vtrain2 = np.concatenate(realvtrain2).astype(float) 
        vtest2 = np.concatenate(np.array([np.real(np.dot(CR2,vtest2[i])) for i in xrange(ns)],dtype=float)).astype(float)

        # Build training kernels.
        ktrain0 = np.real(k0tr) + lm0*np.identity(nt)
        [ktrain2,ktrainpred2] = utils.kern_utils.build_training_kernel(nt,5,k2tr,lm2)
    
        # Invert training kernels.
        invktrvec0 = scipy.linalg.solve(ktrain0,vtrain0)
        invktrvec2 = scipy.linalg.solve(ktrain2,vtrain2)

        # Build testing kernels.
        ktest0 = np.real(k0te)
        ktest2 = utils.kern_utils.build_testing_kernel(ns,nt,5,k2te)
 
        # Predict on test data set.
        outvec0 = np.dot(ktest0,invktrvec0)
        outvec0 += meantrain0 
        outvec2 = np.dot(ktest2,invktrvec2)

        intrins_dev0 += np.std(vtest0)**2
        abs_error0 += np.sum((outvec0-vtest0)**2)/(ns) 

        intrins_dev2 += np.std(vtest2)**2
        abs_error2 += np.sum((outvec2-vtest2)**2)/(5*ns)

        # Convert the predicted full tensor back to Cartesian coordinates.
        outvec2s = outvec2.reshape((ns,5))
        outsphr2 = np.zeros((ns,5),dtype=complex)
        alpsphe = np.zeros((ns,6),dtype=complex)
        alpcart = np.zeros((ns,6),dtype=float)
        alphas = np.zeros((ns,9),dtype=float)
        for i in xrange(ns):
            outsphr2[i] = np.dot(np.conj(CR2).T,outvec2s[i])
            alpsphe[i] = [outvec0[i],outsphr2[i][0],outsphr2[i][1],outsphr2[i][2],outsphr2[i][3],outsphr2[i][4]]
            alpcart[i] = np.real(np.dot(alpsphe[i],np.conj(CS).T))
        predcart = np.concatenate([[alpcart[i][0],alpcart[i][1]/np.sqrt(2.0),alpcart[i][2]/np.sqrt(2.0),alpcart[i][1]/np.sqrt(2.0),alpcart[i][3],alpcart[i][4]/np.sqrt(2.0),alpcart[i][2]/np.sqrt(2.0),alpcart[i][4]/np.sqrt(2.0),alpcart[i][5]] for i in xrange(ns)]).astype(float)

    intrins_dev0 = np.sqrt(intrins_dev0/float(ncycles))
    abs_error0 = np.sqrt(abs_error0/float(ncycles))
    intrins_error0 = 100*np.sqrt(abs_error0**2/intrins_dev0**2)

    intrins_dev2 = np.sqrt(intrins_dev2/float(ncycles))
    abs_error2 = np.sqrt(abs_error2/float(ncycles))
    intrins_error2 = 100*np.sqrt(abs_error2**2/intrins_dev2**2)

    print ""
    print "testing data points: ", ns
    print "training data points: ", nt
    print "Results for lambda_1 and lambda_3 = ", lm0, lm2
    print "--------------------------------"
    print "RESULTS FOR L=0 MODULI"
    print "-----------------------------------------------------"
    print "STD", intrins_dev0
    print "ABS RSME", abs_error0
    print "RMSE = %.4f %%"%intrins_error0
    print "-----------------------------------------------------"
    print "RESULTS FOR L=2 MODULI"
    print "-----------------------------------------------------"
    print "STD",intrins_dev2
    print "ABS RMSE",abs_error2
    print "RMSE = %.4f %%"%intrins_error2

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lm", "--lmda", nargs='+', help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",type=float, help="Fraction of data points used for testing")
    parser.add_argument("-t", "--tensors", help="File containing tensors")
    parser.add_argument("-k0", "--kernel0", help="File containing L=0 kernel")
    parser.add_argument("-k2", "--kernel2", help="File containing L=2 kernel")
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
        print "Tensors file must be specified!"
        sys.exit(0)
    tens=[line.rstrip('\n') for line in open(tfile)]

    print ""
    print "Loading kernel matrices..."

    if args.kernel0:
        kfile0 = args.kernel0
    else:
        print "Kernel file must be specified!"
        sys.exit(0)
    kernel0 = np.loadtxt(kfile0,dtype=float)

    if args.kernel2:
        kfile2 = args.kernel2
    else:
        print "Kernel file must be specified!"
        sys.exit(0)
    # Read in L=2 kernel
    kernel2 = np.loadtxt(kfile2,dtype=float)

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

    return [lm[0],lm[2],ftr,tens,kernel0,kernel2,sel,rdm]

###############################################################################################################################

if __name__ == '__main__':
    # Read in all arguments and call the main function.
    args = add_command_line_arguments_learn("SA-GPR for rank-2 tensors")
    [lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm] = set_variable_values_learn(args)
    do_sagpr2(lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm)
