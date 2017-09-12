#!/usr/bin/python

import numpy as np
import math
import scipy.linalg
import argparse 
from random import shuffle

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

def do_sagpr2(lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm):

    # initialize regression
    ndata = len(alps)

    if rdm == 0:
        trrangemax =  np.asarray(range(sel[0],sel[1]),int)
    else:
        data_list = range(ndata)
        shuffle(data_list)
        trrangemax = np.asarray(data_list[:rdm],int).copy()
    terange = np.setdiff1d(range(ndata), trrangemax)

    ns = len(terange)
    ntmax = len(trrangemax)
    nt = int(fractrain*ntmax)
    trrange = trrangemax[0:nt]

    if (len(alps) != ntmax+ns):
        print "Polarizabilities file must have the same length as features file!"
        sys.exit(0)

    # Build kernels
    kernel0 = np.zeros((ndata,ndata),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):
            kernel0[i,j] = kernel0_flatten[k]
            k += 1
    kernel2 = np.zeros((ndata,ndata,5,5),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):        
            for iim in xrange(5):
                for jjm in xrange(5):
                    kernel2[i,j,iim,jjm] = kernel2_flatten[k]
                    k += 1

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
    alptrain = np.zeros((nt,6),dtype=complex)
    alptest = np.zeros((ns,6),dtype=complex)
    for i in xrange(nt):
        alptrain[i] = np.array([vtrain[i][0],vtrain[i][1]*np.sqrt(2.0),vtrain[i][2]*np.sqrt(2.0),vtrain[i][4],vtrain[i][5]*np.sqrt(2.0),vtrain[i][8]],dtype=complex)
    for i in xrange(ns):
        alptest[i]  = np.array([vtest[i][0],vtest[i][1]*np.sqrt(2.0),vtest[i][2]*np.sqrt(2.0),vtest[i][4],vtest[i][5]*np.sqrt(2.0),vtest[i][8]],dtype=complex)

    # Unitary transormation matrix from Cartesian to spherical (l=0,m=0 | l=2,m=-2,-1,0,+1,+2), Condon-Shortley convention.
    CS = np.array([[-1.0/np.sqrt(3.0),0.5,0.0,-1.0/np.sqrt(6.0),0.0,0.5],[0.0,-0.5j,0.0,0.0,0.0,0.5j],[0.0,0.0,0.5,0.0,-0.5,0.0],[-1.0/np.sqrt(3.0),-0.5,0.0,-1.0/np.sqrt(6.0),0.0,-0.5],[0.0,0.0,-0.5j,0.0,-0.5j,0.0],[-1.0/np.sqrt(3.0),0.0,0.0,2.0/np.sqrt(6.0),0.0,0.0]],dtype = complex)
    degeneracy = [1.0,np.sqrt(2.0),np.sqrt(2.0),1.0,np.sqrt(2.0),1.0]
    for i in xrange(6):
        CS[i] = CS[i] * degeneracy[i]
    # Transformation matrix from complex to real spherical harmonics (l=2,m=-2,-1,0,+1,+2).
    CR2 = np.array([[1.0j,0.0,0.0,0.0,-1.0j],[0.0,1.0j,0.0,1.0j,0.0],[0.0,0.0,np.sqrt(2.0),0.0,0.0],[0.0,1.0,0.0,-1.0,0.0],[1.0,0.0,0.0,0.0,1.0]],dtype=complex) / np.sqrt(2.0)

    # Extract the complex spherical components (l=0,l=2) of the polarizabilities.
    vtrain0 = np.zeros(nt,dtype=complex)        # m =       0
    vtest0  = np.zeros(ns,dtype=complex)        # m =       0
    vtrain2 = np.zeros((nt,5),dtype=complex)    # m = -2,-1,0,+1,+2
    vtest2  = np.zeros((ns,5),dtype=complex)    # m = -2,-1,0,+1,+2
    for i in xrange(nt):
        dotpr = np.dot(alptrain[i],CS)
        vtrain0[i] = dotpr[0]
        vtrain2[i] = dotpr[1:6]
    for i in xrange(ns):
        dotpr = np.dot(alptest[i],CS)
        vtest0[i] = dotpr[0]
        vtest2[i] = dotpr[1:6]

    vtrain0 = np.real(vtrain0).astype(float)
    meantrain0 = np.mean(vtrain0)
    vtrain0 -= meantrain0        
    vtest0 = np.real(vtest0).astype(float)
    # For l=2, convert the complex spherical components into real spherical components.
    realvtrain2 = np.array([np.real(np.dot(CR2,vtrain2[i])) for i in xrange(nt)],dtype=float)
    vtrain2 = np.concatenate(realvtrain2).astype(float) 
    vtest2 = np.concatenate(np.array([np.real(np.dot(CR2,vtest2[i])) for i in xrange(ns)],dtype=float)).astype(float)
    print "Training and testing vectors built."

    # Build training kernels.
    ktrain0 = np.real(k0tr) + lm0*np.identity(nt)
    ktrain2 = np.zeros((5*nt,5*nt),dtype=float)
    ktrainpred2 = np.zeros((5*nt,5*nt),dtype=float)
    CC2 = np.conj(CR2)
    CT2 = np.transpose(CR2)
    for i in xrange(nt):
        for j in xrange(nt):
            k2rtr = k2tr[i][j]
            for al in xrange(5):
                for be in xrange(5):
                    aval = 5*i + al
                    bval = 5*j + be
                    ktrain2[aval][bval] = k2rtr[al][be] + lm2*(aval==bval)
                    ktrainpred2[aval][bval] = k2rtr[al][be] 

    # Invert training kernels.
    invktrvec0 = scipy.linalg.solve(ktrain0,vtrain0)
    invktrvec2 = scipy.linalg.solve(ktrain2,vtrain2)
    print "Training kernels built and inverted."

    # Build testing kernels.
    ktest0 = np.real(k0te)
    ktest2 = np.zeros((5*ns,5*nt),dtype=float)
    for i in xrange(ns):
        for j in xrange(nt):
            k2rte = k2te[i][j]
            for al in xrange(5):
                for be in xrange(5):
                    aval = 5*i + al
                    bval = 5*j + be
                    ktest2[aval][bval] = k2rte[al][be]
    print "Testing kernels built."
    print ""
    print "testing data points: ", ns    
    print "training data points: ", nt   
    print "Results for lambda_0 and lambda_2 = ", lm0, lm2
    print "--------------------------------"
    # Predict on train data set.
    outvec0 = np.dot(np.real(k0tr),invktrvec0)
    outvec2 = np.dot(ktrainpred2,invktrvec2)
    # Print out errors and diagnostics.
    intrins_dev0   = np.std(vtrain0) 
    intrins_error0 = 100.0 * np.sqrt(np.sum((outvec0-vtrain0)**2)/(nt))/intrins_dev0
    intrins_dev2   = np.std(vtrain2)
    intrins_error2 = 100.0 * np.sqrt(np.sum((outvec2-vtrain2)**2)/(5*nt))/intrins_dev2
    # Convert the predicted full tensor back to Cartesian coordinates.    
    outvec0 += meantrain0 
    outvec2s = outvec2.reshape((nt,5))
    outsphr2 = np.zeros((nt,5),dtype=complex)
    alpsphe = np.zeros((nt,6),dtype=complex)
    alpcart = np.zeros((nt,6),dtype=float)
    alphas = np.zeros((nt,9),dtype=float)
    for i in xrange(nt):
        outsphr2[i] = np.dot(np.conj(CR2).T,outvec2s[i])
        alpsphe[i] = [outvec0[i],outsphr2[i][0],outsphr2[i][1],outsphr2[i][2],outsphr2[i][3],outsphr2[i][4]]
        alpcart[i] = np.real(np.dot(alpsphe[i],np.conj(CS).T))
    predcart = np.concatenate([[alpcart[i][0],alpcart[i][1]/np.sqrt(2.0),alpcart[i][2]/np.sqrt(2.0),alpcart[i][1]/np.sqrt(2.0),alpcart[i][3],alpcart[i][4]/np.sqrt(2.0),alpcart[i][2]/np.sqrt(2.0),alpcart[i][4]/np.sqrt(2.0),alpcart[i][5]] for i in xrange(nt)]).astype(float)
    # Print out errors and diagnostics.
    testcart = np.real(np.concatenate(vtrain)).astype(float)
    intrins_dev = np.std(np.split(testcart,nt),axis=0)
    intrins_error = 100*np.sqrt(np.mean(np.sum(np.split((predcart-testcart)**2,nt),axis=0)/nt/(intrins_dev**2)))

    print " TRAIN STD  (l=0) = %.4f"%intrins_dev0
    print " TRAIN STD  (l=2) = %.4f"%intrins_dev2
    print " TRAIN STD  CARTE = %.4f"%np.sqrt(np.mean(intrins_dev**2))
    print " TRAIN RMSE (l=0) = %.4f %%"%intrins_error0
    print " TRAIN RMSE (l=2) = %.4f %%"%intrins_error2
    print " TRAIN RMSE CARTE = %.4f %%"%intrins_error

    # Predict on test data set..
    outvec0 = np.dot(ktest0,invktrvec0)
    outvec0 += meantrain0 
    outvec2 = np.dot(ktest2,invktrvec2)

    mean0 = np.mean(vtest0)
    intrins_dev0   = np.std(vtest0)
    intrins_error0 = 100.0 * np.sqrt(np.sum((outvec0-vtest0)**2)/(ns))/intrins_dev0
    abs_error0 = np.sqrt(np.sum((outvec0-vtest0)**2)/(ns)) 
    mean2 = 0.0
    for i in range(ns):
        mean2 += np.linalg.norm(vtest2.reshape((ns,5))[i])
    mean2 /= float(ns)
    std2 = 0.0
    for i in range(ns):
        std2 += (np.linalg.norm(vtest2.reshape((ns,5))[i]) - mean2)**2
    std2 = np.sqrt(std2/float(ns))
    abs_error2 = 0.0
    for i in range(ns):
        abs_error2 += (np.linalg.norm(vtest2.reshape((ns,5))[i]) - np.linalg.norm(outvec2.reshape((ns,5))[i]))**2
    abs_error2 = np.sqrt(abs_error2/float(ns))
    
    intrins_dev2   = np.std(vtest2)
    intrins_error2 = 100.0 * np.sqrt(np.sum((outvec2-vtest2)**2)/(5*ns))/intrins_dev2
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
    predcart = np.concatenate([[alpcart[i][0],alpcart[i][1]/np.sqrt(2.0),alpcart[i][2]/np.sqrt(2.0),alpcart[i][1]/np.sqrt(2.0),alpcart[i][3],alpcart[i][4]/np.sqrt(2.0),alpcart[i][2]/np.sqrt    (2.0),alpcart[i][4]/np.sqrt(2.0),alpcart[i][5]] for i in xrange(ns)]).astype(float)
    # Print out errors and diagnostics.
    testcart = np.real(np.concatenate(vtest)).astype(float)
    intrins_dev = np.std(np.split(testcart,ns),axis=0)
    intrins_error = 100*np.sqrt(np.mean(np.sum(np.split((predcart-testcart)**2,ns),axis=0)/ns/(intrins_dev**2)))
    
    corr = open("correlations.txt","w")
    for i in range(ns):
        print >> corr, np.split(testcart,ns)[i][0],np.split(predcart,ns)[i][0] 
    corr.close()
    
    print "--------------------------------"
    print " TEST STD  (l=0) = %.4f"%intrins_dev0
    print " TEST STD  (l=2) = %.4f"%intrins_dev2
    print " TEST STD  CARTE = %.4f"%np.sqrt(np.mean(intrins_dev**2))
    print " TEST RMSE (l=0) = %.4f %%"%intrins_error0    
    print " TEST RMSE (l=2) = %.4f %%"%intrins_error2    
    print " TEST RMSE CARTE = %.4f %%"%intrins_error

    print ""
    print "RESULTS FOR L=0 MODULI"
    print "-----------------------------------------------------"
    print "MEAN", mean0 
    print "STD", intrins_dev0
    print "RSME", abs_error0
    print ""
    print "RESULTS FOR L=2 MODULI"
    print "-----------------------------------------------------"
    print "MEAN", mean2
    print "STD",std2
    print "RMSE",abs_error2

    # PREDICTION INVERTING CLAUSIUS MOSSOTTI   
    print ""
    print "RESULTS INVERTING CLAUSIUS MOSSOTTI"

    cell_data = open("cell_400.in","r")
    clines = cell_data.readlines()
    cell_data.close()
    lines = [clines[i] for i in terange]

    nat = 32
    ang2bohr = 1.889725989
    vol = np.zeros(ns,dtype=float)
    for i in range(ns):
        cl = lines[i].split()
        cell = []
        for j in range(3):
            arr = [cl[j*3],cl[j*3+1],cl[j*3+2]]
            cell.append(arr)
        ccell = np.asarray(cell,dtype=float)*ang2bohr
        vol[i] = np.absolute(np.dot(ccell[0],np.cross(ccell[1],ccell[2])))

    epspred = np.zeros((ns,9),dtype=float)
    epstest = np.zeros((ns,9),dtype=float)
    rebuilt = open("rebuilt_eps.txt","w")
    for i in range(ns):
        alphatest = np.real(np.split(vtest[i],3))
        alphapred = np.split(np.split(predcart,ns)[i],3)
        epstest[i] = - np.concatenate(np.dot(np.linalg.inv(nat*(alphatest/vol[i])-np.eye(3)),(2*nat*(alphatest/vol[i])+np.eye(3))))
        epspred[i] = - np.concatenate(np.dot(np.linalg.inv(nat*(alphapred/vol[i])-np.eye(3)),(2*nat*(alphapred/vol[i])+np.eye(3))))
        print >> rebuilt, epstest[i][0],epstest[i][1], epspred[i][0],epspred[i][1]
    rebuilt.close()

    epstest0 = np.zeros(ns,dtype=float)        # m =       0
    epspred0 = np.zeros(ns,dtype=float)        # m =       0
    epstest2 = np.zeros((ns,5),dtype=float)    # m = -2,-1,0,+1,+2
    epspred2 = np.zeros((ns,5),dtype=float)    # m = -2,-1,0,+1,+2
    for i in xrange(ns):
        epstest6 = np.array([epstest[i][0],epstest[i][1]*np.sqrt(2.0),epstest[i][2]*np.sqrt(2.0),epstest[i][4],epstest[i][5]*np.sqrt(2.0),epstest[i][8]],dtype=complex)
        epspred6 = np.array([epspred[i][0],epspred[i][1]*np.sqrt(2.0),epspred[i][2]*np.sqrt(2.0),epspred[i][4],epspred[i][5]*np.sqrt(2.0),epspred[i][8]],dtype=complex)
        test = np.dot(epstest6,CS)
        pred = np.dot(epspred6,CS)
        epstest0[i] = np.real(test[0])
        epstest2[i] = np.real(np.dot(CR2,test[1:6]))
        epspred0[i] = np.real(pred[0])
        epspred2[i] = np.real(np.dot(CR2,pred[1:6]))

    mean0 = np.mean(epstest0)
    intrins_dev0 = np.std(epstest0)
    intrins_error0 = 100.0 * np.sqrt(np.sum((epspred0-epstest0)**2)/(ns))/intrins_dev0
    abs_error0 = np.sqrt(np.sum((epspred0-epstest0)**2)/(ns)) 
    mean2 = 0.0
    for i in range(ns):
        mean2 += np.linalg.norm(epstest2.reshape((ns,5))[i])
    mean2 /= float(ns)
    std2 = 0.0
    for i in range(ns):
        std2 += (np.linalg.norm(epstest2.reshape((ns,5))[i]) - mean2)**2
    std2 = np.sqrt(std2/float(ns))
    abs_error2 = 0.0
    for i in range(ns):
        abs_error2 += (np.linalg.norm(epstest2.reshape((ns,5))[i]) - np.linalg.norm(epspred2.reshape((ns,5))[i]))**2
    abs_error2 = np.sqrt(abs_error2/float(ns))
    
    intrins_dev2   = np.std(epstest2)
    intrins_error2 = 100.0 * np.sqrt(np.sum((epspred2-epstest2)**2)/(5*ns))/intrins_dev2
    intrins_dev = np.std(np.split(epstest,ns),axis=0)
    intrins_error = 100*np.sqrt(np.mean(np.sum(np.split((epspred-epstest)**2,ns),axis=0)/ns/(intrins_dev**2)))    
    print "--------------------------------"
    print " TEST STD  (l=0) = %.4f"%intrins_dev0
    print " TEST STD  (l=2) = %.4f"%intrins_dev2
    print " TEST STD  CARTE = %.4f"%np.sqrt(np.mean(intrins_dev**2))
    print " TEST RMSE (l=0) = %.4f %%"%intrins_error0
    print " TEST RMSE (l=2) = %.4f %%"%intrins_error2
    print " TEST RMSE CARTE = %.4f %%"%intrins_error

    print ""
    print "RESULTS FOR L=0 MODULI"
    print "-----------------------------------------------------"
    print "MEAN", mean0 
    print "STD", intrins_dev0
    print "RSME", abs_error0
    print ""
    print "RESULTS FOR L=2 MODULI"
    print "-----------------------------------------------------"
    print "MEAN", mean2
    print "STD",std2
    print "RMSE",abs_error2

if __name__ == '__main__':
    # Read in all arguments and call the main function.
    args = add_command_line_arguments_learn("a")
    [lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm] = set_variable_values_learn(args)
    do_sagpr2(lm0,lm2,fractrain,alps,kernel0_flatten,kernel2_flatten,sel,rdm)
