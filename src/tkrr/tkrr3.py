#!/usr/bin/python

import sys
import numpy as np
import math
import scipy.linalg
import argparse 
from random import shuffle

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-t", "--tensors", help="File containing tensors")
    parser.add_argument("-k1", "--kernel1", help="File containing L=1 kernel")
    parser.add_argument("-k3", "--kernel3", help="File containing L=3 kernel")
    parser.add_argument("-sel", "--select",nargs='+', help="Select maximum training partition")
    parser.add_argument("-ftr", "--ftrain",type=float, help="Fraction of data points used for testing")
    parser.add_argument("-lm", "--lmda", nargs='+', help="Lambda values list for KRR calculation")
    parser.add_argument("-rdm", "--random",type=int, help="Number of random training points")
    args = parser.parse_args()
    return args

def set_variable_values_learn(args):
    lm0=0.001
    lm1=0.001
    lm2=0.001
    lm3=0.001
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

    # Read in L=1 kernel
    if args.kernel1:
        kfile1 = args.kernel1
    else:
        print "Kernel 1 file must be specified!"
        sys.exit(0)
    kernel1 = np.loadtxt(kfile1,dtype=float)

    # Read in L=3 kernel
    if args.kernel3:
        kfile3 = args.kernel3
    else:
        print "Kernel 3 file must be specified!"
        sys.exit(0)
    kernel3 = np.loadtxt(kfile3,dtype=float)

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

    return [lm[1],lm[3],ftr,tens,kernel1,kernel3,sel,rdm]

args = add_command_line_arguments_learn("a")
[lm1,lm3,fractrain,bets,kernel1_flatten,kernel3_flatten,sel,rdm] = set_variable_values_learn(args)

intrins_dev1 = 0.0
abs_error1 = 0.0
intrins_dev3 = 0.0
abs_error3 = 0.0
ncycles = 5.0

print "Results averaged over "+str(int(ncycles))+" cycles"

for ic in range(int(ncycles)):

    ndata = len(bets)
    if rdm == 0:
        trrangemax =  sorted(set(range(sel[0],sel[1])))
        terange =  sorted(set(range(ndata))-set(range(sel[0],sel[1])))
    else:
        data_list = range(ndata)
        shuffle(data_list)
        trrangemax = sorted([data_list[i] for i in range(rdm)])
        terange =  sorted(set(range(ndata))-set(trrangemax))

    ns = len(terange)
    ntmax = len(trrangemax)
    nt = int(fractrain*ntmax)
    trrange = trrangemax[0:nt]

    if (len(bets) != ntmax+ns):
        print "Hyperpolarizabilities file must have the same length as features file!"
        sys.exit(0)

    kernel1 = np.zeros((ndata,ndata,3,3),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):
            for iim in xrange(3):
                for jjm in xrange(3):
                    kernel1[i,j,iim,jjm] = kernel1_flatten[k]
                    k += 1

    kernel3 = np.zeros((ndata,ndata,7,7),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):
            for iim in xrange(7):
                for jjm in xrange(7):
                    kernel3[i,j,iim,jjm] = kernel3_flatten[k]
                    k += 1

    betstrain = [bets[i] for i in trrange]
    betstest = [bets[i] for i in terange]
    vtrain = np.array([i.split() for i in betstrain]).astype(complex)
    vtest = np.array([i.split() for i in betstest]).astype(complex)
    k1tr = [[kernel1[i,j] for j in trrange] for i in trrange]
    k1te = [[kernel1[i,j] for j in trrange] for i in terange]
    k3tr = [[kernel3[i,j] for j in trrange] for i in trrange]
    k3te = [[kernel3[i,j] for j in trrange] for i in terange]

    # Extract the 10 non-equivalent components xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz; include degeneracy.
    bettrain = np.zeros((nt,10),dtype=complex)
    bettest = np.zeros((ns,10),dtype=complex)
    for i in xrange(nt):
        bettrain[i] = np.array( [vtrain[i][0],vtrain[i][1]*np.sqrt(3.0),vtrain[i][2]*np.sqrt(3.0),vtrain[i][4]*np.sqrt(3.0),vtrain[i][5]*np.sqrt(6.0),vtrain[i][8]*np.sqrt(3.0),vtrain[i][13],vtrain[i][14]*np.sqrt(3.0),vtrain[i][17]*np.sqrt(3.0),vtrain[i][26]],dtype = complex)
    for i in xrange(ns):
        bettest[i] = np.array( [vtest[i][0],vtest[i][1]*np.sqrt(3.0),vtest[i][2]*np.sqrt(3.0),vtest[i][4]*np.sqrt(3.0),vtest[i][5]*np.sqrt(6.0),vtest[i][8]*np.sqrt(3.0),vtest[i][13],vtest[i][14]*np.sqrt(3.0),vtest[i][17]*np.sqrt(3.0),vtest[i][26]],dtype = complex)

    # Unitary transormation matrix from Cartesian to spherical (l=1,m=-1,0,+1 | l=3,m=-3,-2,-1,0,+1,+2,+3), Condon-Shortley convention; include degeneracy.
    CS = np.array([[-3.0/np.sqrt(30.0),0.0,3.0/np.sqrt(30.0),1.0/np.sqrt(8.0),0.0,-3.0/np.sqrt(120.0),0.0,3.0/np.sqrt(120.0),0.0,-1.0/np.sqrt(8.0)],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),-1.0j/np.sqrt(8.0),0.0,1.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(120.0),0.0,-1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,1.0/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),-1.0/np.sqrt(8.0),0.0,-1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(8.0)],[0.0,0.0,0.0,0.0,-1.0j/np.sqrt(12.0),0.0,0.0,0.0,1.0j/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),0.0,0.0,4.0/np.sqrt(120.0),0.0,-4.0/np.sqrt(120.0),0.0,0.0],[3.0j/np.sqrt(30.0),0.0,3.0j/np.sqrt(30.0),1.0j/np.sqrt(8.0),0.0,3.0j/np.sqrt(120.0),0.0,3.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,-1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,-1.0/np.sqrt(12.0),0.0],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),0.0,0.0,-4.0j/np.sqrt(120),0.0,-4.0j/np.sqrt(120),0.0,0.0],[0.0,-3.0/np.sqrt(15.0),0.0,0.0,0.0,0.0,2.0/np.sqrt(10.0),0.0,0.0,0.0]],dtype=complex)
    degeneracy = [1.0,np.sqrt(3.0),np.sqrt(3.0),np.sqrt(3.0),np.sqrt(6.0),np.sqrt(3.0),1.0,np.sqrt(3.0),np.sqrt(3.0),1.0]
    for i in xrange(10):
        CS[i] = CS[i] * degeneracy[i]
    # Transformation matrix from complex to real spherical harmonics (l=1,m=-1,0,+1).
    CR1 = np.array([[1.0j,0.0,1.0j],[0.0,np.sqrt(2.0),0.0],[1.0,0.0,-1.0]],dtype=complex) / np.sqrt(2.0)
    # Transformation matrix from complex to real spherical harmonics (l=3,m=-3,-2,-1,0,+1,+2,+3).
    CR3 =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,1.0j],[0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0],[0.0,0.0,1.0j,0.0,1.0j,0.0,0.0],[0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0],[0.0,0.0,1.0,0.0,-1.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0,0.0,0.0,-1.0]],dtype=complex) / np.sqrt(2.0)

    # Extract the complex spherical components (l=1,l=3) of the hyperpolarizabilities.
    vtrain1 = np.zeros((nt,3),dtype=complex)        # m = -1,0,+1
    vtest1  = np.zeros((ns,3),dtype=complex)        # m = -1,0,+1
    vtrain3 = np.zeros((nt,7),dtype=complex)        # m = -3,-2,-1,0,+1,+2,+3
    vtest3  = np.zeros((ns,7),dtype=complex)        # m = -3,-2,-1,0,+1,+2,+3
    for i in xrange(nt):
        dotpr = np.dot(bettrain[i],CS)
        vtrain1[i] = dotpr[0:3]
        vtrain3[i] = dotpr[3:]
    for i in xrange(ns):
        dotpr = np.dot(bettest[i],CS)
        vtest1[i] = dotpr[0:3]
        vtest3[i] = dotpr[3:]

    # Convert the complex spherical components into real spherical components.
    vtrain1 = np.concatenate(np.array([np.real(np.dot(CR1,vtrain1[i])) for i in xrange(nt)],dtype=float)).astype(float)
    vtest1  = np.concatenate(np.array([np.real(np.dot(CR1,vtest1[i]))  for i in xrange(ns)],dtype=float)).astype(float)
    vtrain3 = np.concatenate(np.array([np.real(np.dot(CR3,vtrain3[i])) for i in xrange(nt)],dtype=float)).astype(float)
    vtest3  = np.concatenate(np.array([np.real(np.dot(CR3,vtest3[i]))  for i in xrange(ns)],dtype=float)).astype(float)

    ktrain1 = np.zeros((3*nt,3*nt),dtype=float)
    ktrain3 = np.zeros((7*nt,7*nt),dtype=float)
    ktrainpred1 = np.zeros((3*nt,3*nt),dtype=float)
    ktrainpred3 = np.zeros((7*nt,7*nt),dtype=float)
    for i in xrange(nt):
        for j in xrange(nt):
            k1rtr = k1tr[i][j]
            k3rtr = k3tr[i][j]
            for al in xrange(3):
                for be in xrange(3):
                    aval = 3*i + al
                    bval = 3*j + be
                    ktrain1[aval][bval] = k1rtr[al][be] + lm1*(aval==bval)
                    ktrainpred1[aval][bval] = k1rtr[al][be]
            for al in xrange(7):
                for be in xrange(7):
                    aval = 7*i + al
                    bval = 7*j + be
                    ktrain3[aval][bval] = k3rtr[al][be] + lm3*(aval==bval)
                    ktrainpred3[aval][bval] = k3rtr[al][be]

    # Invert training kernels.
    invktrvec1 = scipy.linalg.solve(ktrain1,vtrain1)
    invktrvec3 = scipy.linalg.solve(ktrain3,vtrain3)

    # Build testing kernels.
    ktest1 = np.zeros((3*ns,3*nt),dtype=float)
    ktest3 = np.zeros((7*ns,7*nt),dtype=float)
    for i in xrange(ns):
        for j in xrange(nt):
            k1rte = k1te[i][j]
            k3rte = k3te[i][j]
            for al in xrange(3):
                for be in xrange(3):
                    aval = 3*i + al
                    bval = 3*j + be
                    ktest1[aval][bval] = k1rte[al][be]
            for al in xrange(7):
                for be in xrange(7):
                    aval = 7*i + al
                    bval = 7*j + be
                    ktest3[aval][bval] = k3rte[al][be]

# Predict on train data set.
#outvec1 = np.dot(ktrainpred1,invktrvec1)
#outvec3 = np.dot(ktrainpred3,invktrvec3)
## Print out errors and diagnostics.
#intrins_dev1   = np.std(vtrain1)
#intrins_error1 = 100.0 * np.sqrt(np.sum((outvec1-vtrain1)**2)/(3*nt))/intrins_dev1
#intrins_dev3   = np.std(vtrain3)
#intrins_error3 = 100.0 * np.sqrt(np.sum((outvec3-vtrain3)**2)/(7*nt))/intrins_dev3
## Convert the predicted full tensor back to Cartesian coordinates.
#outvec1s = outvec1.reshape((nt,3))
#outvec3s = outvec3.reshape((nt,7))
#outsphr1 = np.zeros((nt,3),dtype=complex)
#outsphr3 = np.zeros((nt,7),dtype=complex)
#betsphe = np.zeros((nt,10),dtype=complex)
#betcart = np.zeros((nt,10),dtype=float)
#betas = np.zeros((nt,27),dtype=float)
#for i in xrange(nt):
#    outsphr1[i] = np.dot(np.conj(CR1).T,outvec1s[i])
#    outsphr3[i] = np.dot(np.conj(CR3).T,outvec3s[i])
#    betsphe[i] = [outsphr1[i][0],outsphr1[i][1],outsphr1[i][2],outsphr3[i][0],outsphr3[i][1],outsphr3[i][2],outsphr3[i][3],outsphr3[i][4],outsphr3[i][5],outsphr3[i][6]]
#    betcart[i] = np.real(np.dot(betsphe[i],np.conj(CS).T))
#predcart = np.concatenate([ [betcart[i][0],betcart[i][1]/np.sqrt(3.0),betcart[i][2]/np.sqrt(3.0),betcart[i][1]/np.sqrt(3.0),betcart[i][3]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][2]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][5]/np.sqrt(3.0),betcart[i][1]/np.sqrt(3.0),betcart[i][3]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][3]/np.sqrt(3.0),betcart[i][6],betcart[i][7]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][7]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][2]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][5]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][7]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][5]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][9]] for i in xrange(nt)]).astype(float)
# Print out errors and diagnostics.
#testcart = np.real(np.concatenate(vtrain)).astype(float)
#intrins_dev = np.std(np.split(testcart,nt),axis=0)
#intrins_error = 100*np.sqrt(np.mean(np.sum(np.split((predcart-testcart)**2,nt),axis=0)/nt/(intrins_dev**2)))
#print " TRAIN STD  (l=1) = %.4f"%intrins_dev1
#print " TRAIN STD  (l=3) = %.4f"%intrins_dev3
#print " TRAIN STD  CARTE = %.4f"%np.sqrt(np.mean(intrins_dev**2))
#print " TRAIN RMSE (l=1) = %.4f %%"%intrins_error1
#print " TRAIN RMSE (l=3) = %.4f %%"%intrins_error3
#print " TRAIN RMSE CARTE = %.4f %%"%intrins_error

    # Predict on test data set..
    outvec1 = np.dot(ktest1,invktrvec1)
    outvec3 = np.dot(ktest3,invktrvec3)
    # Convert the predicted full tensor back to Cartesian coordinates.
    outvec1s = outvec1.reshape((ns,3))
    outvec3s = outvec3.reshape((ns,7))
    outsphr1 = np.zeros((ns,3),dtype=complex)
    outsphr3 = np.zeros((ns,7),dtype=complex)
    betsphe = np.zeros((ns,10),dtype=complex)
    betcart = np.zeros((ns,10),dtype=float)
    betas = np.zeros((ns,27),dtype=float)
    for i in xrange(ns):
        outsphr1[i] = np.dot(np.conj(CR1).T,outvec1s[i])
        outsphr3[i] = np.dot(np.conj(CR3).T,outvec3s[i])
        betsphe[i] = [outsphr1[i][0],outsphr1[i][1],outsphr1[i][2],outsphr3[i][0],outsphr3[i][1],outsphr3[i][2],outsphr3[i][3],outsphr3[i][4],outsphr3[i][5],outsphr3[i][6]]
        betcart[i] = np.real(np.dot(betsphe[i],np.conj(CS).T))
    predcart = np.concatenate([ [betcart[i][0],betcart[i][1]/np.sqrt(3.0),betcart[i][2]/np.sqrt(3.0),betcart[i][1]/np.sqrt(3.0),betcart[i][3]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][2]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][5]/np.sqrt(3.0),betcart[i][1]/np.sqrt(3.0),betcart[i][3]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][3]/np.sqrt(3.0),betcart[i][6],betcart[i][7]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][7]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][2]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][5]/np.sqrt(3.0),betcart[i][4]/np.sqrt(6.0),betcart[i][7]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][5]/np.sqrt(3.0),betcart[i][8]/np.sqrt(3.0),betcart[i][9]] for i in xrange(ns)]).astype(float)
    intrins_dev1   += np.std(vtest1)**2
    abs_error1 += np.sum((outvec1-vtest1)**2)/(3*ns)
    intrins_dev3   += np.std(vtest3)**2
    abs_error3 += np.sum((outvec3-vtest3)**2)/(7*ns)

# Convert the predicted full tensor back to Cartesian coordinates.
# Print out errors and diagnostics.
#testcart = np.real(np.concatenate(vtest)).astype(float)
#intrins_dev = np.std(np.split(testcart,ns),axis=0)
#intrins_error = 100*np.sqrt(np.mean(np.sum(np.split((predcart-testcart)**2,ns),axis=0)/ns/(intrins_dev**2)))

intrins_dev1 = np.sqrt(intrins_dev1/float(ncycles))
abs_error1 = np.sqrt(abs_error1/float(ncycles))
intrins_error1 = 100*np.sqrt(abs_error1**2/intrins_dev1**2)

intrins_dev3 = np.sqrt(intrins_dev3/float(ncycles))
abs_error3 = np.sqrt(abs_error3/float(ncycles))
intrins_error3 = 100*np.sqrt(abs_error3**2/intrins_dev3**2)

print ""
print "testing data points: ", ns    
print "training data points: ", nt   
print "Results for lambda_1 and lambda_3 = ", lm1, lm3
print "--------------------------------"
print "RESULTS FOR L=1"
print "--------------------------------"
print " TEST STD  = %.6f"%intrins_dev1
print " ABS  RMSE = %.6f"%abs_error1
print " TEST RMSE = %.6f %%"%intrins_error1
print "--------------------------------"
print "RESULTS FOR L=3"
print "--------------------------------"
print " TEST STD  = %.6f"%intrins_dev3
print " ABS  RMSE = %.6f"%abs_error3
print " TEST RMSE = %.6f %%"%intrins_error3

#mean1 = 0.0
#for i in range(ns):
#    mean1 += np.linalg.norm(vtest1.reshape((ns,3))[i])
#mean1 /= float(ns)
#std1 = 0.0
#for i in range(ns):
#    std1 += (np.linalg.norm(vtest1.reshape((ns,3))[i]) - mean1)**2
#std1 = np.sqrt(std1/float(ns))
#abs_error1 = 0.0
#for i in range(ns):
#    abs_error1 += (np.linalg.norm(vtest1.reshape((ns,3))[i]) - np.linalg.norm(outvec1.reshape((ns,3))[i]))**2
#abs_error1 = np.sqrt(abs_error1/float(ns))
#
#mean3 = 0.0
#for i in range(ns):
#    mean3 += np.linalg.norm(vtest3.reshape((ns,7))[i])
#mean3 /= float(ns)
#std3 = 0.0
#for i in range(ns):
#    std3 += (np.linalg.norm(vtest3.reshape((ns,7))[i]) - mean3)**2
#std3 = np.sqrt(std3/float(ns))
#abs_error3 = 0.0
#for i in range(ns):
#    abs_error3 += (np.linalg.norm(vtest3.reshape((ns,7))[i]) - np.linalg.norm(outvec3.reshape((ns,7))[i]))**2
#abs_error3 = np.sqrt(abs_error3/float(ns))
#
#print ""
#print "RESULTS FOR L=1 MODULI"
#print "---------------------------------"
#print "MEAN","STD","RMSE", mean1, std1, abs_error1
#print ""
#print "RESULTS FOR L=3 MODULI"
#print "---------------------------------"
#print "MEAN","STD","RMSE", mean3, std3, abs_error3
