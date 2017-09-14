#!/usr/bin/python

import numpy as np
from random import shuffle

###############################################################################################################################

def shuffle_data(ndata,sel,rdm,fractrain):

    if rdm == 0:
        trrangemax = np.asarray(range(sel[0],sel[1]),int)
    else:
        data_list = range(ndata)
        shuffle(data_list)
        trrangemax = np.asarray(data_list[:rdm],int).copy()
    terange = np.setdiff1d(range(ndata),trrangemax)

    ns = len(terange)
    ntmax = len(trrangemax)
    nt = int(fractrain*ntmax)
    trrange = trrangemax[0:nt]

    return [ns,nt,ntmax,trrange,terange]

###############################################################################################################################

def unflatten_kernel(ndata,size,kernel_flatten):
    # Unpack kernel into the desired format for the code
    kernel = np.zeros((ndata,ndata,size,size),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):
            for iim in xrange(size):
                for jjm in xrange(size):
                    kernel[i,j,iim,jjm] = kernel_flatten[k]
                    k += 1

    return kernel

###############################################################################################################################

def unflatten_kernel0(ndata,kernel_flatten):
    # Unpack kernel into the desired format for the code
    kernel = np.zeros((ndata,ndata),dtype=float)
    k=0
    for i in xrange(ndata):
        for j in xrange(ndata):
            kernel[i,j] = kernel_flatten[k]
            k += 1

    return kernel

###############################################################################################################################

def build_training_kernel(nt,size,ktr,lm):
    # Build training kernel
    ktrain = np.zeros((size*nt,size*nt),dtype=float)
    ktrainpred = np.zeros((size*nt,size*nt),dtype=float)
    for i in xrange(nt):
        for j in xrange(nt):
            krtr = ktr[i][j]
            for al in xrange(size):
                for be in xrange(size):
                    aval = size*i + al
                    bval = size*j + be
                    ktrain[aval][bval] = krtr[al][be] + lm*(aval==bval)
                    ktrainpred[aval][bval] = krtr[al][be]

    return [ktrain,ktrainpred]

###############################################################################################################################

def build_testing_kernel(ns,nt,size,kte):
    # Build testing kernel
    ktest = np.zeros((size*ns,size*nt),dtype=float)
    for i in xrange(ns):
        for j in xrange(nt):
            krte = kte[i][j]
            for al in xrange(size):
                for be in xrange(size):
                    aval = size*i + al
                    bval = size*j + be
                    ktest[aval][bval] = krte[al][be]

    return ktest

###############################################################################################################################

def partition_spherical_components(train,test,CS,sizes,ns,nt):
    # Extract the complex spherical components of the tensors.
    vtrain = []
    vtest = []
    for i in xrange(len(sizes)):
        if (sizes[i]==1):
            vtrain.append( np.zeros(nt,dtype=complex) )
            vtest.append(  np.zeros(ns,dtype=complex) )
        else:
            vtrain.append( np.zeros((nt,sizes[i]),dtype=complex) )
            vtest.append(  np.zeros((ns,sizes[i]),dtype=complex) )
    for i in xrange(nt):
        dotpr = np.dot(train[i],CS)
        k = 0
        for j in xrange(len(sizes)):
            vtrain[j][i] = dotpr[k:k+sizes[j]]
            k += sizes[j]
    for i in xrange(ns):
        dotpr = np.dot(test[i],CS)
        k = 0
        for j in xrange(len(sizes)):
            vtest[j][i] = dotpr[k:k+sizes[j]]
            k += sizes[j]

    return [vtrain,vtest]

###############################################################################################################################
#
#
#
#        # Extract the complex spherical components (l=0,l=2) of the polarizabilities.
#        vtrain0 = np.zeros(nt,dtype=complex)        # m =       0
#        vtest0  = np.zeros(ns,dtype=complex)        # m =       0
#        vtrain2 = np.zeros((nt,5),dtype=complex)    # m = -2,-1,0,+1,+2
#        vtest2  = np.zeros((ns,5),dtype=complex)    # m = -2,-1,0,+1,+2
#        for i in xrange(nt):
#            dotpr = np.dot(alptrain[i],CS)
#            vtrain0[i] = dotpr[0]
#            vtrain2[i] = dotpr[1:6]
#        for i in xrange(ns):
#            dotpr = np.dot(alptest[i],CS)
#            vtest0[i] = dotpr[0]
#            vtest2[i] = dotpr[1:6]
#
#
#
#        # Extract the complex spherical components (l=1,l=3) of the hyperpolarizabilities.
#        vtrain1 = np.zeros((nt,3),dtype=complex)        # m = -1,0,+1
#        vtest1  = np.zeros((ns,3),dtype=complex)        # m = -1,0,+1
#        vtrain3 = np.zeros((nt,7),dtype=complex)        # m = -3,-2,-1,0,+1,+2,+3
#        vtest3  = np.zeros((ns,7),dtype=complex)        # m = -3,-2,-1,0,+1,+2,+3
#        for i in xrange(nt):
#            dotpr = np.dot(bettrain[i],CS)
#            vtrain1[i] = dotpr[0:3]
#            vtrain3[i] = dotpr[3:]
#        for i in xrange(ns):
#            dotpr = np.dot(bettest[i],CS)
#            vtest1[i] = dotpr[0:3]
#            vtest3[i] = dotpr[3:]
