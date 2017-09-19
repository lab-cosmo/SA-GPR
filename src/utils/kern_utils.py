#!/usr/bin/python

import numpy as np
import sys
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
    if size>1:
        kernel = np.zeros((ndata,ndata,size,size),dtype=float)
        k=0
        for i in xrange(ndata):
            for j in xrange(ndata):
                for iim in xrange(size):
                    for jjm in xrange(size):
                        kernel[i,j,iim,jjm] = kernel_flatten[k]
                        k += 1
    else:
        kernel = np.zeros((ndata,ndata),dtype=float)
        k=0
        for i in xrange(ndata):
            for j in xrange(ndata):
                kernel[i,j] = kernel_flatten[k]
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
    if size>1:
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
    else:
        ktrain = np.real(ktr) + lm*np.identity(nt)
        ktrainpred = np.real(ktr)

    return [ktrain,ktrainpred]

###############################################################################################################################

def build_testing_kernel(ns,nt,size,kte):
    # Build testing kernel
    if size>1:
        ktest = np.zeros((size*ns,size*nt),dtype=float)
        for i in xrange(ns):
            for j in xrange(nt):
                krte = kte[i][j]
                for al in xrange(size):
                    for be in xrange(size):
                        aval = size*i + al
                        bval = size*j + be
                        ktest[aval][bval] = krte[al][be]
    else:
        ktest = np.real(kte)

    return ktest

###############################################################################################################################

def partition_spherical_components(train,test,CS,CR,sizes,ns,nt):
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
    if sum(sizes)>1:
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
    else:
        for i in xrange(nt):
            vtrain[0][i] = np.dot(train[i],CS)
        for i in xrange(ns):
            vtest[0][i]  = np.dot(test[i],CS)

    # Convert the complex spherical components into real spherical components.
    vtrain_out = []
    vtest_out = []
    for i in xrange(len(vtrain)):
        if (CR[i] is None):
            vtrain_out.append(vtrain[i])
            vtest_out.append( vtest[i] )
        else:
            vtrain_out.append(np.concatenate(np.array([np.real(np.dot(CR[i],vtrain[i][j])) for j in xrange(nt)],dtype=float)).astype(float))
            vtest_out.append( np.concatenate(np.array([np.real(np.dot(CR[i],vtest[i][j]))  for j in xrange(ns)],dtype=float)).astype(float))

    return [vtrain_out,vtest_out]

###############################################################################################################################

def get_non_equivalent_components(train,test):
    # Get the non-equivalent components for a tensor, along with their degeneracies.
    nt = len(train)
    ns = len(test)
    rank = int(np.log(len(train[0])) / np.log(3.0))
    # For each element we assign a label.
    labels = []
    labels.append(np.zeros(rank,dtype=int))
    for i in xrange(1,len(train[0])):
        lbl = list(labels[i-1])
        lbl[rank-1] += 1
        for j in xrange(rank-1,-1,-1):
            if (lbl[j] > 2):
                lbl[j] = 0
                lbl[j-1] += 1
        labels.append(np.array(lbl))
    # Now go through these and find their degeneracies.
    masks = np.zeros(len(train[0]))
    return_list = []
    for i in xrange(len(train[0])):
        return_list.append([])
    for i in xrange(len(train[0])):
        lb1 = sorted(labels[i])
        # Compare this label with all of the previous ones, and see if it's the same within permutation.
        unique = True
        for j in xrange(0,i-1):
            if ((lb1 == sorted(labels[j])) & (unique==True)):
                # They are the same.
                unique = False
                masks[j] += 1
                return_list[j].append(i)
        if (unique):
            masks[i] = 1
            return_list[i].append(i)
    unique_vals = 0
    for j in xrange(len(train[0])):
        if (masks[j] > 0):
            unique_vals += 1
    out_train = np.zeros((nt,unique_vals),dtype=complex)
    out_test  = np.zeros((ns,unique_vals),dtype=complex)
    mask1 = np.zeros(unique_vals,dtype=float)
    return_out = []
    for i in xrange(nt):
        k = 0
        for j in xrange(len(train[i])):
            if (masks[j] > 0):
                out_train[i][k] = train[i][j] * np.sqrt(float(masks[j]))
                mask1[k] = np.sqrt(float(masks[j]))
                k += 1
    for i in xrange(ns):
        k = 0
        for j in xrange(len(test[i])):
            if (masks[j] > 0):
                out_test[i][k] = test[i][j] * np.sqrt(float(masks[j]))
                k += 1
    for j in xrange(len(test[0])):
        if (masks[j] > 0):
            return_out.append(return_list[j])

    return [out_train,out_test,mask1,return_out]
    

###############################################################################################################################

def complex_to_real_transformation(sizes):
    # Transformation matrix from complex to real spherical harmonics.

    matrices = []
    for i in xrange(len(sizes)):
        lval = (sizes[i]-1)/2
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in xrange( (sizes[i]-1)/2 ):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[(sizes[i]-1)/2][(sizes[i]-1)/2] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

###############################################################################################################################

def partition_kernels_properties(data,kernels,trrange,terange):
    # Partition kernels and properties for training and testing.
    train  = [data[i] for i in trrange]
    test   = [data[i] for i in terange]
    vtrain = np.array([i.split() for i in train]).astype(float)
    vtest  = np.array([i.split() for i in test]).astype(float)
    ktr = []
    kte = []
    for kernel in kernels:
        kttr    = [[kernel[i,j] for j in trrange] for i in trrange]
        ktte    = [[kernel[i,j] for j in trrange] for i in terange]
        ktr.append(kttr)
        kte.append(ktte)

    return [vtrain,vtest,ktr,kte]

###############################################################################################################################

def spherical_to_cartesian(outvec,sizes,ns,CR,CS,mask1,mask2):
    # Convert the spherical tensor representation back to Cartesian.

    outvecs = []
    outsphr = []
    sphe = np.zeros((ns,sum(sizes)),dtype=complex)
    cart = np.zeros((ns,sum(sizes)),dtype=float)
    for j in xrange(len(sizes)):
        outvecs.append(outvec[j].reshape((ns,sizes[j])))
        outsphr.append(np.zeros((ns,sizes[j]),dtype=complex))
    for i in xrange(ns):
        for j in xrange(len(sizes)):
            outsphr[j][i] = np.dot(np.conj(CR[j]).T,outvecs[j][i])
        sphe[i] = np.concatenate([outsphr[j][i] for j in xrange(len(sizes))])
        cart[i] = np.real(np.dot(sphe[i],np.conj(CS).T))

        predcart = []
        for i in xrange(ns):
            crt = np.zeros(len(np.concatenate(mask2)),dtype=float)
            for j in xrange(len(mask2)):
                for k in xrange(len(mask2[j])):
                    crt[mask2[j][k]] = cart[i][j] / mask1[j]
            predcart.append(crt)
        predcart = np.concatenate(predcart)

    return predcart

###############################################################################################################################

def get_CS_matrix(rank,mask1,mask2):
    # Get Cartesian to spherical tensor transformation matrix.
    if (rank==0):
        CS = np.array([1.0],dtype=complex)
#        for i in xrange(1):
#            CS[i] = CS[i] * mask1[i]
    elif (rank==1):
        CS = np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0)
    elif (rank==2):
        CS = np.array([[-1.0/np.sqrt(3.0),0.5,0.0,-1.0/np.sqrt(6.0),0.0,0.5],[0.0,-0.5j,0.0,0.0,0.0,0.5j],[0.0,0.0,0.5,0.0,-0.5,0.0],[-1.0/np.sqrt(3.0),-0.5,0.0,-1.0/np.sqrt(6.0),0.0,-0.5],[0.0,0.0,-0.5j,0.0,-0.5j,0.0],[-1.0/np.sqrt(3.0),0.0,0.0,2.0/np.sqrt(6.0),0.0,0.0]],dtype = complex)
    elif (rank==3):
        CS = np.array([[-3.0/np.sqrt(30.0),0.0,3.0/np.sqrt(30.0),1.0/np.sqrt(8.0),0.0,-3.0/np.sqrt(120.0),0.0,3.0/np.sqrt(120.0),0.0,-1.0/np.sqrt(8.0)],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),-1.0j/np.sqrt(8.0),0.0,1.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(120.0),0.0,-1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,1.0/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),-1.0/np.sqrt(8.0),0.0,-1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(8.0)],[0.0,0.0,0.0,0.0,-1.0j/np.sqrt(12.0),0.0,0.0,0.0,1.0j/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),0.0,0.0,4.0/np.sqrt(120.0),0.0,-4.0/np.sqrt(120.0),0.0,0.0],[3.0j/np.sqrt(30.0),0.0,3.0j/np.sqrt(30.0),1.0j/np.sqrt(8.0),0.0,3.0j/np.sqrt(120.0),0.0,3.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,-1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,-1.0/np.sqrt(12.0),0.0],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),0.0,0.0,-4.0j/np.sqrt(120),0.0,-4.0j/np.sqrt(120),0.0,0.0],[0.0,-3.0/np.sqrt(15.0),0.0,0.0,0.0,0.0,2.0/np.sqrt(10.0),0.0,0.0,0.0]],dtype=complex)
    else:
        print "This L value is not yet accounted for!"
        sys.exit(0)
    for i in xrange(len(mask1)):
        CS[i] = CS[i] * mask1[i]

    return CS
    #TODO: The code below is in progress. It could perhaps be done iteratively.
        

    if (rank==0):
        CS = np.array([1.0],dtype=complex)
        for i in xrange(1):
            CS[i] = CS[i] * mask1[i]
    else:
        # One way to do this is to build it up from the L=1 rank to the appropriate one.
        # This may or may not be the most efficient way, but it might be the easiest to code for now.
        cmatr = []
        cmatr.append(np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0))
        if (rank>1):
            for i in xrange(2,rank+1):
                # Generate the next transformation matrix by using the previous one.
                cs_new = np.zeros((3**rank,sum(get_degen(rank))),dtype=complex)
                # Work out the row and column labels for the new transformation matrix.
                # The row labels are easy, as these are just the Cartesian axes.
                row_labels = []
                row_labels.append(np.zeros(rank,dtype=int))
                for i in xrange(1,3**rank):
                    lbl = list(row_labels[i-1])
                    lbl[rank-1] += 1
                    for j in xrange(rank-1,-1,-1):
                        if (lbl[j] > 2):
                            lbl[j] = 0
                            lbl[j-1] += 1
                    row_labels.append(np.array(lbl))
                # The column labels are trickier; these are based on l and m values.
#                for l in xrange(rank+1):
#                    for m in xrange(-l,l+1):
#                lvals = get_lvals(rank)
#                for l in lvals:
#                    for m in xrange(-l,l+1):
                        
                print "here"
        CS = cmatr[-1]
        if (rank>1):
            # Take care of degeneracies.
            print "here"

    for i in xrange(1):
        CS[i] = CS[i] * mask1[i]

    return CS


###############################################################################################################################

def get_lvals(rank):
    # Get the lvals for a given rank.

    if (rank%2 == 0):
        lvals = [l for l in xrange(0,rank+1,2)]
    else:
        # Odd L
        lvals = [l for l in xrange(1,rank+1,2)]

    return lvals

###############################################################################################################################

def get_degen(rank):
    # Get the degeneracies for a given rank.

    lvals = get_lvals(rank)
    return [2*l+1 for l in lvals]

###############################################################################################################################
