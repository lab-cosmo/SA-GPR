#!/usr/bin/python

import sys
from numpy import *
import numpy as np
from sympy.physics.quantum.cg import CG
import scipy
from scipy import special
import utils.read_xyz
from timeit import default_timer as tit
from itertools import product
import time
import pow_spec

##########################################################################################################

# Build SOAP training and testing spherical kernels for l=0

def build_SOAP0_kernels(nt,ns,lc,natmax,natr,nate,nntrain,lentrain,thetatrain,
        phitrain,efactrain,nntest,lentest,thetatest,phitest,efactest,nnmax,vrb):

    lcut = 2
    if (lc[0] != 0):
        lcut = lc[0]

    mcut = 2*lcut+1

    Isoaptrain  = zeros((lcut+1,mcut,mcut),dtype=complex)
    Isoaptest   = zeros((lcut+1,mcut,mcut),dtype=complex)
    ktr         = zeros( (nt,nt),dtype = complex)
    kte         = zeros( (ns,nt),dtype = complex)
    divfac      = np.array([np.sqrt(1.0/float(2*l + 1)) for l in xrange(lcut+1)])
    sktrain     = zeros((nt,nt,natmax,natmax),dtype=complex)
    sktest      = zeros((ns,nt,natmax,natmax),dtype=complex)
    sktete      = zeros((ns,natmax),dtype=complex)

    # Build power spectra. Remember that under swapping configurations the power spectrum transforms as I^l_mk(j,i) = I^l*_km(i,j).

    start_tr = time.time()
    if vrb:
        print "Calculating spherical harmonics."

    sph_i6 = np.zeros((nt,natmax,nnmax,lcut+1,2*lcut+1),complex)
    for i in xrange(nt):
        for ii in xrange(natr[i]):
            for iii in xrange(nntrain[i,ii]):
                for l in xrange(lcut+1):
                    for im in xrange(2*l+1):
                        m = im-l
                        sph_i6[i,ii,iii,l,im] = special.sph_harm(m,l,phitrain[i,ii,iii],thetatrain[i,ii,iii])/np.sqrt(np.sqrt(2*l+1))
    sph_j6 = np.conj(sph_i6)

    sph_ti6 = np.zeros((ns,natmax,nnmax,lcut+1,2*lcut+1),complex)
    for i in xrange(ns):
        for ii in xrange(nate[i]):
            for iii in xrange(nntest[i,ii]):
                for l in xrange(lcut+1):
                    for im in xrange(2*l+1):
                        m = im-l
                        sph_ti6[i,ii,iii,l,im] = special.sph_harm(m,l,phitest[i,ii,iii],thetatest[i,ii,iii])/np.sqrt(np.sqrt(2*l+1))
    sph_tj6 = np.conj(sph_ti6)

    if vrb:
        print "Calculating training kernels." 
    sktrain = np.zeros((nt,nt,natmax,natmax),dtype=complex)
    time_in = 0
    start_ps = time.time()
    for i in xrange(nt): 
        for j in xrange(nt):
            for ii in xrange(natr[i]):
                for jj in xrange(natr[j]):  
                    Isoaptrain = zeros((lcut+1,mcut,mcut),dtype=complex)
                    sph_in = np.zeros((nntrain[i,ii],nntrain[j,jj],lcut+1),complex)
                    start_in = time.time()
                    for iii in xrange(nntrain[i,ii]):
                         for jjj in xrange(nntrain[j,jj]):
                             sph_in[iii,jjj,:] = special.sph_in(lcut,lentrain[i,ii,iii]*lentrain[j,jj,jjj])[0]   
                    end_in = time.time()
                    time_in += end_in-start_in
                    Isoaptrain[:,:,:] += np.einsum('a,b,abl,alm,blk->lmk',
                              efactrain[i,ii,0:nntrain[i,ii]],efactrain[j,jj,0:nntrain[j,jj]], sph_in[:,:,:],
                              sph_i6[i,ii,0:nntrain[i,ii],:,:],sph_j6[j,jj,0:nntrain[j,jj],:,:])  
                    sktrain[i,j,ii,jj] = np.einsum('lmk,lmk',   np.conj(Isoaptrain), Isoaptrain)
    end_ps = time.time()
    print "time spent for spherical bessel i_n", time_in
    print "time spent for training power spectrum", end_ps-start_ps 

    # Sum kernel.
    if vrb:
        print "Normalizing training kernels."
    for i in xrange(nt):
        for j in xrange(nt):
            if i==0 and j==4:
               mat = np.zeros( (natr[i],natr[j]),dtype=complex)
            for ii in xrange(natr[i]):
                for jj in xrange(natr[j]):
                    if i==0 and j==4:
                        mat[ii,jj] =  sktrain[i,j,ii,jj] / np.sqrt(sktrain[i,i,ii,ii]*sktrain[j,j,jj,jj])
                    ktr[i,j] += sktrain[i,j,ii,jj] / np.sqrt(sktrain[i,i,ii,ii]*sktrain[j,j,jj,jj])
            ktr[i,j] /= float(natr[i]*natr[j])
    end_tr = time.time()
    print "time spent for training kernel", end_tr-start_tr 

    # Save training kernel matrix
    np.savetxt("ktrain0"+"_"+str(nt)+"_"+str(nt)+".txt", ktr, fmt='%1.12e')

    sktest = np.zeros((ns,nt,natmax,natmax),dtype=complex)
    sktete = np.zeros((ns,natmax),dtype=complex)
    if vrb:
        print "Calculating testing kernels."
    for i in xrange(ns):
        for j in xrange(nt):
            for ii in xrange(nate[i]):
                for jj in xrange(natr[j]):
                    Isoaptest = zeros((lcut+1,mcut,mcut),dtype=complex)
                    sph_in = np.zeros((nntest[i,ii],nntrain[j,jj],lcut+1),complex)
                    for iii in xrange(nntest[i,ii]):
                         for jjj in xrange(nntrain[j,jj]):
                            sph_in[iii,jjj,:] = special.sph_in(lcut,lentest[i,ii,iii]*lentrain[j,jj,jjj])[0]  
                    Isoaptest[:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                         efactest[i,ii,0:nntest[i,ii]],efactrain[j,jj,0:nntrain[j,jj]],sph_in[:,:,:],
                         sph_ti6[i,ii,0:nntest[i,ii],:,:],sph_j6[j,jj,0:nntrain[j,jj],:,:])    
                    sktest[i,j,ii,jj] = np.einsum('lmk,lmk',np.conj(Isoaptest), Isoaptest)
    # Calculate the kernel of an element of the testing set with itself, for normalization.
    for i in xrange(ns):
        for ii in xrange(nate[i]):
            Isoaptest = zeros((lcut+1,mcut,mcut),dtype=complex)
            sph_in = np.zeros((nntest[i,ii],nntest[i,ii],lcut+1),complex)
            for iii in xrange(nntest[i,ii]):
                for jjj in xrange(nntest[i,ii]):
                    sph_in[iii,jjj,:] = special.sph_in(lcut,lentest[i,ii,iii]*lentest[i,ii,jjj])[0]  
            Isoaptest[:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                 efactest[i,ii,0:nntest[i,ii]],efactest[i,ii,0:nntest[i,ii]],sph_in[:,:,:],
                 sph_ti6[i,ii,0:nntest[i,ii],:,:],sph_tj6[i,ii,0:nntest[i,ii],:,:])    
            sktete[i,ii] = np.einsum('lmk,lmk',np.conj(Isoaptest), Isoaptest)               
    # Normalize the kernel.
    if vrb:
        print "Normalizing testing kernels."
    for i in xrange(ns):
        for j in xrange(nt):
            for ii in xrange(nate[i]):
                for jj in xrange(natr[j]):
                    kte[i,j] += sktest[i,j,ii,jj] / np.sqrt(sktete[i,ii] * sktrain[j,j,jj,jj])
            kte[i,j] /= (nate[i]*natr[j])

    # Save testing kernel matrix
    np.savetxt("ktest0"+"_"+str(ns)+"_"+str(nt)+".txt", kte, fmt='%1.12e')

    return [ktr,kte]

#########################################################################################

# Build SOAP training and testing spherical kernels for general l

def build_SOAP_kernels(lval,nt,ns,lc,natmax,natr,nate,nntrain,
        lentrain,thetatrain,phitrain,efactrain,nntest,lentest,thetatest,phitest,efactest,nnmax,vrb):

    lcut=max(2,lval)
    if (lc[lval] != 0):
        lcut = lc[lval]
    mcut = 2*lcut+1

    CGcoeffs    = np.zeros((lcut+1,2*(lcut+1)+1,lcut+1,2*(lcut+1)+1,lcut+1,2*(lcut+1)+1),dtype=float)
    ktr         = np.zeros((nt,nt,2*lval+1,2*lval+1),dtype=complex)
    kte         = np.zeros((ns,nt,2*lval+1,2*lval+1),dtype=complex)
    sktrain     = np.zeros((nt,nt,natmax,natmax,2*lval+1,2*lval+1),dtype=complex)
    sktest      = np.zeros((ns,nt,natmax,natmax,2*lval+1,2*lval+1),dtype=complex)
    sktete      = np.zeros((ns,2*lval+1,2*lval+1),dtype=complex)
    n2tr        = np.zeros((nt,natmax),dtype=float)
    n2te        = np.zeros((ns,natmax),dtype=float)
    divfac      = np.array([np.sqrt(1.0/float(2*l + 1)) for l in xrange(lcut+1)])

    # Precompute Clebsch-Gordan coefficients.
    if vrb:
        print "Computing Clebsch-Gordan coefficients."
    CG2 = np.zeros((lcut+1,lcut+1,mcut,mcut,2*lval+1,2*lval+1),dtype=float)
    for l,l1 in product(xrange(lcut+1),xrange(lcut+1)):
        for im,ik,iim,iik in product(xrange(2*l+1),xrange(2*l+1),xrange(2*lval+1),xrange(2*lval+1)):
            CG2[l,l1,im,ik,iim,iik] = CG(lval,iim-lval,l1,im-l-iim+lval,l,im-l).doit() * CG(lval,iik-lval,l1,ik-l-iik+lval,l,ik-l).doit() * divfac[l] * divfac[l]

    # Build power spectra. Recall that under swapping configurations the power spectrum transforms as I^l_mk(j,i) = I^l*_km(i,j).

    if vrb:
        print "Computing training spherical harmonics."
    sph_i6 = np.zeros((nt,natmax,nnmax,lcut+1,2*lcut+1),complex)
    for i in xrange(nt):
        for ii in xrange(natr[i]):
            for iii in xrange(nntrain[i,ii]):
                for l in xrange(lcut+1):
                    for im in xrange(2*l+1):
                        m = im-l
                        sph_i6[i,ii,iii,l,im] = special.sph_harm(m,l,phitrain[i,ii,iii],thetatrain[i,ii,iii])
    sph_j6 = conj(sph_i6) # precompute conjugate

    if vrb:
        print "Computing training kernels."
    sktrain = np.zeros((nt,nt,natmax,natmax,2*lval+1,2*lval+1),dtype=complex)

    for i,j in product(xrange(nt),xrange(nt)):
        for ii,jj in product(xrange(natr[i]),xrange(natr[j])):
            sph_in = np.zeros((nntrain[i,ii],nntrain[j,jj],lcut+1),dtype=float)
            for iii,jjj,lc in product(xrange(nntrain[i,ii]),xrange(nntrain[j,jj]),xrange(lcut+1)):
                sph_in[iii,jjj,lc] = special.spherical_in(lc,lentrain[i,ii,iii]*lentrain[j,jj,jjj]))

            for lc in xrange(lcut+1):
                for im in xrange(2*lc+1):
                    for ik in xrange(2*lc+1):
                        prod = np.einsum('i,j,ij,i,j->ij',efactrain[i,ii,:],efactrain[j,jj,:],sph_in[:,:,lc],sph_i6[i,ii,:,lc,im],sph_j6[j,jj,:,lc,ik])
                        Isoaptrain = np.linalg.det(prod)
#            Isoaptrain = np.einsum('a,b,abl,alm,blk->lmk',
#                      efactrain[i,ii,0:nntrain[i,ii]],efactrain[j,jj,0:nntrain[j,jj]],sph_in[:,:,:],
#                      sph_i6[i,ii,0:nntrain[i,ii],:,:],sph_j6[j,jj,0:nntrain[j,jj],:,:])
            sktrain[i,j,ii,jj,:,:] = pow_spec.fill_spectra(lval,lcut,mcut,Isoaptrain,CG2)
    print "time spent to combine power spectra ", time_bi,time_bi1,time_bi2,time_bi3
                    
    for i in xrange(nt):
        for ii in xrange(natr[i]):
            n2tr[i,ii] = 1.0 / np.sqrt(np.linalg.norm(sktrain[i,i,ii,ii,:,:]))
    if vrb:
        print "Normalizing training kernels."
    for i,j in product(xrange(nt),xrange(nt)):
        for ii,jj in product(xrange(natr[i]),xrange(natr[j])):
            ktr[i,j,:,:] += sktrain[i,j,ii,jj,:,:] * n2tr[i,ii] * n2tr[j,jj] 
        ktr[i,j] /= (natr[i]*natr[j])

    # Save training kernel matrix
    ktrain2_file = open("ktrain"+str(lval)+"_"+str(nt)+"_"+str(nt)+".txt","w")
    for i in xrange(nt):
        for j in xrange(nt):
            for iim in xrange(2*lval+1):
                for iik in xrange(2*lval+1):
                    print >> ktrain2_file, ktr[i,j,iim,iik]

    # Testing power spectrum and spherical kernel. 
    if vrb:
        print "Computing testing spherical harmonics."   
    sph_ti6 = np.zeros((ns,natmax,nnmax,lcut+1,2*lcut+1),complex)
    for i in xrange(ns):
        for ii in xrange(nate[i]):
            for iii in xrange(nntest[i,ii]):
                for l in xrange(lcut+1):
                    for im in xrange(2*l+1):
                        sph_ti6[i,ii,iii,l,im] = special.sph_harm(im-l,l,phitest[i,ii,iii],thetatest[i,ii,iii])
    sph_tj6 = np.conj(sph_ti6)
    sktest = np.zeros((ns,nt,natmax,natmax,2*lval+1,2*lval+1),dtype=complex)
    sktete = np.zeros((ns,natmax,2*lval+1,2*lval+1),dtype=complex)
    if vrb:
        print "Computing testing kernels."
    for i,j in product(xrange(ns),xrange(nt)):
        for ii,jj in product(xrange(nate[i]),xrange(natr[j])):
#            Isoaptest = zeros((lcut+1,mcut,mcut),dtype=complex)
            sph_in = np.zeros((nntest[i,ii],nntrain[j,jj],lcut+1),dtype=float)
            for iii,jjj,lc in product(xrange(nntest[i,ii]),xrange(nntrain[j,jj]),xrange(lcut+1)):
                 sph_in[iii,jjj,lc] = special.spherical_in(lc,lentest[i,ii,iii]*lentrain[j,jj,jjj])
            Isoaptest = np.einsum('a,b,abl,alm,blk->lmk',
                efactest[i,ii,0:nntest[i,ii]],efactrain[j,jj,0:nntrain[j,jj]],sph_in[:,:,:],
                sph_ti6[i,ii,0:nntest[i,ii],:,:],sph_j6[j,jj,0:nntrain[j,jj],:,:])
            # Build the spherical kernel.
#            PSC = np.conj(Isoaptest)
#            PS = np.zeros((2*lval+1,2*lval+1,lcut+1,lcut+1,mcut,mcut),dtype=complex)
#            for l in xrange(lcut+1):
#                for im,ik,l1 in product(xrange(2*l+1),xrange(2*l+1),xrange(lcut+1)):
#                    for iim,jjm in product(xrange(rg[l,im,l1,0],rg[l,im,l1,1]),xrange(rg[l,ik,l1,0],rg[l,ik,l1,1])):
#                        PS[iim,jjm,l1,l,im,ik] = Isoaptest[l1,im-l+l1-iim+lval,ik-l+l1-jjm+lval]
#            PS = pow_spec.fill_spectra(lval,lcut,mcut,Isoaptest,CG2)
            sktest[i,j,ii,jj,:,:] = pow_spec.fill_spectra(lval,lcut,mcut,Isoaptest,CG2)
#            sktest[i,j,ii,jj,:,:] = np.einsum('lpmkij,lpmkij->ij',CG2[:,:,:,:,:,:],PS[:,:,:,:,:,:])#,PSC[:,:,:])
    # Calculate the kernel of an element of the testing set with itself, for normalization.
    for i in xrange(ns):
        for ii in xrange(nate[i]):
#            Isoaptrain = zeros((lcut+1,mcut,mcut),dtype=complex)
            sph_in = np.zeros((nntest[i,ii],nntest[i,ii],lcut+1),dtype=float)
            for iii,jjj,lc in product(xrange(nntest[i,ii]),xrange(nntest[i,ii]),xrange(lcut+1)):
                 sph_in[iii,jjj,lc] = special.spherical_in(lc,lentest[i,ii,iii]*lentest[i,ii,jjj])
            Isoaptest = np.einsum('a,b,abl,alm,blk->lmk',
                efactest[i,ii,0:nntest[i,ii]],efactest[i,ii,0:nntest[i,ii]],sph_in[:,:,:],
                sph_ti6[i,ii,0:nntest[i,ii],:,:],sph_tj6[i,ii,0:nntest[i,ii],:,:])
            # Build the spherical kernel.
#            PSC = np.conj(Isoaptest)
#            PS = np.zeros((2*lval+1,2*lval+1,lcut+1,lcut+1,mcut,mcut),dtype=complex)
#            for l in xrange(lcut+1):
#                for im,ik,l1 in product(xrange(2*l+1),xrange(2*l+1),xrange(lcut+1)):
#                    for iim,jjm in product(xrange(rg[l,im,l1,0],rg[l,im,l1,1]),xrange(rg[l,ik,l1,0],rg[l,ik,l1,1])):
#                        PS[iim,jjm,l1,l,im,ik] = Isoaptest[l1,im-l+l1-iim+lval,ik-l+l1-jjm+lval]
#            PS = pow_spec.fill_spectra(lval,lcut,mcut,Isoaptest,CG2)
            sktete[i,ii,:,:] = pow_spec.fill_spectra(lval,lcut,mcut,Isoaptest,CG2)
#            sktete[i,ii,:,:] = np.einsum('lpmkij,lpmkij->ij',CG2[:,:,:,:,:,:],PS[:,:,:,:,:,:])#,PSC[:,:,:])
    # Normalize kernels.
    if vrb:
        print "Normalizing testing kernels."
    for i in xrange(ns):
        for ii in xrange(nate[i]):
            n2te[i,ii] = 1.0 / np.sqrt(np.linalg.norm(sktete[i,ii,:,:]))
        for j in xrange(nt):
            for ii in xrange(nate[i]):
                for jj in xrange(natr[j]):
                    kte[i,j,:,:] += sktest[i,j,ii,jj,:,:] * n2te[i,ii] * n2tr[j,jj]
            kte[i,j] /= (nate[i]*natr[j])

    # Save testing kernel matrix
    ktest2_file = open("ktest"+str(lval)+"_"+str(ns)+"_"+str(nt)+".txt","w")
    for i in xrange(ns):
        for j in xrange(nt):
            for iim in xrange(2*lval+1):
                for iik in xrange(2*lval+1):
                    print >> ktest2_file, kte[i,j,iim,iik]


    return [ktr,kte]
 
#########################################################################################

def build_kernels(nums,ftrs,vcell,nt,ns,sg,lc,rcut,cweight,fwidth,vrb):

    # Interpret the coordinate file.
    [coords,cell,all_names] = utils.read_xyz.readftrs(ftrs,vcell)
    [natmax,natr,nate,nntrain,lentrain,thetatrain,phitrain,efactrain,nntest,
            lentest,thetatest,phitest,efactest,nnmax] = utils.read_xyz.find_neighbours(all_names,coords,cell,rcut,cweight,fwidth,ns,nt,sg)

    kernels = []
    for i in range(len(nums)):
        n = nums[i]
        if (n == 0):
            if (vrb):
                print "Calculating L=0 kernel."
            kernels.append(build_SOAP0_kernels(nt,ns,lc,natmax,natr,nate,nntrain,
                lentrain,thetatrain,phitrain,efactrain,nntest,lentest,thetatest,phitest,efactest,nnmax,vrb))
#            apd = build_SOAP_kernels(0,nt,ns,lc,natmax,natr,nate,nntrain,
#                lentrain,thetatrain,phitrain,efactrain,nntest,lentest,thetatest,phitest,efactest,nnmax,vrb)
#            if (n==0):
#                apd = [apd[0][:,:,0,0],apd[1][:,:,0,0]]
#            kernels.append(apd)
        elif (n > 0):
            if (vrb):
                print "Calculating L=%i kernel."%n
            kernels.append(build_SOAP_kernels(n,nt,ns,lc,natmax,natr,nate,nntrain,
                lentrain,thetatrain,phitrain,efactrain,nntest,lentest,thetatest,phitest,efactest,nnmax,vrb))
        else:
            print "Kernel rank (n = %i) not recognized!"%nums[i]
            sys.exit(0)

    print "Basic kernels built."

    return kernels
