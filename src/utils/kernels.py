#!/usr/bin/python

import sys
from numpy import *
import numpy as np
from sympy.physics.quantum.cg import CG
import scipy
from scipy import special
import utils.read_xyz
from itertools import product
import time
import pow_spec

##########################################################################################################

# Build SOAP kernels for L=0

def build_SOAP0_kernels(npoints,lcut,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb):

    mcut = 2*lcut+1
    divfac = np.array([1.0/float(2*l+1) for l in xrange(lcut+1)])

    sph_i6 = np.zeros((npoints,natmax,nspecies,nnmax,lcut+1,mcut), dtype=complex)
    for i in xrange(npoints):                          # over configurations
        for ii in xrange(nat[i]):                      # over all the atoms' centers of that configuration 
            for ix in xrange(nspecies):                # over all the different kind of species 
                for iii in xrange(nneigh[i,ii,ix]):    # over the neighbors of that specie around that center of that configuration 
                    for l in xrange(lcut+1):           # over angular momentum   
                        for im in xrange(2*l+1):       # over z projections
                            m = im-l
                            sph_i6[i,ii,ix,iii,l,im] = special.sph_harm(m,l,phi[i,ii,ix,iii],theta[i,ii,ix,iii])
    sph_j6 = np.conj(sph_i6)

    skernel = np.zeros((npoints,npoints,natmax,natmax), dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            for ii in xrange(nat[i]):
                for jj in xrange(nat[j]):
                    ISOAP = np.zeros((nspecies,lcut+1,mcut,mcut),dtype=complex)
                    for ix in xrange(nspecies):
                        sph_in = np.zeros((nneigh[i,ii,ix],nneigh[j,jj,ix],lcut+1),dtype=complex)
                        for iii in xrange(nneigh[i,ii,ix]):
                             for jjj in xrange(nneigh[j,jj,ix]):
                                 sph_in[iii,jjj,:] = special.sph_in(lcut,length[i,ii,ix,iii]*length[j,jj,ix,jjj])[0]   
                        ISOAP[ix,:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                                         efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                         sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:]     ) 
                    skernel[i,j,ii,jj] = np.sum(np.real(np.einsum('almk,blmk,l->ab', np.conj(ISOAP[:,:,:,:]), ISOAP[:,:,:,:], divfac[:])))

    kernel = np.zeros((npoints,npoints),dtype=float)
    kloc = np.zeros((npoints,npoints,natmax,natmax),dtype=float)
    for i in xrange(npoints):
        for j in xrange(npoints):
            for ii in xrange(nat[i]):
                for jj in xrange(nat[j]):
                    kloc[i,j,ii,jj] = skernel[i,j,ii,jj] / np.sqrt(skernel[i,i,ii,ii]*skernel[j,j,jj,jj]) 
                    kernel[i,j] += kloc[i,j,ii,jj] 
            kernel[i,j] /= float(nat[i]*nat[j])

#    env = open("env01_kernel.txt","w")
#    for ii in range(nat[0]):
#        for jj in range(nat[1]):
#            print >> env, kloc[0,1,ii,jj]
#    env.close()

    return [kernel]

#########################################################################################

# Build SOAP kernel for L>0
from time import time
def build_SOAP_kernels(lval,npoints,lcut,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb):

    mcut = 2*lcut+1
    divfac = np.array([np.sqrt(1.0/float(2*l + 1)) for l in xrange(lcut+1)])
    start = time()
    # precompute CG coefficients
    CG1 = np.zeros((lcut+1,lcut+1,mcut,2*lval+1),dtype=float)
    for l,l1 in product(xrange(lcut+1),xrange(lcut+1)):
        for im,iim in product(xrange(2*l+1), xrange(2*lval+1)):
            CG1[l,l1,im,iim] = CG(lval,iim-lval,l1,im-l-iim+lval,l,im-l).doit()
	
    CG2 = np.zeros((lcut+1,lcut+1,mcut,mcut,2*lval+1,2*lval+1),dtype=float)
    for l,l1 in product(xrange(lcut+1),xrange(lcut+1)):
        for im,ik,iim,iik in product(xrange(2*l+1),xrange(2*l+1),xrange(2*lval+1),xrange(2*lval+1)):
            CG2[l,l1,im,ik,iim,iik] = CG1[l, l1, im, iim] * CG1[l, l1, ik, iik] * divfac[l] * divfac[l]

    print "CG done", time()-start, CG2.sum()
    start=time()
    # compute spherical harmonics
    sph_i6 = np.zeros((npoints,natmax,nspecies,nnmax,lcut+1,2*lcut+1),dtype=complex)
    for i in xrange(npoints):
        for ii in xrange(nat[i]):
            for ix in xrange(nspecies):
                for iii in xrange(nneigh[i,ii,ix]):
                    for l in xrange(lcut+1):
                        for im in xrange(2*l+1):
                            m = im-l
                            sph_i6[i,ii,ix,iii,l,im] = special.sph_harm(m,l,phi[i,ii,ix,iii],theta[i,ii,ix,iii])
    sph_j6 = conj(sph_i6)
    print "SPH done", time()-start, sph_i6.sum()
    start=time()

    # compute local tensorial kernels
    
    skernel = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1), complex)
    einpath = None
    listl = np.asarray(xrange(lcut+1))            
    ISOAP = np.zeros((nspecies,lcut+1,mcut,mcut),dtype=complex)    
    for i,j in product(xrange(npoints),xrange(npoints)):
        for ii,jj in product(xrange(nat[i]),xrange(nat[j])):  
            ISOAP[:] = 0.0          
            for ix in xrange(nspecies):
                sph_in = np.zeros((nneigh[i,ii,ix],nneigh[j,jj,ix],lcut+1),dtype=float)
                for iii,jjj in product(xrange(nneigh[i,ii,ix]),xrange(nneigh[j,jj,ix])):
                     sph_in[iii,jjj,:] = special.spherical_in(listl,length[i,ii,ix,iii]*length[j,jj,ix,jjj])
                if einpath is None: # only computes einpath once - assuming number of neighbors is about constant
                    einpath = np.einsum_path('a,b,abl,alm,blk->lmk',
                                efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:], optimize='greedy' )[0]
                ISOAP[ix,:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                                efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:], optimize=einpath )
            skernel[i,j,ii,jj,:,:] = pow_spec.fill_spectra(lval,lcut,mcut,nspecies,ISOAP,CG2)

    print "KERNEL DONE", time()-start, ISOAP.sum(), skernel.sum()
    start= time()
    # compute normalization factors
    norm = np.zeros((npoints,natmax), dtype=float)
    for i in xrange(npoints):
        for ii in xrange(nat[i]):
            norm[i,ii] = 1.0 / np.sqrt(np.linalg.norm(skernel[i,i,ii,ii,:,:]))

    # compute the kernel
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1), dtype=complex)
    for i,j in product(xrange(npoints),xrange(npoints)):
        for ii,jj in product(xrange(nat[i]),xrange(nat[j])):
            kernel[i,j,:,:] += skernel[i,j,ii,jj,:,:] * norm[i,ii] * norm[j,jj] 
        kernel[i,j] /= float(nat[i]*nat[j])
    print "FINISHED ", time()-start, kernel.sum()
    return [kernel]
 
#########################################################################################

def build_kernels(n,ftrs,vcell,npoints,sg,lc,rcut,cweight,fwidth,vrb,periodic,centers):

    # Interpret the coordinate file.
    [coords,cell,all_names] = utils.read_xyz.readftrs(ftrs,vcell)
    [natmax,nat,nneigh,length,theta,phi,efact,nnmax,nspecies] = utils.read_xyz.find_neighbours(all_names,coords,cell,rcut,cweight,fwidth,npoints,sg,periodic,centers)

    kernels = []
    
    if (n == 0):

        if (vrb):
            print "Calculating L=0 kernel."

        kernels.append(build_SOAP0_kernels(npoints,lc,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb))

    elif (n > 0):

        if (vrb):
            print "Calculating L=%i kernel."%n

        kernels.append(build_SOAP_kernels(n,npoints,lc,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb))

    else:

        print "Kernel rank (L = %i) not recognized!"%n
        sys.exit(0)

    print "Kernel built."

    return kernels
