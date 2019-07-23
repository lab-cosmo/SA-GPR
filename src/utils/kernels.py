#!/usr/bin/python
from __future__ import absolute_import
import sys
from builtins import range
from numpy import *
import numpy as np
from sympy.physics.quantum.cg import CG
import scipy
from scipy import special
import utils.read_xyz
from itertools import product
import time
import utils.pow_spec as pow_spec
import utils.com_spe as com_spe
# import pow_spec
# import com_spe

##########################################################################################################

def build_SOAP0_kernels(npoints,lcut,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb,nlist):
    """Compute the L=0 SOAP kernel according to Eqns.(33-34) of Ref. Phys. Rev. B 87, 184115 (2013)"""

    mcut = 2*lcut+1
    divfac = np.array([1.0/float(2*l+1) for l in range(lcut+1)])

    # Precompute spherical harmonics evaluated at the direction of atomic positions
    sph_i6 = np.zeros((npoints,natmax,nspecies,nnmax,lcut+1,mcut), dtype=complex)
    for i in range(npoints):                                    # Loop over configurations
        for ii in range(nat[i]):                                # Loop over all the atomic centers of that configuration
            for ix in range(nspecies):                          # Loop over all the different kind of species
                for iii in range(nneigh[i,ii,ix]):              # Loop over the neighbors of that species around that center of that configuration
                    for l in range(lcut+1):                     # Loop over angular momentum channels
                        for im in range(2*l+1):                 # Loop over the 2*l+1 components of the l subspace
                            m = im-l
                            sph_i6[i,ii,ix,iii,l,im] = special.sph_harm(m,l,phi[i,ii,ix,iii],theta[i,ii,ix,iii])
    sph_j6 = np.conj(sph_i6)

    # Precompute the kernel of local environments considering each atom species to be independent from each other
    skernel = np.zeros((npoints,npoints,natmax,natmax), dtype=float)
    for i in range(npoints):
        for j in range(npoints):
            for ii in range(nat[i]):
                for jj in range(nat[j]):
                    # Compute power spectrum I(x,x') for each atomic species
                    ISOAP = np.zeros((nspecies,lcut+1,mcut,mcut),dtype=complex)
                    for ix in range(nspecies):

                        # Precompute modified spherical Bessel functions of the first kind
                        sph_in = np.zeros((nneigh[i,ii,ix],nneigh[j,jj,ix],lcut+1),dtype=complex)
                        for iii in range(nneigh[i,ii,ix]):
                             for jjj in range(nneigh[j,jj,ix]):
                                 sph_in[iii,jjj,:] = special.spherical_in(lcut,length[i,ii,ix,iii]*length[j,jj,ix,jjj])

                        # Perform contraction over neighbour indexes
                        ISOAP[ix,:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                                         efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                         sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:]   )

                    # Compute the dot product of power spectra contracted over l,m,k and summing over all pairs of atomic species a,b
                    skernel[i,j,ii,jj] = np.real(com_spe.combine_spectra(lcut,mcut,nspecies,ISOAP,divfac))

    # Compute global kernel between structures, averaging over all the (normalized) kernels of local environments
    kernel = np.zeros((npoints,npoints),dtype=float)
    kloc = np.zeros((npoints,npoints,natmax,natmax),dtype=float)
    for i in range(npoints):
        for j in range(npoints):
            for ii in range(nat[i]):
                for jj in range(nat[j]):
                    kloc[i,j,ii,jj] = skernel[i,j,ii,jj] / np.sqrt(skernel[i,i,ii,ii]*skernel[j,j,jj,jj])
                    kernel[i,j] += kloc[i,j,ii,jj]
            kernel[i,j] /= float(nat[i]*nat[j])

    kernels = [kloc,kernel]

    # If needed, compute kernels which arise from exponentiation of the local environment kernels to a power n
    skernelsq = np.zeros((npoints,npoints,natmax,natmax),dtype=float)
    skerneln  = np.zeros((npoints,npoints,natmax,natmax),dtype=float)
    for i,j in product(range(npoints),range(npoints)):
        for ii,jj in product(range(nat[i]),range(nat[j])):
            skernelsq[i,j,ii,jj] = skernel[i,j,ii,jj]*skernel[i,j,ii,jj]
    for n in nlist:
        if n!=0:
            for i,j in product(range(npoints),range(npoints)):
                for ii,jj in product(range(nat[i]),range(nat[j])):
                    skerneln[i,j,ii,jj] = skernel[i,j,ii,jj]
            for m in range(1,n):
                for i,j in product(range(npoints),range(npoints)):
                    for ii,jj in product(range(nat[i]),range(nat[j])):
                        skerneln[i,j,ii,jj] = skerneln[i,j,ii,jj]*skernelsq[i,j,ii,jj]
            # Compute the nth kernel.
            kerneln = np.zeros((npoints,npoints),dtype=float)
            for i,j in product(range(npoints),range(npoints)):
                for ii,jj in product(range(nat[i]),range(nat[j])):
                    kerneln[i,j] += skerneln[i,j,ii,jj] / np.sqrt(skerneln[i,i,ii,ii]*skerneln[j,j,jj,jj])
        else:
            kerneln = np.zeros((npoints,npoints),dtype=float)
            for i,j in product(range(npoints),range(npoints)):
                for ii,jj in product(range(nat[i]),range(nat[j])):
                    kerneln[i,j] = kernel[i,j]
        kernels.append(kerneln)

    return [kernels]

#############################################################################################################

def build_SOAP_kernels(lval,npoints,lcut,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb,nlist):
    """Compute spherical tensor SOAP kernel of order lval>0 according to Eqns.(S15-S16) of the Suppl. Info. of Ref. arXiv:1709.06757 (2017)"""

    mcut = 2*lcut+1
    divfac = np.array([np.sqrt(1.0/float(2*l + 1)) for l in range(lcut+1)])

    # Precompute the Clebsch-Gordan coefficients
    CG1 = np.zeros((lcut+1,lcut+1,mcut,2*lval+1),dtype=float)
    for l,l1 in product(range(lcut+1),range(lcut+1)):
        for im,iim in product(range(2*l+1), range(2*lval+1)):
            CG1[l,l1,im,iim] = CG(lval,iim-lval,l1,im-l-iim+lval,l,im-l).doit()

    CG2 = np.zeros((lcut+1,lcut+1,mcut,mcut,2*lval+1,2*lval+1),dtype=float)
    for l,l1 in product(range(lcut+1),range(lcut+1)):
        for im,ik,iim,iik in product(range(2*l+1),range(2*l+1),range(2*lval+1),range(2*lval+1)):
            CG2[l,l1,im,ik,iim,iik] = CG1[l,l1,im,iim] * CG1[l,l1,ik,iik] * divfac[l] * divfac[l]

    # Precompute spherical harmonics evaluated at the direction of atomic positions
    sph_i6 = np.zeros((npoints,natmax,nspecies,nnmax,lcut+1,2*lcut+1),dtype=complex)
    for i in range(npoints):
        for ii in range(nat[i]):
            for ix in range(nspecies):
                for iii in range(nneigh[i,ii,ix]):
                    for l in range(lcut+1):
                        for im in range(2*l+1):
                            m = im-l
                            sph_i6[i,ii,ix,iii,l,im] = special.sph_harm(m,l,phi[i,ii,ix,iii],theta[i,ii,ix,iii])
    sph_j6 = conj(sph_i6)

    # Precompute the tensorial kernel of local environments considering each atom species to be independent from each other
    skernel = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1), complex)
    einpath = None
    listl = np.asarray(range(lcut+1))
    ISOAP = np.zeros((nspecies,lcut+1,mcut,mcut),dtype=complex)
    for i in range(npoints):
      for j in range(i+1):
        for ii,jj in product(range(nat[i]),range(nat[j])):
            ISOAP[:] = 0.0
            for ix in range(nspecies):

                # Precompute modified spherical Bessel functions of the first kind
                sph_in = np.zeros((nneigh[i,ii,ix],nneigh[j,jj,ix],lcut+1),dtype=float)
                for iii,jjj in product(range(nneigh[i,ii,ix]),range(nneigh[j,jj,ix])):
                    sph_in[iii,jjj,:] = special.spherical_in(listl, length[i,ii,ix,iii]*length[j,jj,ix,jjj])

                if einpath is None: # only computes einpath once - assuming number of neighbors is roughly constant
                    einpath = np.einsum_path('a,b,abl,alm,blk->lmk',
                                efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:], optimize='optimal' )[0]

                # Perform contraction over neighbour indexes using the optimized path for Einstein summations
                ISOAP[ix,:,:,:] = np.einsum('a,b,abl,alm,blk->lmk',
                                efact[i,ii,ix,0:nneigh[i,ii,ix]], efact[j,jj,ix,0:nneigh[j,jj,ix]], sph_in[:,:,:],
                                sph_i6[i,ii,ix,0:nneigh[i,ii,ix],:,:], sph_j6[j,jj,ix,0:nneigh[j,jj,ix],:,:], optimize=einpath )

            # Make use of a Fortran 90 subroutine to combine the power spectra and the CG coefficients
            skernel[i,j,ii,jj,:,:] = pow_spec.fill_spectra(lval,lcut,mcut,nspecies,ISOAP,CG2)

            # Exploit Hermiticity
            if not j == i :
                skernel[j,i,jj,ii,:,:] = np.conj(skernel[i,j,ii,jj,:,:].T)

    # Precompute normalization factors
    norm = np.zeros((npoints,natmax), dtype=float)
    for i in range(npoints):
        for ii in range(nat[i]):
            norm[i,ii] = 1.0 / np.sqrt(np.linalg.norm(skernel[i,i,ii,ii,:,:]))

    # compute the kernel
    kernel = np.zeros((npoints,npoints,2*lval+1,2*lval+1), dtype=complex)
    kloc = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1),dtype=complex)
    for i,j in product(range(npoints),range(npoints)):
        for ii,jj in product(range(nat[i]),range(nat[j])):
            kloc[i,j,ii,jj,:,:] = skernel[i,j,ii,jj,:,:] * norm[i,ii] * norm[j,jj]
            kernel[i,j,:,:] += kloc[i,j,ii,jj,:,:]
        kernel[i,j] /= float(nat[i]*nat[j])

    kernels = [kloc,kernel]

    # If needed, compute kernels which arise from exponentiation of the local environment kernels to a power n
    skernelsq = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1), complex)
    skerneln = np.zeros((npoints,npoints,natmax,natmax,2*lval+1,2*lval+1),  complex)
    for i,j in product(range(npoints),range(npoints)):
        for ii,jj in product(range(nat[i]),range(nat[j])):
            skernelsq[i,j,ii,jj,:,:] = np.dot(np.conj(skernel[i,j,ii,jj,:,:].T),skernel[i,j,ii,jj,:,:])
    for n in nlist:
        if n!=0:
            for i,j in product(range(npoints),range(npoints)):
                for ii,jj in product(range(nat[i]),range(nat[j])):
                    skerneln[i,j,ii,jj,:,:] = skernel[i,j,ii,jj,:,:]
            for m in range(n):
                for i,j in product(range(npoints),range(npoints)):
                    for ii,jj in product(range(nat[i]),range(nat[j])):
                        skerneln[i,j,ii,jj,:,:] = np.dot(skerneln[i,j,ii,jj,:,:],skernelsq[i,j,ii,jj,:,:])
            for i in range(npoints):
                for ii in range(nat[i]):
                    norm[i,ii] = 1.0 / np.sqrt(np.linalg.norm(skerneln[i,i,ii,ii,:,:]))
            # Compute the nth kernel.
            kerneln = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=complex)
            for i,j in product(range(npoints),range(npoints)):
                for ii,jj in product(range(nat[i]),range(nat[j])):
                    kerneln[i,j,:,:] += skerneln[i,j,ii,jj,:,:] * norm[i,ii] * norm[j,jj]
                kerneln[i,j] /= float(nat[i]*nat[j])
        else:
            kerneln = np.zeros((npoints,npoints,2*lval+1,2*lval+1),dtype=complex)
            for i,j in product(range(npoints),range(npoints)):
                kerneln[i,j,:,:] = kernel[i,j,:,:]
        kernels.append(kerneln)

    return [kernels]

#########################################################################################

def build_kernels(n,ftrs,npoints,sg,lc,rcut,cweight,vrb,centers,nlist):
    """Wrapper for kernel computation."""

    # Interpret the coordinate file
    [coords,cell,all_names] = utils.read_xyz.readftrs(ftrs)
    # Do neighbour list and precompute variables for SOAP power spectrum
    [natmax,nat,nneigh,length,theta,phi,efact,nnmax,nspecies,centers,atom_indexes] = utils.read_xyz.find_neighbours(all_names,coords,cell,rcut,cweight,npoints,sg,centers)

    if (n == 0):

        if (vrb):
            print("Calculating L=0 kernel.")

        [kernels] = build_SOAP0_kernels(npoints,lc,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb,nlist)

    elif (n > 0):

        if (vrb):
            print ("Calculating L= {} kernel.".format(n))

        [kernels] = build_SOAP_kernels(n,npoints,lc,natmax,nspecies,nat,nneigh,length,theta,phi,efact,nnmax,vrb,nlist)

    else:

        print("Kernel rank (L = {}) not recognized!".format(n))
        sys.exit(0)

    print("Kernel built.")

    return [centers,atom_indexes,natmax,nat,kernels]
