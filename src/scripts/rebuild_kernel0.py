#!/usr/bin/env python2
import numpy as np

# MC code
soapl = 0
rcut=5.0
nstruct = 900
nblocks = 9
nel = nstruct/nblocks
mcut = 2*soapl+1
mcut2 = mcut*mcut
nperrow = 2*nel*mcut2
kij = np.zeros((nstruct, nstruct, mcut, mcut),dtype=float)
for i in range(nblocks):
    for j in range(i+1):
        block = np.loadtxt("Block_"+str(i+1)+"-"+str(j+1)+"/kernel"+str(soapl)+"_"+str(nel*2)+"_sigma0.3_lcut6_cutoff"+str(rcut)+"_cweight1.0_n0.txt",dtype=float)        
        for k in xrange(nel):
            for h in xrange(nel):
                for ii in range(mcut):
                    for jj in range(mcut):
                        kij[i*nel+k,j*nel+h,ii,jj] = block[nperrow*k+ mcut2*(nel+h) + (mcut*ii+jj)]
                        kij[j*nel+h,i*nel+k,ii,jj] = block[nperrow*(nel+h) + mcut2*k + (mcut*ii+jj)]


kernel_file = open("kernel"+str(soapl)+"_"+str(nstruct)+"_sigma"+str(0.3)+"_lcut"+str(6)+"_cutoff"+str(rcut)+"_cweight"+str(1.0)+".txt","w")
for i in range(nstruct):
    for j in range(nstruct):
        for ii in range(mcut):
            for jj in range(mcut):
                print >> kernel_file, kij[i,j,ii,jj]
kernel_file.close()
