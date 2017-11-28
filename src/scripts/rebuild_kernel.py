#!/usr/bin/env python2
import numpy as np
import sys
import argparse
import math

parser = argparse.ArgumentParser(description="Rebuild kernel from blocks.")
parser.add_argument("-l",  "--lval",help="l-value of kernels to rebuild.")
parser.add_argument("-ns", "--nstruct", type=int,   required=True,       help="Number of structures.")
parser.add_argument("-nb", "--nblocks", type=int,   required=True,       help="Number of blocks.")
parser.add_argument("-rc", "--rcut",    type=float, required=True,       help="Radial cutoff.")
parser.add_argument("-lc", "--lcut",    type=int,   required=True,       help="Angular cutoff.")
parser.add_argument("-sg", "--sigma",   type=float, required=True,       help="Gaussian width.")
parser.add_argument("-cw", "--cweight", type=float, required=True,       help="Central weight.")
args = parser.parse_args()

soapl = int(args.lval)
nstruct = int(args.nstruct)
nblocks = int(args.nblocks)
rcut = float(args.rcut)
lcut = int(args.lcut)
sg = float(args.sigma)
cw = float(args.cweight)

blocksize = int(math.ceil(float(nstruct)/float(nblocks)))
mcut = 2*soapl+1
mcut2 = mcut*mcut
kij = np.zeros((nstruct, nstruct, mcut, mcut),dtype=float)
print "Each block will contain (up to) %i frames."%(blocksize)
for i in xrange(nblocks):
    for j in xrange(i+1):
        imin = i*blocksize
        imax = min(i*blocksize + blocksize,nstruct)
        jmin = j*blocksize
        jmax = min(j*blocksize + blocksize,nstruct)
        di = imax-imin
        dj = jmax-jmin
        thisblocksize = di + dj
        dirname = 'Block_' + str(i) + '_' + str(j)
        block = np.loadtxt(dirname+"/kernel"+str(soapl)+"_"+str(thisblocksize)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+"_n0.txt",dtype=float) 
        for k in xrange(di):
            for h in xrange(dj):
                for ii in xrange(mcut):
                    for jj in xrange(mcut):
                        kij[imin+k,jmin+h,ii,jj] = block[(di+dj)*mcut2*k + mcut*mcut*(di+h) + mcut*ii + jj]
                        kij[jmin+h,imin+k,jj,ii] = block[(di+dj)*mcut2*k + mcut*mcut*(di+h) + mcut*ii + jj]

# Print out kernel file.
kernel_file = open("kernel"+str(soapl)+"_"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt","w")
for i in range(nstruct):
    for j in range(nstruct):
        for ii in range(mcut):
            for jj in range(mcut):
                print >> kernel_file, kij[i,j,ii,jj]
kernel_file.close()
