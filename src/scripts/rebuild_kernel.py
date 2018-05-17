#!/usr/bin/env python2
import numpy as np
import sys
import argparse
import math
import time

parser = argparse.ArgumentParser(description="Rebuild kernel from blocks.")
parser.add_argument("-l",  "--lval",                                            help="l-value of kernels to rebuild.")
parser.add_argument("-ns", "--nstruct", type=int,   required=True,              help="Number of structures.")
parser.add_argument("-nb", "--nblocks", type=int,   required=True,              help="Number of blocks.")
parser.add_argument("-rc", "--rcut",    type=float, required=True,              help="Radial cutoff.")
parser.add_argument("-lc", "--lcut",    type=int,   required=True,              help="Angular cutoff.")
parser.add_argument("-sg", "--sigma",   type=float, required=True,              help="Gaussian width.")
parser.add_argument("-cw", "--cweight", type=float, required=True,              help="Central weight.")
parser.add_argument("-att","--atomtype",type=int,   required=False,  default=0, help="Number of identical atoms.")
parser.add_argument("-nat","--natoms",  type=int,   required=False,  default=1, help="Number of identical atoms.")
args = parser.parse_args()

atom_type = int(args.atomtype)
natoms = int(args.natoms)
soapl = int(args.lval)
nstruct = int(args.nstruct)
nblocks = int(args.nblocks)
rcut = float(args.rcut)
lcut = int(args.lcut)
sg = float(args.sigma)
cw = float(args.cweight)



blocksize = int(math.ceil(float(nstruct*natoms)/float(nblocks)))
nblocks = int(math.ceil(float(nstruct*natoms)/float(blocksize)))
mcut = 2*soapl+1
mcut2 = mcut*mcut
kij = np.zeros((nstruct*natoms, nstruct*natoms, mcut, mcut),dtype=float)
print "Each block will contain (up to) %i frames."%(blocksize)
for i in xrange(nblocks):
    for j in xrange(i+1):
        start = time.time()
        imin = i*blocksize
        imax = min(i*blocksize + blocksize,nstruct*natoms)
        jmin = j*blocksize
        jmax = min(j*blocksize + blocksize,nstruct*natoms)
        di = imax-imin
        dj = jmax-jmin
        thisblocksize = di + dj
        dirname = 'Block_' + str(i) + '_' + str(j)
        if atom_type == 0:
    	    block = np.load(dirname+"/kernel"+str(soapl)+"_nconf"+str(thisblocksize)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".npy")
    	    block = np.reshape(block,np.size(block))
            #block = np.loadtxt(dirname+"/kernel"+str(soapl)+"_nconf"+str(thisblocksize)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt",dtype=float) 
        else:
            block = np.load(dirname+"/kernel"+str(soapl)+"_atom"+str(atom_type)+"_nconf"+str(thisblocksize/natoms)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".npy") 
    	    block = np.reshape(block,np.size(block))
            #block = np.loadtxt(dirname+"/kernel"+str(soapl)+"_atom"+str(atom_type)+"_nconf"+str(thisblocksize/natoms)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt",dtype=float) 
        end = time.time()
        print "LOADING TIME = ",end-start
        start = end
        for k in xrange(di):
            for h in xrange(dj):
                for ii in xrange(mcut):
                    for jj in xrange(mcut):
                        kij[imin+k,jmin+h,ii,jj] = block[(di+dj)*mcut2*k + mcut*mcut*(di+h) + mcut*ii + jj]
                        kij[jmin+h,imin+k,jj,ii] = block[(di+dj)*mcut2*k + mcut*mcut*(di+h) + mcut*ii + jj]
        end = time.time()
        print "LOOPING TIME = ",end-start

# Print out kernel file.
if atom_type == 0:
#    kernel_file = open("kernel"+str(soapl)+"_nconf"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt","w")
    kernel_file = open("kernel"+str(soapl)+"_nconf"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".npy","w")
else:
#    kernel_file = open("kernel"+str(soapl)+"_atom"+str(atom_type)+"_nconf"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt","w")
    kernel_file = open("kernel"+str(soapl)+"_atom"+str(atom_type)+"_nconf"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".npy","w")
np.save(kernel_file, kij)
#for i in range(nstruct*natoms):
#    for j in range(nstruct*natoms):
#        for ii in range(mcut):
#            for jj in range(mcut):
#                print >> kernel_file, kij[i,j,ii,jj]
kernel_file.close()
