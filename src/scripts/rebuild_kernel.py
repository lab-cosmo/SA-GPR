#!/usr/bin/env python2
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description="Rebuild kernel from blocks.")
parser.add_argument("-l", "--lval",help="l-value of kernels to rebuild.")
parser.add_argument("-ns", "--nstruct", help="Number of structures.")
parser.add_argument("-nb", "--nblocks", help="Number of blocks.")
parser.add_argument("-rc", "--rcut", help="Radial cutoff.")
parser.add_argument("-lc", "--lcut", help="Angular cutoff.")
parser.add_argument("-sg", "--sigma", help="Gaussian width.")
parser.add_argument("-cw", "--cweight", help="Central weight.")
args = parser.parse_args()

if args.lval:
    soapl = int(args.lval)
else:
    print "L value must be specified!"
    sys.exit(0)

if args.nstruct:
    nstruct = int(args.nstruct)
else:
    print "Number of structures must be specified!"
    sys.exit(0)

if args.nblocks:
    nblocks = int(args.nblocks)
else:
    print "Number of blocks must be specified!"
    sys.exit(0)

if args.rcut:
    rcut = float(args.rcut)
else:
    print "Radial cutoff must be specified!"
    sys.exit(0)

if args.lcut:
    lcut = int(args.lcut)
else:
    print "Angular cutoff must be specified!"
    sys.exit(0)

if args.sigma:
    sg = float(args.sigma)
else:
    print "Gaussian width must be specified!"
    sys.exit(0)

if args.cweight:
    cw = float(args.cweight)
else:
    print "Central weight must be specified!"
    sys.exit(0)

nel = nstruct/nblocks
mcut = 2*soapl+1
mcut2 = mcut*mcut
nperrow = 2*nel*mcut2
kij = np.zeros((nstruct, nstruct, mcut, mcut),dtype=float)
for i in range(nblocks):
    for j in range(i+1):
        block = np.loadtxt("Block_"+str(i+1)+"-"+str(j+1)+"/kernel"+str(soapl)+"_"+str(nel*2)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+"_n0.txt",dtype=float)        
        for k in xrange(nel):
            for h in xrange(nel):
                for ii in range(mcut):
                    for jj in range(mcut):
                        kij[i*nel+k,j*nel+h,ii,jj] = block[nperrow*k+ mcut2*(nel+h) + (mcut*ii+jj)]
                        kij[j*nel+h,i*nel+k,ii,jj] = block[nperrow*(nel+h) + mcut2*k + (mcut*ii+jj)]


kernel_file = open("kernel"+str(soapl)+"_"+str(nstruct)+"_sigma"+str(sg)+"_lcut"+str(lcut)+"_cutoff"+str(rcut)+"_cweight"+str(cw)+".txt","w")
for i in range(nstruct):
    for j in range(nstruct):
        for ii in range(mcut):
            for jj in range(mcut):
                print >> kernel_file, kij[i,j,ii,jj]
kernel_file.close()
