#!/usr/bin/env python2

from ase.io import read,write
import argparse
import os

parser = argparse.ArgumentParser(description="Separate XYZ file into blocks for kernel calculation")
parser.add_argument("-f", "--file",   type=str,    required=True, help="Files to read in")
parser.add_argument("-n", "--nblock", type=int,    required=True, help="Number of blocks")
args = parser.parse_args()

nblock = args.nblock
infile = args.file

all_coords = read(infile,':',format='extxyz')
nconfigs = len(all_coords)
blocksize = (nconfigs / nblock) + 1

print "Read in file with %i frames."%(nconfigs)
print "Creating %i blocks."%(nblock)
print "Each block will contain (up to) %i frames."%(blocksize)

for i in xrange(nblock):
    for j in xrange(i+1):
        dirname = 'Block_' + str(i) + '_' + str(j)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        imin = i*blocksize
        imax = min(i*blocksize + blocksize,nconfigs)
        jmin = j*blocksize
        jmax = min(j*blocksize + blocksize,nconfigs)
        coords_out = [all_coords[ij] for ij in range(imin,imax)+range(jmin,jmax)]
        write(dirname + '/coords.xyz',coords_out,format='extxyz')
