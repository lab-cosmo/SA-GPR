#!/usr/bin/env python2
from __future__ import print_function
from builtins import range
import argparse
import sys
import numpy as np
import utils.kern_utils
from ase.io import read,write

parser = argparse.ArgumentParser(description="Convert Cartesian to spherical tensors")
parser.add_argument("-f", "--files",    required=True, help="Files to convert")
parser.add_argument("-o", "--output",   default='',    help="Output file")
parser.add_argument("-p", "--property", required=True, help="Property to convert")
parser.add_argument("-r", "--rank",     required=True, help="Tensor rank")
args = parser.parse_args()

file_to_convert = args.files
property_to_convert = args.property
rank = int(args.rank)
outfile = args.output

if outfile == '':
    outfile = file_to_convert

print("Converting rank-{rank} property {property} in file {file}".format(rank=rank, file=file_to_convert, property=property_to_convert))

# Read in tensor file
ftrs = read(file_to_convert,':')
if rank == 0:
    tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
elif rank == 2:
    tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
else:
    tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
ndata = len(tens)
num_elem = len(tens[0].split())
data = [tens[i] for i in range(len(tens))]
tens = np.array([i.split() for i in data]).astype(float)

# Now transform the tensor into its spherical components

# Get list of degeneracies for these components
if (rank%2 == 0):
    # Even L
    lvals = [l for l in range(0,rank+1,2)]
else:
    # Odd L
    lvals = [l for l in range(1,rank+1,2)]
degen = [2*l + 1 for l in lvals]

# Extract the non-equivalent components, including degeneracy; put these into two new arrays,
# with_degen, which contains the non-equivalent (Cartesian) components of the tensor, weighted
# by their degeneracy, and mask_out1, which contains the degeneracies
labels = []
labels.append(np.zeros(rank,dtype=int))
for i in range(1, len(tens[0])):
    lbl = list(labels[i-1])
    lbl[rank-1] += 1
    for j in range(rank-1, -1, -1):
        if (lbl[j] > 2):
            lbl[j] = 0
            lbl[j-1] += 1
    labels.append(np.array(lbl))
masks = np.zeros(len(tens[0]), dtype=int)
mask2 = [[] for i in range(len(tens[0]))]
for i in range(len(tens[0])):
    lb1 = sorted(labels[i])
    # Compare this with all previous labels, and see if it's the same within permutation
    unique = True
    for j in range(0, i-1):
        if ((lb1 == sorted(labels[j])) & (unique == True)):
            # They are the same
            unique = False
            masks[j] += 1
            mask2[j].append(i)
    if (unique):
        masks[i] = 1
        mask2[i].append(i)
unique_vals = 0
for j in range(len(tens[0])):
    if (masks[j] > 0):
        unique_vals += 1
with_degen = np.zeros((ndata,unique_vals),dtype=complex)
mask_out1 = np.zeros(unique_vals,dtype=float)
mask_out2 = []
for i in range(len(tens)):
    k = 0
    for j in range(len(tens[0])):
        if (masks[j]>0):
            with_degen[i][k] = tens[i][j] * np.sqrt(float(masks[j]))
            mask_out1[k] = np.sqrt(float(masks[j]))
            k += 1

# Find the unitary transformation from Cartesian to spherical tensors
CS = utils.kern_utils.get_CS_matrix(rank,mask_out1,mask_out2)
# Find the transformation matrix from complex to real spherical harmonics
CR = utils.kern_utils.complex_to_real_transformation(degen)

# Get the real spherical harmonics of the tensors by using these transformation matrices
vout = []
for i in range(len(lvals)):
    if (degen[i]==1):
        vout.append( np.zeros(ndata,dtype=complex) )
    else:
        vout.append( np.zeros((ndata, degen[i]), dtype=complex) )
if sum(degen)>1:
    for i in range(ndata):
        dotpr = np.dot(with_degen[i], CS)
        k = 0
        for j in range(len(degen)):
            vout[j][i] = dotpr[k:k+degen[j]]
            k += degen[j]
else:
    for i in range(ndata):
        vout[0][i] = np.dot(with_degen[i], CS)
vout_real = []
for i in range(len(vout)):
    if (CR[i] is None):
        vout_real.append(vout[i])
    else:
        vout_real.append(np.concatenate(np.array([np.real(np.dot(CR[i], vout[i][j])) for j in range(ndata)], dtype=float)).astype(float))

# Print out these components to files
for i in range(len(lvals)):
    prop_out = property_to_convert + "_L" + str(lvals[i])
    print("Outputting property {} to {}.".format(prop_out, outfile))
    for j in range(ndata):
        if (degen[i]==1):
            to_print = str(vout_real[i][j][0])
        else:
            to_print = (" ".join([str(k) for k in vout_real[i][j*degen[i]:(j+1)*degen[i]]]))
        ftrs[j].info[prop_out] = to_print

write(outfile, ftrs)
