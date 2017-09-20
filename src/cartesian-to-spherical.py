#!usr/bin/python

import argparse
import sys
import numpy as np
import utils.kern_utils

parser = argparse.ArgumentParser(description="Convert Cartesian to spherical tensors.")
parser.add_argument("-f", "--files",nargs='+',help="Files to convert.")
args = parser.parse_args()

if args.files:
    filelist = args.files
else:
    print "Files for transformation must be specified!"
    sys.exit(0)

for fl in filelist:
    print "Converting %s"%fl
    tens=[line.rstrip('\n') for line in open(fl)]
    ndata = len(tens)
    num_elem = len(tens[0].split())
    data = [tens[i] for i in xrange(len(tens))]
    tens = np.array([i.split() for i in data]).astype(float)
    rank = int(np.log(float(num_elem)) / np.log(3.0))
    if num_elem != 3**rank:
        print "Number of tensor elements is incorrect!"
        sys.exit(0)
    # Now transform the tensor into its spherical components.
    # List degeneracies
    if (rank%2 == 0):
        # Even L
        lvals = [l for l in xrange(0,rank+1,2)]
    else:
        # Odd L
        lvals = [l for l in xrange(1,rank+1,2)]
    degen = [2*l + 1 for l in lvals]

    # Extract the non-equivalent components, including degeneracy.
    labels = []
    labels.append(np.zeros(rank,dtype=int))
    for i in xrange(1,len(tens[0])):
        lbl = list(labels[i-1])
        lbl[rank-1] += 1
        for j in xrange(rank-1,-1,-1):
            if (lbl[j] > 2):
                lbl[j] = 0
                lbl[j-1] += 1
        labels.append(np.array(lbl))
    masks = np.zeros(len(tens[0]),dtype=int)
    mask2 = [ [] for i in xrange(len(tens[0]))]
    for i in xrange(len(tens[0])):
        lb1 = sorted(labels[i])
        # Compare this with all previous labels, and see if it's the same within permutation.
        unique = True
        for j in xrange(0,i-1):
            if ((lb1 == sorted(labels[j])) & (unique==True)):
                # They are the same.
                unique = False
                masks[j] += 1
                mask2[j].append(i)
        if (unique):
            masks[i] = 1
            mask2[i].append(i)
    unique_vals = 0
    for j in xrange(len(tens[0])):
        if (masks[j] > 0):
            unique_vals += 1
    with_degen = np.zeros((ndata,unique_vals),dtype=complex)
    mask_out1 = np.zeros(unique_vals,dtype=float)
    mask_out2 = []
    for i in xrange(len(tens)):
        k = 0
        for j in xrange(len(tens[0])):
            if (masks[j]>0):
                with_degen[i][k] = tens[i][j] * np.sqrt(float(masks[j]))
                mask_out1[k] = np.sqrt(float(masks[j]))
                k += 1

    # Find the unitary transformation from Cartesian to spherical tensors.
    CS = utils.kern_utils.get_CS_matrix(rank,mask_out1,mask_out2)
    # Find the transformation matrix from complex to real spherical harmonics.
    CR = utils.kern_utils.complex_to_real_transformation(degen)

    # Get the real spherical harmonics of the tensors.
    vout = []
    for i in xrange(len(lvals)):
        if (degen[i]==1):
            vout.append( np.zeros(ndata,dtype=complex) )
        else:
            vout.append( np.zeros((ndata,degen[i]),dtype=complex) )
    if sum(degen)>1:
        for i in xrange(ndata):
            dotpr = np.dot(with_degen[i],CS)
            k = 0
            for j in xrange(len(degen)):
                vout[j][i] = dotpr[k:k+degen[j]]
                k += degen[j]
    else:
        for i in xrange(ndata):
            vout[0][i] = np.dot(with_degen[i],CS)
    vout_real = []
    for i in xrange(len(vout)):
        if (CR[i] is None):
            vout_real.append(vout[i])
        else:
            vout_real.append(np.concatenate(np.array([np.real(np.dot(CR[i],vout[i][j])) for j in xrange(ndata)],dtype=float)).astype(float))

    # Now print out these components to files.
    for i in xrange(len(lvals)):
        outfile = fl + ".L" + str(lvals[i])
        print "  Outputting to ",outfile
        file_out = open(outfile,"w")
        for j in xrange(ndata):
            if (degen[i]==1):
                to_print = str(vout_real[i][j][0])
            else:
                to_print = (", ".join([str(k) for k in vout_real[i][j*degen[i]:(j+1)*degen[i]]]))
            print >> file_out, to_print
#            if (degen[i]==1):
#                to_print = str(vout_real[i])
#            else:
#                to_print = (", ".join([str(k) for k in vout_real[i][j*degen[i]:(j+1)*degen[i]]]))
#            print to_print
#            print >> file_out, vout_real[i][j]
