#!usr/bin/python

import utils.kernels
import utils.parsing
import argparse
import sys
import numpy as np
import run.sagpr0
import run.sagpr1
import run.sagpr2
import run.sagpr3

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r", "--rank", help="Rank of tensor to learn")
    parser.add_argument("-lm", "--lmda", nargs='+', help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",type=float, help="Fraction of data points used for testing")
    parser.add_argument("-t", "--tensors", help="File containing tensors")
    parser.add_argument("-k", "--kernel",nargs='+', help="Files containing kernels")
    parser.add_argument("-sel", "--select",nargs='+', help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",type=int, help="Number of random training points")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):

    # default values
    lm0 = 0.001
    lm1 = 0.001
    lm2 = 0.001
    lm3 = 0.001
    ftr = 1

    if args.rank:
        rank = int(args.rank)
    else:
        print "Rank of tensor must be specified!"
        sys.exit(0)
 
    lm = [lm0,lm1,lm2,lm3]
    if args.lmda:
        lmlist = args.lmda
        if sum([lmlist[i].count(',') for i in xrange(len(lmlist))]) > 0:
            for i in xrange(len(lmlist)):
                lmlist[i] = lmlist[i].split(',')
            lmlist = np.concatenate(lmlist)
        if (len(lmlist)%2 != 0):
            print "Error: list of lambdas must have the format n,lambda[n],m,lambda[m],..."
            sys.exit(0)
        for i in xrange(len(lmlist)/2):
            nval = int(lmlist[2*i])
            lmval = float(lmlist[2*i+1])
            lm[nval] = lmval

    if args.ftrain:
        ftr = args.ftrain 
    if args.tensors:
        tfile = args.tensors
    else:
        print "Features file must be specified!"
        sys.exit(0)
    # Read in features
    tens=[line.rstrip('\n') for line in open(tfile)]

    kernels = ["","","",""]
    if args.kernel:
        krlist = args.kernel
        if sum([krlist[i].count(',') for i in xrange(len(krlist))]) > 0:
            for i in xrange(len(krlist)):
                krlist[i] = krlist[i].split(',')
            krlist = np.concatenate(krlist)
        if (len(krlist)%2 != 0):
            print "Error: list of kernels must have the format n,kernel[n],m,kernel[m],..."
            sys.exit(0)
        for i in xrange(len(krlist)/2):
            nval = int(krlist[2*i])
            krval = str(krlist[2*i + 1])
            kernels[nval] = krval

    beg = 0
    end = int(len(tens)/2)
    sel = [beg,end]
    if args.select:
        sellist = args.select
        for i in xrange(len(sellist)):
            sel[0] = int(sellist[0])
            sel[1] = int(sellist[1])

    rdm = 0
    if args.random:
        rdm = args.random 

    return [lm,ftr,tens,kernels,sel,rdm,rank]

###############################################################################################################################

# This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.

# Parse input arguments.
args = add_command_line_arguments_learn("SA-GPR")
[lm,fractrain,tens,kernels,sel,rdm,rank] = set_variable_values_learn(args)

# Read-in kernels.
print "Loading kernel matrices..."
kernel = []
for k in xrange(len(kernels)):
    if (kernels[k]!=''):
        kernel.append(np.loadtxt(kernels[k],dtype=float))
    else:
        kernel.append(0)

# Get list of l values.
lvals = []
if (rank%2 == 0):
    lvals = [l for l in xrange(0,rank+1,2)]
else:
    # Odd L
    lvals = [l for l in xrange(1,rank+1,2)]

lms     = [lm[i]     for i in lvals]
kernels = [kernel[i] for i in lvals]

# Call the appropriate subroutine.
if (rank==0):
#    run.sagpr0.do_sagpr0(lvals,[lm[0]],fractrain,tens,[kernel[0]],sel,rdm)
    run.sagpr0.do_sagpr0(lvals,lms,fractrain,tens,kernels,sel,rdm)
elif (rank==1):
#    run.sagpr1.do_sagpr1(lvals,[lm[1]],fractrain,tens,[kernel[1]],sel,rdm)
    run.sagpr1.do_sagpr1(lvals,lms,fractrain,tens,kernels,sel,rdm)
elif (rank==2):
#    run.sagpr2.do_sagpr2(lvals,[lm[0],lm[2]],fractrain,tens,[kernel[0],kernel[2]],sel,rdm)
    run.sagpr2.do_sagpr2(lvals,lms,fractrain,tens,kernels,sel,rdm)
elif (rank==3):
#    run.sagpr3.do_sagpr3(lvals,[lm[1],lm[3]],fractrain,tens,[kernel[1],kernel[3]],sel,rdm)
    run.sagpr3.do_sagpr3(lvals,lms,fractrain,tens,kernels,sel,rdm)
else:
    print "The code is currently not setup for this rank!"
    sys.exit(0)
