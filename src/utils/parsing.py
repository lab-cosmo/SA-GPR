#!/usr/bin/python
import argparse
import sys
import numpy as np


def add_command_line_arguments_tenskernel(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lval", "--lvalue", help="Order of the spherical tensor")
    parser.add_argument("-f", "--features",help="File containing atomic fingerprints (coordinates)")
    parser.add_argument("-sg", "--sigma", type=float, help="Sigma for SOAP kernels")
    parser.add_argument("-lc", "--lcut", type=int, help="lcut for SOAP kernels")
    parser.add_argument("-c", "--cell",help="File containing cell vectors")
    parser.add_argument("-rc", "--rcut", type=float,help="Cutoff value for bulk systems as a fraction of the box length")
    parser.add_argument("-cw", "--cweight", type=float,help="Central atom weight")
    parser.add_argument("-fw", "--fwidth", type=float,help="Width of the radial filtering to atomic densities")
    parser.add_argument("-vr", "--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("-per","--periodic",type=bool, help="Call for periodic systems")
    parser.add_argument("-sub", "--subset", type=float, help="Fraction of the input data set")
    parser.add_argument("-cen", "--center", nargs='+', help="List of atoms to center on (default all)")
    parser.add_argument("-n",   "--nlist", nargs='+', help="List of n values for kernel calculation")
    args = parser.parse_args()
    return args


def set_variable_values_tenskernel(args):

    # Set defaults
    sg = 0.3
    lc = 6
    rc = 3.0
    cw = 1.0
    fw = 1e-10
    per = False
    sub = 1.0
    cen = []

    # Use command-line arguments to set the values of important variables
    if args.lvalue:
        lval = int(args.lvalue)
    else:
        print "Spherical tensor order must be specified!"
        sys.exit(0)

    if args.periodic:
        per = True

    if args.sigma:
        sg = args.sigma

    if args.lcut:
        lc = args.lcut

    if args.cweight:
        cw = args.cweight

    if args.fwidth:
        fw = args.fwidth

    if args.rcut:
        rc = args.rcut

    if args.subset:
        sub = args.subset

    if args.center:
        cent = args.center
        for i in range(len(cent)):
            cen.append(int(cent[i]))

    if args.features:
        ffile = args.features
    else:
        print "Features file must be specified!"
        sys.exit(0)

    # Read in features
    ftrs=[line.rstrip('\n') for line in open(ffile)]

    npoints = int(sub*len(ftrs))

    if args.cell:
        cfile = args.cell
        # Read in unit cell
        vcell=[line.rstrip('\n') for line in open(cfile)]
    else:
        # If we are considering gas-phase systems, we don't need the unit cell.
        vcell = []

    if args.nlist:
        nls = args.nlist
        if sum([nls.count(',') for i in xrange(len(nls))]) > 0:
            for j in xrange(len(nls)):
                nls[j] = nls[j].split(',')
            nls = np.concatenate(nls)
        nlist = []
        for i in xrange(len(nls)):
            nlist.append(int(nls[i]))
        nlist.append(0)
        nlist = set(nlist)
    else:
        nlist = [0]

    return [ftrs,vcell,npoints,lval,sg,lc,rc,cw,fw,args.verbose,per,cen,nlist]


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
    parser.add_argument("-nc",  "--ncycles", type=int, help="Number of cycles for regression with random selection")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):

    # default values
    lm0 = 0.001
    ftr = 1

    if args.rank:
        rank = int(args.rank)
    else:
        print "Rank of tensor must be specified!"
        sys.exit(0)
 
    lm = [lm0 for l in xrange(rank+1)]
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

    if args.ncycles:
        ncycles = args.ncycles
    else:
        ncycles = 1

    return [lm,ftr,tens,kernels,sel,rdm,rank,ncycles]
###############################################################################################################################
