#!/usr/bin/python
import argparse
import sys
import numpy as np

###############################################################################################################################

def add_command_line_arguments_tenskernel(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lval", "--lvalue",    type=int,   required=True,                   help="Order of the spherical tensor")
    parser.add_argument("-f",    "--features",  type=str,   required=True,                   help="File containing atomic fingerprints (coordinates)")
    parser.add_argument("-sg",   "--sigma",     type=float, default=0.3,                     help="Sigma for SOAP kernels")
    parser.add_argument("-lc",   "--lcut",      type=int,   default=6,                       help="lcut for SOAP kernels")
    parser.add_argument("-c",    "--cell",      type=str,                                    help="File containing cell vectors")
    parser.add_argument("-rc",   "--rcut",      type=float, default=3.0,                     help="Cutoff value for bulk systems as a fraction of the box length")
    parser.add_argument("-cw",   "--cweight",   type=float, default=1.0,                     help="Central atom weight")
    parser.add_argument("-fw",   "--fwidth",    type=float, default=1e-10,                   help="Width of the radial filtering to atomic densities")
    parser.add_argument("-vr",   "--verbose",               action='store_true',             help="Verbose mode")
    parser.add_argument("-per",  "--periodic",              action='store_true',             help="Call for periodic systems")
    parser.add_argument("-sub",  "--subset",    type=float, default=1.0,                     help="Fraction of the input data set")
    parser.add_argument("-cen",  "--center",    type=int,   required=True,       nargs='+',  help="List of atoms to center on (default all)")
    parser.add_argument("-n",    "--nlist",     type=int,   default=[0],         nargs='+',  help="List of n values for kernel calculation")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_tenskernel(args):

    # Use command-line arguments to set the values of important variables
    lval = int(args.lvalue)
    sg = args.sigma
    lc = args.lcut
    cw = args.cweight
    fw = args.fwidth
    rc = args.rcut
    sub = args.subset
    cen = args.center
    nlist = args.nlist

    # Read in features
    ffile = args.features
    ftrs=[line.rstrip('\n') for line in open(ffile)]
    npoints = int(sub*len(ftrs))

    # Optionally, read in unit cell
    if args.cell:
        # For a condensed-phase system, this is specified
        vcell=[line.rstrip('\n') for line in open(args.cell)]
    else:
        # If we are considering gas-phase systems, we don't need the unit cell
        vcell = []

    return [ftrs,vcell,npoints,lval,sg,lc,rc,cw,fw,args.verbose,args.periodic,cen,nlist]

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r",   "--rank",    type=int,   required=True,              help="Rank of tensor to learn")
    parser.add_argument("-lm",  "--lmda",    type=float, required=True,  nargs='+',  help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",  type=float, default=1.0,                 help="Fraction of data points used for testing")
    parser.add_argument("-t",   "--tensors", type=str,   required=True,              help="File containing tensors")
    parser.add_argument("-k",   "--kernel",  type=str,   required=True,  nargs='+',  help="Files containing kernels")
    parser.add_argument("-sel", "--select",  type=int,                   nargs='+',  help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",  type=int,   default=0,                  help="Number of random training points")
    parser.add_argument("-nc",  "--ncycles", type=int,   default=1,                  help="Number of cycles for regression with random selection")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):

    rank = int(args.rank)
    ftr = args.ftrain 
    # Get list of l values
    if (rank%2 == 0):
        # Even L
        lvals = [l for l in xrange(0,rank+1,2)]
    else:
        # Odd L
        lvals = [l for l in xrange(1,rank+1,2)]
    lm = args.lmda
    if (len(lm) != len(lvals)):
        print "Number of regularization parameters must equal number of L values!"
        sys.exit(0)
 
    # Read in features
    tfile = args.tensors
    tens=[line.rstrip('\n') for line in open(tfile)]

    kernels = args.kernel

    # If a selection is given for the training set, read it in
    if args.select:
        sel = args.select
        if (len(sel)!=2):
            print "Beginning and end of selection must be specified!"
            sys.exit(0)

    rdm = args.random
    ncycles = args.ncycles

    return [lvals,lm,ftr,tens,kernels,sel,rdm,rank,ncycles]
###############################################################################################################################
