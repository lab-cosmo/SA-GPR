#!/usr/bin/python
from __future__ import print_function
from builtins import range
import argparse
import sys
import numpy as np
from ase.io import read

###############################################################################################################################

def add_command_line_arguments_tenskernel(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lval", "--lvalue",    type=int,   required=True,                   help="Order of the spherical tensor")
    parser.add_argument("-f",    "--features",  type=str,   required=True,                   help="File containing atomic coordinates and properties")
    parser.add_argument("-sg",   "--sigma",     type=float, default=0.3,                     help="Sigma for SOAP kernels")
    parser.add_argument("-lc",   "--lcut",      type=int,   default=6,                       help="lcut for SOAP kernels")
    parser.add_argument("-rc",   "--rcut",      type=float, default=3.0,                     help="Cutoff value for bulk systems as a fraction of the box length")
    parser.add_argument("-cw",   "--cweight",   type=float, default=1.0,                     help="Central atom weight")
    parser.add_argument("-vr",   "--verbose",               action='store_true',             help="Verbose mode")
    parser.add_argument("-sub",  "--subset",    type=float, default=1.0,                     help="Fraction of the input data set")
    parser.add_argument("-cen",  "--center",    type=str,   default='',          nargs='+',  help="List of atoms to center on (default all)")
    parser.add_argument("-n",    "--nlist",     type=int,   default=[0],         nargs='+',  help="List of n values for kernel calculation")
    parser.add_argument("-atom", "--atomic",                action='store_true',             help="Call for kernels of atomic environments")
    parser.add_argument("-ex",   "--extrap",                action='store_true',             help="Call for kernels to be used in extrapolation")
    parser.add_argument("-nt",   "--ntest",     type=int,   default=1,                       help="Number of points for extrapolation")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_tenskernel(args):

    # Use command-line arguments to set the values of important variables
    lval = int(args.lvalue)
    sg = args.sigma
    lc = args.lcut
    cw = args.cweight
    rc = args.rcut
    sub = args.subset
    cen = args.center
    nlist = args.nlist
    ntest = args.ntest

    # Read in features
    ffile = args.features
    ftrs = read(ffile,':')
    npoints = int(sub*len(ftrs))

    return [ftrs,npoints,lval,sg,lc,rc,cw,args.verbose,cen,nlist,args.atomic,args.extrap,ntest]

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-r",   "--rank",     type=int,   required=True,              help="Rank of tensor to learn")
    parser.add_argument("-lm",  "--lmda",     type=float, required=True,  nargs='+',  help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",   type=float, default=1.0,                help="Fraction of data points used for testing")
    parser.add_argument("-f",   "--features", type=str,   required=True,              help="File containing atomic coordinates")
    parser.add_argument("-p",   "--property", type=str,   required=True,              help="Property to be learned")
    parser.add_argument("-k",   "--kernel",   type=str,   required=True,  nargs='+',  help="Files containing kernels")
    parser.add_argument("-sel", "--select",   type=int,   default=[],     nargs='+',  help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",   type=int,   default=0,                  help="Number of random training points")
    parser.add_argument("-nc",  "--ncycles",  type=int,   default=1,                  help="Number of cycles for regression with random selection")
    parser.add_argument("-perat","--peratom",             action='store_true',        help="Call for scaling the properties by the number of atoms")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):

    rank = int(args.rank)
    ftr = args.ftrain
    # Get list of l values
    if (rank%2 == 0):
        # Even L
        lvals = [l for l in range(0,rank+1,2)]
    else:
        # Odd L
        lvals = [l for l in range(1,rank+1,2)]
    lm = args.lmda
    if (len(lm) != len(lvals)):
        print("Number of regularization parameters must equal number of L values!")
        sys.exit(0)

    # Read in features
    ftrs = read(args.features,':')

    nat = []
    [nat.append(ftrs[i].get_number_of_atoms()) for i in range(len(ftrs))]
    if args.peratom:
        if rank == 0:
            tens = [str(ftrs[i].info[args.property]/nat[i]) for i in range(len(ftrs))]
        elif rank == 2:
            tens = [' '.join((np.concatenate(ftrs[i].info[args.property])/nat[i]).astype(str)) for i in range(len(ftrs))]
        else:
            tens = [' '.join((np.array(ftrs[i].info[args.property])/nat[i]).astype(str)) for i in range(len(ftrs))]
    else:
        if rank == 0:
               tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
        elif rank == 2:
            tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
        else:
            tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]

    kernels = args.kernel

    # If a selection is given for the training set, read it in
    sel = args.select
    if (len(sel)!=2 & len(sel)!=0):
        print("Beginning and end of selection must be specified!")
        sys.exit(0)

    rdm = args.random
    ncycles = args.ncycles

    return [lvals,lm,ftr,tens,kernels,sel,rdm,rank,ncycles,nat,args.peratom]
#########################################################################
