#!/usr/bin/python
import argparse
import sys
import numpy as np

def add_command_line_arguments_tenskernel(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-lval", "--lvalue", help="Order of the spherical tensor")
    parser.add_argument("-f", "--features",help="File containing atomic fingerprints (coordinates)")
    parser.add_argument("-sg", "--sigma", type=float, help="Sigma for SOAP kernels")
    parser.add_argument("-lc", "--lcut", type=float, help="lcut for SOAP kernels")
    parser.add_argument("-c", "--cell",help="File containing cell vectors")
    parser.add_argument("-rc", "--rcut", type=float,help="Cutoff value for bulk systems as a fraction of the box length")
    parser.add_argument("-cw", "--cweight", type=float,help="Central atom weight")
    parser.add_argument("-fw", "--fwidth", type=float,help="Width of the radial filtering to atomic densities")
    parser.add_argument("-vr", "--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("-per","--periodic",type=bool, help="Call for periodic systems")
    parser.add_argument("-sub", "--subset", type=float, help="Fraction of the input data set")
    parser.add_argument("-cen", "--center", nargs='+', help="List of atoms to center on (default all)")
    parser.add_argument("-nc",  "--ncycles", type=int, help="Number of cycles for regression with random selection")
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

    if args.ncycles:
        ncycles = args.ncycles
    else:
        ncycles = 1

    print rc
    return [ftrs,vcell,npoints,lval,sg,lc,rc,cw,fw,args.verbose,per,cen,ncycles]
   
