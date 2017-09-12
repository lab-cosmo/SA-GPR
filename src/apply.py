#!usr/bin/python

import tkrr.scalars
import tkrr.dipoles
import tkrr.alphas
import tkrr.betas
import utils.kernels
import utils.parsing
import utils.inout
import argparse
import sys
from numpy import *

# This is a wrapper that calls python scripts to apply the tensorial KRR framework.

# INPUT ARGUMENTS.
args = utils.parsing.add_command_line_arguments_apply("Application of TKRR",True)
[invec,ftrs,vcell,nt,ns,r,sg,lm,compare,ofile,lc,rcut,cweight,fwidth,vrb] = utils.parsing.set_variable_values_apply(args,True)

print "\nUsing TKRR to predict properties of %i configurations. The training set has a length of %i.\n"%(ns,nt)

# Build kernels using our chosen definition.
#rmatr1=[]
#rmatr2=[]
#[k2tr,k2te] = utils.kernels.build_kernels(ktype,ftrs,nt,ns,sg,rmatr1,rmatr2,r,(compare != ""))

if (r==0):
    # Do scalars
    # Also read in the mean value.
    mean_scalar = float([line.rstrip('\n') for line in open(args.vector)][-1])
    [[k0tr,k0te]] = utils.kernels.build_kernels([0],ftrs,vcell,nt,ns,sg,lc,rcut,cweight,fwidth,vrb)
    out_tensor = tkrr.scalars.predict_scalars(compare,lm,nt,ns,invec,k0te,mean_scalar)
elif (r==1):
    # Do dipoles
    [[k1tr,k1te]] = utils.kernels.build_kernels([1],ftrs,vcell,nt,ns,sg,lc,rcut,cweight,fwidth,vrb)
    out_tensor = tkrr.dipoles.predict_dipoles(compare,lm,nt,ns,invec,k1te)
elif (r==2):
    # Do polarizabilities
    [[k0tr,k0te],[k2tr,k2te]] = utils.kernels.build_kernels([0,2],ftrs,vcell,nt,ns,sg,lc,rcut,cweight,fwidth,vrb)
    out_tensor = tkrr.alphas.predict_alphas(compare,lm,nt,ns,invec,k0te,k2te)
#    out_tensor = tkrr.alphas.predict_alphas(compare,lm,nt,ns,invec,k2te)
elif (r==3):
    # Do hyperpolarizabilities
    [[k1tr,k1te],[k3tr,k3te]] = utils.kernels.build_kernels([1,3],ftrs,vcell,nt,ns,sg,lc,rcut,cweight,fwidth,vrb)
    out_tensor = tkrr.betas.predict_betas(compare,lm,nt,ns,invec,k1te,k3te)
else:
    print "Rank specified is not yet covered by this code!"
    sys.exit(0)

if (ofile != ""):
    print "Outputting results to %s\n"%ofile
    utils.inout.output_predictions(ofile,out_tensor)
