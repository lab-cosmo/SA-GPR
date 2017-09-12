#!/usr/bin/python

def output_file(ofile,nt,ns,r,infile,tfile,sg,lm,intrins_error,invktrvecs):
    fl = open(ofile,'w')
    fl.write("TKRR with %i total configurations, of which %i used for training and %i for testing.\n\n"%(nt+ns,nt,ns))
    fl.write("Learning kernels of rank %i from file %s. Features in file %s\n"%(r,infile,tfile))
    fl.write("Sigma = %f, lambda = %.12f %.12f %.12f %.12f\n\n"%(sg,lm[0],lm[1],lm[2],lm[3]))
    fl.write("Intrinsic error = %.3f %%\n\n"%(intrins_error))
    fl.write("Kernel vectors:\n")
    fl.write("==============\n\n")
    for j in range(len(invktrvecs)):
        invktrvec = invktrvecs[j]
        fl.write("\n")
        for i in range(len(invktrvec)):
            fl.write("%f\n"%invktrvec[i])
    fl.close()

def output_predictions(ofile,ov):
    fl = open(ofile,'w')
    for i in range(len(ov)):
        fl.write("%s\n"%ov[i])
    fl.close()
