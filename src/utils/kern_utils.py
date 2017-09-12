#!/usr/bin/python

import numpy as np
from random import shuffle

###############################################################################################################################

def shuffle_data(ndata,sel,rdm,fractrain):

    if rdm == 0:
        trrangemax = np.asarray(range(sel[0],sel[1]),int)
    else:
        data_list = range(ndata)
        shuffle(data_list)
        trrangemax = np.asarray(data_list[:rdm],int).copy()
    terange = np.setdiff1d(range(ndata),trrangemax)

    ns = len(terange)
    ntmax = len(trrangemax)
    nt = int(fractrain*ntmax)
    trrange = trrangemax[0:nt]

    return [ns,nt,ntmax,trrange,terange]

###############################################################################################################################
