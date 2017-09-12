#!/usr/bin/python

import numpy as np

dipoles = np.loadtxt("dipole_1000.in",dtype=float)

avmol = 0.0
for i in range(1000):
    avdip = np.linalg.norm(dipoles[i])
    avmol += avdip
avmol /=32000
print avmol

