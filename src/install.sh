#!/bin/bash
f2py=~/local/bin/f2py
cd utils
 $f2py -c --opt='-O3' fill_power_spectra.f90 -m pow_spec
cd ../
