#!/bin/bash

cd utils
 f2py -c --opt='-O3' sph_in.f -m spherical_in
cd ../
