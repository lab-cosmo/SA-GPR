#!/bin/bash

file=$1
nstruct=$(grep -c 'Lattice' $1)
blocksize=$2
((nblocks=nstruct/blocksize))
blocksize=$(echo $blocksize $(head -n 1 $1) | awk '{print $1*($2+2)}')

for i in $(seq 1 $nblocks); do
   for j in $(seq 1 $i); do
        mkdir -p Block_${i}-${j}
      	awk -v i=$i -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $file > Block_${i}-${j}/coords.xyz
        awk -v i=$j -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $file >> Block_${i}-${j}/coords.xyz
   done
done
