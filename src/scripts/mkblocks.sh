#!/bin/bash

file=$1
cell=$2
nstruct=$(wc -l $1)
blocksize=$3
((nblocks=nstruct/blocksize))

for i in $(seq 1 $nblocks); do
   for j in $(seq 1 $i); do
        mkdir -p Block_${i}-${j}
      	awk -v i=$i -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $file > Block_${i}-${j}/coords.in
        awk -v i=$j -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $file >> Block_${i}-${j}/coords.in
        awk -v i=$i -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $cell > Block_${i}-${j}/cell.in
        awk -v i=$j -v bs=$blocksize 'FNR>(bs*(i-1)) && FNR<=(bs*i) {print}' $cell >> Block_${i}-${j}/cell.in 
   done
done
