#!/bin/bash

for (( idx1=0; idx1<6; idx1++))
do
for (( idx0=0; idx0<10; idx0++))
do
for (( idx3=0; idx3<5; idx3++))
do

	python ../main_code/main2_weight.py ${idx0} ${idx1} 26 ${idx3}
    
done
done
done
