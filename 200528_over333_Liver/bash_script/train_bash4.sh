#!/bin/bash

for (( idx1=0; idx1<3; idx1++))    ##model
do
for (( idx0=0; idx0<100; idx0++))   ##gene
do
for (( idx3=0; idx3<5; idx3++))    ##k_num
do

	python ../main_code/main4.py ${idx0} ${idx1} 26 ${idx3}
    
done
done
done
