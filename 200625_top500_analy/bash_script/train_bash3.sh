#!/bin/bash

declare -a gene_list=("26" "45" "61")

n_gene=${#gene_list[@]}

for (( idx1=0; idx1<1; idx1++))    ##model
do
for (( idx0=0; idx0<${n_gene}; idx0++))   ##gene
do
for (( idx3=0; idx3<5; idx3++))    ##k_num
do

	python ../main_code/main3.py ${gene_list[$idx0]} ${idx1} 26 ${idx3}
    
done
done
done
