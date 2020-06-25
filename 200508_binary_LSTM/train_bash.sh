#!/bin/bash

for (( idx0=2; idx0<4; idx0++))
do
for (( idx1=0; idx1<3; idx1++))
do
for (( idx3=0; idx3<5; idx3++))
do
	python main2.py ${idx0} ${idx1} 0 ${idx3}
done
done
done