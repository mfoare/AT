#!/bin/bash

LANG=C


if test $# -ne 13; then 
	echo "./at-inp-color.sh  exec 	output-rep  output-file  image 	mask  alpha-1  alpha-2  alpha-ratio  lambda-1  lambda-2  lambda-ratio  eps-1  eps-2"
	echo "ex : ./at-inp-color.sh  at-u0-v1-inpainting-couleur  ./inp/u0-v1/  t  ./Images/triple-color.ppm './mask/mask-disque-t.pgm ./mask/mask-scribble-t.pgm ./mask/mask-text.pgm'   1.0  0.1  1.1  0.1  0.0005  1.1  1.0  0.25"
else

EXEC=$1
OUTPUT_REP=$2
OUTPUT_FILE=$3
IM=$4
MASK=($5)

A1=$6
A2=$7
AR=$8

L1=$9
L2=${10}
LR=${11}

E1=${12}
E2=${13}


#(cd build && make $EXEC)

len=${#MASK[*]}
for I in $( seq 1 1 $(($len-1)) )
do
	M=${MASK[$I]}		
	O=$OUTPUT_FILE'/'$OUTPUT_FILE
	mkdir -p $OUTPUT_REP'/'$OUTPUT_FILE

	for A in $( seq $A1 $AR $A2 )
	do
		./build/$EXEC -i $IM -m $M -o $OUTPUT_REP'/'$O  --alpha $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2 
	done
			
done


fi
