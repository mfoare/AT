#!/bin/bash

LANG=C


if test $# -ne 13; then 
	echo "./supp-mat.sh  exec  output-rep  output-file  image  bruits  alpha-1  alpha-2  alpha-ratio  lambda-1  lambda-2  lambda-ratio  eps-1  eps-2"
	echo "ex : ./supp-mat.sh  at-u0-v1  ./res/u0-v1/  cb  ./Images/carre2Degrades64  '0 2 4 8'  1.0  0.001  1.1  0.1  0.0005  1.1  1.0  0.25"
else

EXEC=$1
OUTPUT_REP=$2
OUTPUT_FILE=$3
IM=$4'-b'
DATA=$4'.pgm'
BRUIT=($5)

A1=$6
A2=$7
AR=$8

L1=$9
L2=${10}
LR=${11}

E1=${12}
E2=${13}


#(cd build && make $EXEC)

len=${#BRUIT[*]}
for I in $( seq 1 1 $(($len-1)) )
do
	B=${BRUIT[$I]}		
	O=$OUTPUT_FILE$B'/'$OUTPUT_FILE$B
	mkdir -p $OUTPUT_REP'/'$OUTPUT_FILE$B

	for A in $( seq $A1 $AR $A2 )
	do
		./build/$EXEC -i $IM$B'.pgm' -o $OUTPUT_REP'/'$O  --alpha $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2 --snr --image-snr $DATA
	done
			
done


fi
