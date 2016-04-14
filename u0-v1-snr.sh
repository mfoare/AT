!/bin/bash

LANG=C

PROG="u0-v1"
FILE='res/'$PROG
EXEC='at-'$PROG

(cd build && make $EXEC)


## carre et losange

#E1=1.0
#E2=0.25

#L1=0.05
#L2=0.001
#LR=1.2

#T=64

#BRUIT=("0" "2" "4" "8")
#ALPHA_B00="0.2 		0.5		0.75"
#ALPHA_B02="0.05 	0.1 	0.25"
#ALPHA_B04="0.025 	0.05	0.12"
#ALPHA_B08="0.012	0.25	0.06"
#ALPHA_B=("$ALPHA_B00" "$ALPHA_B02" "$ALPHA_B04" "$ALPHA_B08")

#for I in $(seq 0 1 3)
#do
#	B=${BRUIT[$I]}
#	ALPHA=${ALPHA_B[$I]}
#	for A in $ALPHA
#	do
#			DIR='./'$FILE'/cb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'carre2Degrades64b0'$B'.pgm' 						-o $DIR/'cb0'$B'_a'$A 	-t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#			
#			DIR='./'$FILE'/lb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'losange2DegradesDroite64b0'$B'.pgm' 		-o $DIR/'lb0'$B'_a'$A 	-t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#			
#			DIR='./'$FILE'/l30b0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'losange2Degrades30Droite64b0'$B'.pgm' 	-o $DIR/'l30b0'$B'_a'$A -t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#			
#			DIR='./'$FILE'/cbb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'cercle-barre-b'$B'.pgm' 	-o $DIR/'cbb0'$B'_a'$A -t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#	done
#done


## lena

#E1=1.0
#E2=0.25

#L1=0.05
#L2=0.001
#LR=1.2

#T=128

#BRUIT=("2")
#ALPHA_B01="0.1 		0.2 	0.5"
#ALPHA_B02="0.4"	#"0.05 	0.1 	0.25"
#ALPHA_B=("$ALPHA_B02")

#for I in $(seq 0 1 3)
#do
#	B=${BRUIT[$I]}
#	ALPHA=${ALPHA_B[$I]}
#	for A in $ALPHA
#	do
#			DIR='./'$FILE'/nb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'lena128b0'$B'.pgm' -o $DIR/'nb0'$B'_a'$A -t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#	done
#done


## peppers

#E1=1.0
#E2=0.25

#L1=0.05
#L2=0.001
#LR=1.2

#T=256

#BRUIT=("05" "2" "4")
#ALPHA_B005="0.2 		0.4 	"
#ALPHA_B02="	0.05 		0.1 	0.25"
#ALPHA_B04="	0.025 	0.05	0.12"
#ALPHA_B=("$ALPHA_B005" "$ALPHA_B02" "$ALPHA_B04")

#for I in $(seq 0 1 3)
#do
#	B=${BRUIT[$I]}
#	ALPHA=${ALPHA_B[$I]}
#	for A in $ALPHA
#	do
#			DIR='./'$FILE'/pb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'peppers-256-b0'$B'.pgm' -o $DIR/'pb0'$B'_a'$A -t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#	done
#done



# triple-spirale

E1=1.0
E2=0.25

L1=0.05
L2=0.0005
LR=1.2

T=64

BRUIT=("1")
ALPHA_B01="0.3" #"0.1 		0.2		0.5 "
ALPHA_B02="0.15 0.2 0.3 0.4"	#"0.05 	0.1 	0.25"
ALPHA_B04="0.025 	0.05	0.12"
ALPHA_B08="0.012	0.25	0.06"
ALPHA_B=("$ALPHA_B01")

for I in $(seq 0 1 0)
do
	B=${BRUIT[$I]}
	ALPHA=${ALPHA_B[$I]}
	for A in $ALPHA
	do
			DIR='./'$FILE'/tsb0'$B'_a'$A
			mkdir -p $DIR
			./build/$EXEC -i ./Images/'triple-spirale-b0'$B'.pgm' 						-o $DIR/'tsb0'$B'_a'$A 	-t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
	done
done


## barbara

#E1=1.0
#E2=0.25

#L1=0.05
#L2=0.001
#LR=1.2

#T=348

#BRUIT=("2")
#ALPHA_B01="0.1 		0.2 	0.5"
#ALPHA_B02="0.25"	#"0.05 	0.1 	0.25"
#ALPHA_B=("$ALPHA_B02")

#for I in $(seq 0 1 0)
#do
#	B=${BRUIT[$I]}
#	ALPHA=${ALPHA_B[$I]}
#	for A in $ALPHA
#	do
#			DIR='./'$FILE'/bb0'$B'_a'$A
#			mkdir -p $DIR
#			./build/$EXEC -i ./Images/'barbara-cropped-b0'$B'.pgm' -o $DIR/'bb0'$B'_a'$A -t $T -a $A --lambda-1 $L1 --lambda-2 $L2 --lambda-ratio $LR --epsilon-1 $E1 --epsilon-2 $E2
#	done
#done


