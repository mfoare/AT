!/bin/bash

LANG=C

PROG="u0-v1"
FILE='snr/'$PROG
EXEC='at-'$PROG'-snr'

(cd build && make $EXEC)

DATA='triple-spirale'
IM='triple-spirale-b01'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/tsb1'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='triple-spirale-b02'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/tsb2'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='triple-spirale-b04'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/tsb4'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='triple-spirale-b08'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/tsb8'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25


DATA='barbara-cropped'
IM='barbara-cropped-b01'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/bb1'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='barbara-cropped-b02'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.pgm -d ./Images/$DATA.pgm -o $REP'/bb2'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25


DATA='mandrill-240'
IM='mandrill-240-b01'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/mb1'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='mandrill-240-b02'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/mb2'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='mandrill-240-b04'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/mb4'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25



DATA='lena-370'
IM='lena-370-b01'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/lb1'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='lena-370-b02'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/lb2'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25
IM='lena-370-b04'
REP=$FILE'/'$IM
mkdir $REP
./build/$EXEC -i ./Images/$IM.ppm -d ./Images/$DATA.ppm -o $REP'/lb4'  --alpha-1 1 --alpha-2 0.01 --alpha-ratio 1.2 --lambda-1 0.01 --lambda-2 0.0001 --lambda-ratio 1.2 --epsilon-1 1 --epsilon-2 0.25

