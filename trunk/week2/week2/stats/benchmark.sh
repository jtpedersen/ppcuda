#!/bin/bash


function run_benchmark {
# goes in 256 steps
    START=$((2**$1))
    STOP=$((2**$2))
    PROGRAM=$3

    echo "#STATS"
    echo "#SIZE simple tiled Textured"

    while [ $START -le $STOP ]
    do
	SIZE=$START
	SIMPLE=$(./$PROGRAM s $SIZE S 1 | head -n 4 | tail -n 1)
	TILED=$(./$PROGRAM t $SIZE S 1 | head -n 4 | tail -n 1)
	TEXTURED=$(./$PROGRAM T $SIZE S 1 | head -n 4 | tail -n 1)
	echo "$SIZE $SIMPLE $TILED $TEXTURED"
	START=$((START+256))
    done
}

GPU=$1

run_benchmark 8 10 32_week2 > 32_${GPU}.dat
run_benchmark 8 10 16_week2 > 16_${GPU}.dat
run_benchmark 8 10 8_week2 > 8_${GPU}.dat
run_benchmark 8 10 4_week2 > 4_${GPU}.dat
run_benchmark 8 10 2_week2 > 2_${GPU}.dat



