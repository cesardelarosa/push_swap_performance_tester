#!/bin/bash

#make all bonus

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 number_of_elements"
	exit 1
fi

N=$1
ARG=$(shuf -i 1-1000 -n "$N" | tr '\n' ' ')
echo "Random numbers: $ARG"
./push_swap $ARG > temp_moves.txt
MOVES=$(wc -l < temp_moves.txt)
echo "Number of moves: $MOVES"
RESULT=$(./checker $ARG < temp_moves.txt)
echo "Checker result: $RESULT"
rm temp_moves.txt
