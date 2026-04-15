#!bin/bash

LOG_FILE="output.txt"
DATA_SIZE=100000000
DATA_FILE="data/${DATA_SIZE}.dat"

MIN_N=2
MAX_N=2

MIN_n=28
MAX_n=56

STEP_n=4

> "${LOG_FILE}"

echo "========== start testing =========="

for (( N=${MIN_N}; N<=${MAX_N}; N+=1 )); do
    for (( n=${MIN_n}; n<=${MAX_n}; n+=${STEP_n} )); do
	CMD="srun -N $N -n $n --cpu-bind=cores --exclusive odd_even_sort $DATA_SIZE $DATA_FILE"
	echo "running: N=$N, n=$n ..."
	echo "@@@CMD: ${CMD}" >> "${LOG_FILE}"
	$CMD >> "${LOG_FILE}" 2>&1
	echo "-------------------------------------" >> "${LOG_FILE}"
    done
done

