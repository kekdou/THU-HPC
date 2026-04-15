#!/bin/bash

num_elements=$2

if [ "$num_elements" -le 100 ]; then
    N=1
    n=1
elif [ "$num_elements" -le 1000 ]; then
    N=1
    n=2
elif [ "$num_elements" -le 10000 ]; then
    N=1
    n=11
elif [ "$num_elements" -le 100000 ]; then
    N=2
    n=54
elif [ "$num_elements" -le 1000000 ]; then
    N=2
    n=44
elif [ "$num_elements" -le 10000000 ]; then
    N=2
    n=54
else
    N=2
    n=32
fi

# run on 1 machine * 28 process, feel free to change it!
srun -N $N -n $n --cpu-bind=cores --exclusive $*
