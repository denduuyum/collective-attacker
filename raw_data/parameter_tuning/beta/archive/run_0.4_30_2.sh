#!/bin/bash

names=("humanDiseasome.txt" "facebook.txt" "condmat.txt")
numbers=(10 450 500)

mkdir -p output
rm -v output/run.out
for i in "${!numbers[@]}"; do
    echo -n ${names[i]}
    python3 $1 -l 2 -L ${numbers[i]} -n 30 -b 0.4 ../data/${names[i]} >> output/parameter_tuning/beta/result_summary_0.4_30_2.out
    echo
done
