#!/bin/bash

#names=("Bovine.txt" "Circuit.txt" "Ecoli.txt" "USAir97.txt" "humanDiseasome.txt" "Treni_Roma.txt" "EU_flights.txt" "openflights.txt" "yeast1.txt" "powergrid.txt" "OClinks.txt" "facebook.txt" "grqc.txt" "hepth.txt" "hepph.txt" "astroph.txt" "condmat.txt")
#numbers=(15 30 20 70 10 10 850 140 6 20 1100 450 20 70 3600 12000 500)

names=("humanDiseasome.txt" "EU_flights.txt" "OClinks.txt" "facebook.txt" "hepph.txt")
numbers=(10 850 1100 450 3600)

mkdir -p output
rm -v output/parameter_tuning/beta/beta_result_summary_1.0_30_2.out
for i in "${!numbers[@]}"; do
    echo -n ${names[i]}
    python3 $1 -l 2 -L ${numbers[i]} -n 30 -b 1.0 ../data/${names[i]} >> output/parameter_tuning/beta/beta_result_summary_1.0_30_2_8parallel.out
    echo
done
