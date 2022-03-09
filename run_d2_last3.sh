#!/bin/bash

#names=("Bovine.txt" "Circuit.txt" "Ecoli.txt" "USAir97.txt" "humanDiseasome.txt" "Treni_Roma.txt" "EU_flights.txt" "openflights.txt" "yeast1.txt" "powergrid.txt" "OClinks.txt" "facebook.txt" "grqc.txt" "hepth.txt" "hepph.txt" "astroph.txt" "condmat.txt")
#numbers=(15 30 20 70 10 10 850 140 6 20 1100 450 20 70 3600 12000 500)

names=("internet.txt" "email-Enron.txt" "twitter_combined_dat.txt" "com-amazon.ungraph_dat.txt" "com-youtube.ungraph_dat.txt")
numbers=(11481 18346 40653 167431 567445)

mkdir -p output
#rm -v output/Design2/d2_result_summary_.out
for i in "${!numbers[@]}"; do
    echo -n ${names[i]}
    python3 $1 -l 2 -L ${numbers[i]} -n 30 -b 0.4 ../data/${names[i]} >> output/Design2/d2_result_summary_last3.out
    echo
done
