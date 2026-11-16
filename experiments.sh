#!/bin/bash
# Usage: bash experiments.sh DATA_DIR   (all sweeps, run in parallel; results in results/; defaults: lr 0.03, E 10, 500 rounds)
export OMP_NUM_THREADS=1   # many runs in parallel on a shared server: one thread each
D=${1:-data}
for C in 0.02 0.05 0.1; do for E in 1 5 10; do python run.py --data $D --C $C --E $E & done; done   # Part 1: C and E sweep
python run.py --data $D --parallel --name part1_ray &                                              # Part 1.2: 4 clients/round via Ray
wait
for b in 0 0.05 0.1 0.2 0.5 1.0 2.0; do python run.py --data $D --b $b --name part2_b$b & done      # Part 2: noise sweep (C=0.1)
wait
