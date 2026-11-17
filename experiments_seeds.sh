#!/bin/bash
# Usage: bash experiments_seeds.sh DATA_DIR  -- seeds 1 and 2 of the noise sweep (b=0 is the best Part 1 config); run after experiments.sh
export OMP_NUM_THREADS=1
D=${1:-data}
for s in 1 2; do for b in 0 0.05 0.1 0.2 0.5 1.0 2.0; do python run.py --data $D --b $b --seed $s --name part2_b${b}_s$s & done; done
wait
