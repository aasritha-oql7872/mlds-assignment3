#!/bin/bash
# Usage: bash experiments.sh DATA_DIR   (runs all sweeps; results in results/)
D=${1:-data}
for C in 0.02 0.05 0.1; do for E in 1 5; do python run.py --data $D --C $C --E $E; done; done   # Part 1 sweep
python run.py --data $D --parallel --name part1_ray                                              # Part 1.2 (4 clients/round via Ray)
for b in 0 0.05 0.1 0.2 0.5 1.0; do python run.py --data $D --b $b --name part2_b$b; done        # Part 2 sweep
