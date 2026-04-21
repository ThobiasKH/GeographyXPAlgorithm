# GeographyXPAlgorithm

Prototype Python implementations accompanying the paper on **Edge Geography parameterized by width**.

This repository contains code for the XP algorithm on rooted tree partitions described in Section 5 of the paper, together with an experimental harness used to validate the implementation against brute-force minimax on generated instances.

## Contents

- `ueg_tree_partition_xp_only.py`  
  Minimal implementation of the XP algorithm only.

- `ueg_tree_partition_experiment.py`  
  Experimental harness containing:
  - random instance generation together with rooted tree partitions,
  - brute-force minimax solver,
  - XP solver,
  - verification and benchmarking utilities,
  - optional runtime plots.

## What this repository is for

This code is intended as a **correctness and reproducibility artifact**, not as a practical solver.

The XP algorithm is mathematically interesting, but the hidden dependence on the width parameter is extremely large, so even moderately sized instances can become impractical.

## Requirements

- Python 3.10 or newer
- `matplotlib` only if you want to generate plots

Optional install for plotting:

```bash
pip install matplotlib

## Run the experimental harness on a random instance 

```bash
python ueg_tree_partition_experiment.py demo --n 10 --width 3 --seed 1

## Verify the XP solver agains minimax 

```bash
python ueg_tree_partition_experiment.py verify --trials 20 --n 10 --width 3 --seed 2

## Benchmark both solvers 

```bash
python ueg_tree_partition_experiment.py benchmark --sizes 6,7,8,9,10 --width 3 --trials-per-size 5 --seed 3 --progress

## Generate surface plots over $(n,k)$ 

```bash
python ueg_tree_partition_experiment.py surface \
    --sizes 6,8,10 \
    --widths 2,3,4 \
    --trials-per-point 3 \
    --solver both \
    --plot-prefix surface \
    --log-z

This may produce files such as: 
- surface_tree_partition.png 
- surface_brute_force.png 

## Using the minimal XP solver 

```bash 
from ueg_tree_partition_xp_only import solve_ueg_with_partition

edges = [(0, 1), (1, 2), (1, 3)]
bags = [[0], [1], [2], [3]]
parent = [None, 0, 1, 1]
start = 0

winner_is_p1 = solve_ueg_with_partition(
    n=4,
    edges=edges,
    bags=bags,
    parent=parent,
    start=start,
)

print("P1 wins?" , winner_is_p1)
