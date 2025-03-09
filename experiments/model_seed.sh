#!/bin/bash

# Ensure the 'runs' directory exists
mkdir -p runs/model_seed

# Define the range of seeds
seq 4551 4580 | xargs -P 20 -I {} bash -c 'python adapt_drones/train.py --env_id traj_v3 --seed {} > runs/model_seed/eval_seed_{}.txt 2>&1'
