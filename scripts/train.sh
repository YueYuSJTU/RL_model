#!/bin/bash
python src/train.py \
    --config configs/agent/ppo.yaml \
    --env_config configs/env/c172_trajectory.yaml \
    --exp_name ppo_trajectory \
    --seed 42