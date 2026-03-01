#!/bin/bash

# 使用手动设置的测试池

# Run the training pipeline in debug mode with a timeout
echo "Starting training pipeline test (with 5 minute timeout)..."
timeout 30 /home/ubuntu/miniconda3/envs/js_gpu/bin/python -m src.training.train \
    --config configs/battle_train_config_test_version.yaml \
    --pool_path /home/ubuntu/Workfile/RL/RL_model/opponent_pool/test_pool \
    --debug

EXIT_CODE=$?

# Check if it completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training pipeline test completed successfully!"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "Training pipeline test FAILED: Timed out after 5 minutes!"
else
    echo "Training pipeline test failed with exit code $EXIT_CODE!"
fi
