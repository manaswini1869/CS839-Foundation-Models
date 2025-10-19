#!/bin/bash

# An array of character counts
SIZES=(10000 50000 100000 500000 1000000)

echo "Starting training for all subsets..."

for size in "${SIZES[@]}"; do
    dir_name=$(($size/1000))k
    DATASET_PATH="mark_twain_char/$dir_name"
    OUT_DIR="out-twain-$dir_name"

    echo "--- Training model for $dir_name ---"

    # Adjust training parameters as needed. 
    # For smaller datasets, you may want fewer iterations (max_iters).
    #
    #
   

    python train.py \
    --dataset="$DATASET_PATH" \
    --out_dir="$OUT_DIR" \
    --device=cpu \
    --compile=False \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --block_size=64 \
    --batch_size=12 \
    --max_iters=500 \
    --eval_interval=100
done

echo "All training jobs complete."
