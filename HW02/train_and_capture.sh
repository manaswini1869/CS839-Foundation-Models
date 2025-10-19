#!/bin/bash

# --- CONFIGURATION ---
MODEL_DIR="out-twain-small-captured"
TOTAL_ITERS=1000
INTERVAL=100
# --- END CONFIGURATION ---

echo "Starting training run, capturing checkpoints to $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# This variable will be empty on the first run and '--init_from=resume' after.
resume_flag=""

# Loop from the interval up to the total iterations
for (( i=$INTERVAL; i<=$TOTAL_ITERS; i+=$INTERVAL )); do
    echo
    echo "--- Training up to iteration $i ---"
    
    # Run the training command. The $resume_flag is empty on the first pass.
    python train.py config/train_mark_twain_char.py \
        --out_dir="$MODEL_DIR" \
        --device=cpu \
        --compile=False \
        --n_layer=4 \
        --n_head=4 \
        --n_embd=128 \
        --max_iters="$i" \
        --block_size=64 \
        --batch_size=12 \
        --dropout=0.0 \
        --eval_interval="$INTERVAL" \
        --always_save_checkpoint=True \
        $resume_flag
        
    # After the first run, all subsequent runs should resume.
    resume_flag="--init_from=resume"
        
    # Copy the resulting checkpoint to a versioned file
    cp "$MODEL_DIR/ckpt.pt" "$MODEL_DIR/ckpt_$i.pt"
    echo "--- Captured checkpoint for iteration $i ---"
done

echo "✅ Training and capturing complete."
