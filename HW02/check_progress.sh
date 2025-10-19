#!/bin/bash

# --- CONFIGURATION ---
MODEL_DIR="out-twain-small-captured"
# --- END CONFIGURATION ---

# Check if the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory '$MODEL_DIR' not found."
    exit 1
fi

echo "### Analyzing model progress in: $MODEL_DIR ###"
echo

# Find all intermediate checkpoints and sort them numerically
CHECKPOINTS=$(ls -v "$MODEL_DIR"/ckpt_*.pt 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No intermediate checkpoints (e.g., ckpt_100.pt) found in $MODEL_DIR."
    exit 1
fi

# Loop through all intermediate checkpoints
for ckpt_path in $CHECKPOINTS; do
    iterations=$(echo "$ckpt_path" | grep -o '[0-9]*')
    dest_path="$MODEL_DIR/ckpt.pt"
    
    echo "--- Sample after $iterations training iterations ---"
    
    # Verify the source file exists before copying
    if [ ! -f "$ckpt_path" ]; then
        echo "DEBUG ERROR: Source file $ckpt_path not found!"
        continue
    fi

    # Temporarily COPY the intermediate checkpoint
    cp "$ckpt_path" "$dest_path"
    
    # Verify that the copy was successful
    if [ ! -f "$dest_path" ]; then
        echo "DEBUG ERROR: Failed to create destination file $dest_path!"
        continue
    fi
    
    # Generate a sample from the current checkpoint
    python sample.py --out_dir="$MODEL_DIR" --max_new_tokens=300 --device=cpu
    
    echo "----------------------------------------------------"
    echo
done

# Clean up the temporary checkpoint file
if [ -f "$MODEL_DIR/ckpt.pt" ]; then
    rm "$MODEL_DIR/ckpt.pt"
fi

echo "✅ Analysis complete."
