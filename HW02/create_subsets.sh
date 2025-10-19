#!/bin/bash

# The main directory for your dataset
DATA_DIR="data/mark_twain_char"

# An array of character counts
SIZES=(10000 50000 100000 500000 1000000)

echo "Creating subsets for Mark Twain dataset..."

for size in "${SIZES[@]}"; do
    # Format size for directory name (e.g., 10k, 500k)
    dir_name=$(($size/1000))k
    SUBSET_DIR="$DATA_DIR/$dir_name"

    echo "Creating subset: $dir_name"

    # Create the subdirectory
    mkdir -p "$SUBSET_DIR"

    # Create a smaller input.txt
    head -c "$size" "$DATA_DIR/input.txt" > "$SUBSET_DIR/input.txt"

    # Copy and run the prepare script for the subset
    cp "$DATA_DIR/prepare.py" "$SUBSET_DIR/"
    (cd "$SUBSET_DIR" && python prepare.py)
done

echo "Done creating subsets."
