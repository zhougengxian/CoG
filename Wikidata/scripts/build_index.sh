#!/bin/bash

# Build index chunks in parallel
# Adjust the range {0..5} based on your num_chunks setting
# For num_chunks=6, valid chunk_idx values are 0, 1, 2, 3, 4, 5

# Create logs directory if it doesn't exist
mkdir -p logs

# Start parallel index building processes
for i in {0..5}; do
    echo "Starting build_index for chunk $i..."
    python -m simple_wikidata_db.db_deploy.build_index \
        --input_dir KB/ \
        --num_chunks 6 \
        --chunk_idx $i \
        --output_dir KB/indices \
        --num_workers 400 > logs/build_index_${i}.log 2>&1 &
done

echo "All index building processes started. Check logs/ for progress."
wait
echo "All index building processes completed."
