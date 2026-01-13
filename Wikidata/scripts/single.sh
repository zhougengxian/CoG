#!/bin/bash
# Single command examples for knowledge base deployment steps
# This file contains example commands for each deployment step
# Execute these commands in the Wikidata directory


# Step 1: Build a specific index chunk (example: chunk 5 for 6 chunks)
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir KB/ \
    --num_chunks 6 \
    --chunk_idx 5 \
    --output_dir KB/indices 2>&1 | tee -i logs/build_index_5.log

# Step 2: Merge degree statistics (run after all chunks are built)
python -m simple_wikidata_db.db_deploy.merge_degrees \
    --input_dir KB/tmp \
    --output_dir KB \

# Step 3: Build vector index
Option 1: Using Qwen embedding model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m simple_wikidata_db.db_deploy.build_vector_index \
    --data_dir KB \
    --device 0,1,2,3 \
    --model_name Qwen/Qwen3-Embedding-4B \
    --encode_batch_size 100 \
    --cached_sorted_data_path KB/tmp/sorted_data.pkl \
    --faiss_use_gpu

# Option 2: Using BGE embedding model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m simple_wikidata_db.db_deploy.build_vector_index \
    --data_dir KB \
    --device 0,1,2,3 \
    --model_name BAAI/bge-m3 \
    --encode_batch_size 800 \
    --cached_sorted_data_path KB/tmp/sorted_data.pkl \
    --faiss_use_gpu

# Step 4: Start server processes
# First server (chunk 0) - loads global data
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 0 \
    --port 23150 \
    --host_ip 127.0.0.1 \
    --redis_db 0 \
    --flush_redis > logs/server_log_0.log 2>&1 &

# Second server (chunk 1)
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 1 \
    --port 23151 \
    --host_ip 127.0.0.1 \
    --redis_db 1 \
    --flush_redis > logs/server_log_1.log 2>&1 &

# ... Continue for chunk 2, 3, 4

# Last server (chunk 5) - loads vector index
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 5 \
    --port 23155 \
    --host_ip 127.0.0.1 \
    --redis_db 5 \
    --num_chunks 6 \
    --flush_redis \
    --vector_search_model BAAI/bge-m3 > logs/server_log_5.log 2>&1 &