#!/bin/bash

# Start all server processes for knowledge base query service
# Adjust num_chunks and the loop range based on your setup

pids=()
# Set a trap on EXIT to kill the child processes
trap 'echo "Killing child processes: ${pids[*]}"; kill ${pids[@]} &>/dev/null' EXIT

# Configuration
DATA_DIR="KB/"
HOST_IP="127.0.0.1"
NUM_CHUNKS=6
VECTOR_SEARCH_MODEL="BAAI/bge-m3"  # Must match the model used in Step 3

# Create logs directory if it doesn't exist
mkdir -p logs

# Create server_urls.txt for client
rm -f server_urls.txt

echo "Starting $NUM_CHUNKS server processes..."

# Start servers for chunks 0 to NUM_CHUNKS-1
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    PORT=$((23150 + i))
    
    # Add special parameters for the last server (loads vector index)
    if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
        echo "Starting server for chunk $i (with vector search) on port $PORT..."
        python -m simple_wikidata_db.db_deploy.server \
            --data_dir "$DATA_DIR" \
            --chunk_number $i \
            --port $PORT \
            --host_ip "$HOST_IP" \
            --redis_db $i \
            --num_chunks $NUM_CHUNKS \
            --flush_redis \
            --vector_search_model "$VECTOR_SEARCH_MODEL" > logs/server_log_$i.log 2>&1 &
    else
        echo "Starting server for chunk $i on port $PORT..."
        python -m simple_wikidata_db.db_deploy.server \
            --data_dir "$DATA_DIR" \
            --chunk_number $i \
            --port $PORT \
            --host_ip "$HOST_IP" \
            --redis_db $i \
            --num_chunks $NUM_CHUNKS \
            --flush_redis > logs/server_log_$i.log 2>&1 &
    fi
    
    pids+=($!)
    
    # Add server URL to file for client
    echo "http://$HOST_IP:$PORT" >> server_urls.txt
    
    # Brief delay to avoid overwhelming the system
    sleep 2
done

echo "Started $NUM_CHUNKS server processes with PIDs: ${pids[*]}"
echo "Server URLs saved to server_urls.txt"
echo "Monitor logs with: tail -f logs/server_log_*.log"

wait
