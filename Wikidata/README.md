# simple-wikidata-db

This library provides a set of scripts to download the Wikidata dump, sort it into staging files, and query the data in these staged files in a distributed manner. The staging is optimized for (1) querying time, and (2) simplicity.

This library is helpful if you'd like to issue queries like:

- Fetch all QIDs which are related to [Q38257](https://www.wikidata.org/wiki/Q38257)
- Fetch all triples corresponding to the relation (e.g. [P35](https://www.wikidata.org/wiki/Property:P35))
- Fetch all aliases for a QID

## Environment Setup

### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux
- **Hardware Requirements**:
  - **Storage**: At least 500GB free space for Wikidata dump and processed data
  - **Memory**: 500GB+ RAM recommended for processing (scales with Wikidata dump)
  - **CPU**: Multi-core processor
  - **GPU**: Optional, for vector index building (Step 3)

### Installing Dependencies

**Install project dependencies** from the root directory:

```bash
cd CoGOnGraph
pip install -r requirements.txt
```

This will install all necessary packages including:
- `torch`, `transformers`, `sentence-transformers` - For embedding models
- `redis` - For in-memory database


## Downloading the dump

A full list of available dumps is available [here](https://dumps.wikimedia.org/wikidatawiki/entities/). To fetch the most recent dump, run:

```
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
```

or, if aria2c is installed, run:

```
aria2c --max-connection-per-server 16 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
```

Downloading takes about 2-5 hours (depending on bandwidth).

## Processing the Dump

The original downloaded wikidata dump is a single file and combines different types of information (alias names, properties, relations, etc). We preprocess the dump by iterating over the compressed file, and saving information to different subdirectories. For more information, see the [Data Format](#data-format). To preprocess the dump, run:
Use `simple_wikidata_db/preprocess_dump.py` to preprocess the dump:

```bash
python -m simple_wikidata_db.preprocess_dump \
    --input_file $PATH_TO_COMPRESSED_WIKI_JSON \
    --out_dir $OUTPUT_DIR  \
    --batch_size $BATCH_SIZE
```

**Parameters:**
- `input_file`: Path to the compressed Wikidata JSON dump file (e.g., `latest-all.json.gz`)
- `out_dir`: Output directory where extracted tables will be saved. Subdirectories will be created for each table (e.g., `labels/`, `aliases/`, `entity_rels/`, etc.)
- `language_id`: Language code for extracting entity labels, aliases, descriptions, and Wikipedia links (default: `en`)
- `processes`: Number of concurrent worker processes (default: 90). Adjust based on available CPU cores and memory.
- `batch_size`: Number of records to write per batch file within each table directory (default: 10000)


**Example:**
```bash
python -m simple_wikidata_db.preprocess_dump \
    --input_file latest-all.json.gz \
    --out_dir KB/ \
    --batch_size 10000
```

It takes ~5 hours to process the dump when running with 90 processes on a 1024GB machine with 56 cores. A tqdm progress bar should provide a more accurate estimate while data is being processed.  

## Data Format

The Wikidata dump is made available as a single, unweildy JSON file. To make querying/filtering easier, we split the information contained in this JSON file into multiple **tables**, where each table contains a certain type of information. The tables we create are described below:

| Table name    | Table description   | Table schema|
| --------------- |:--------------------| :-----|
| labels          | Holds the labels for different entities | qid: the QID of the entity <br> label: the entity's label ('name') |
| descriptions    | Holds the descriptions for different entities | qid: the QID of the entity <br> description: the entity's description (short summary at the top of the page) |
| aliases         | Holds the aliases for different entities  | qid: the QID of the entity <br> alias: an alias for the entity |
| entity_rels     | Holds statements where the value of the statement is another wikidata entity | claim_id: the ID for the statement <br> qid: the ID for wikidata entity <br> property_id: the ID for the property <br> value: the qid for the value wikidata entity |
| external_ids    | Holds statements where the value of the statement is an identifier to an external database (e.g. Musicbrainz, Freebase, etc) | claim_id: the ID for the statement <br> qid: the ID for wikidata entity <br> property_id: the ID for the property <br> value: the identifier for the external ID |
| entity_values   | Holds statements where the value of the statement is a string/quantity | claim_id: the ID for the statement <br> qid: the ID for wikidata entity <br> property_id: the ID for the property <br> value: the value for this property |
| qualifiers      | Holds qualifiers for statements |  qualifier_id: the ID for the qualifier <br> claim_id: the ID for the claim being qualified <br> property_id: the ID for the property <br> value: the value of the qualifier |
| wikipedia_links | Holds links to Wikipedia items | qid: the QID of the entity <br> wiki_title: link to corresponding wikipedia entity  |
| plabels | Holds PIDs and their corresponding labels | pid: the PID of the property <br> label: the label for the property |
----

<br><br>
Each table is stored in a directory, where the content of the table is written to multiple jsonl files stored inside the directory (each file contains a subset of the rows in the table). Each line in the file corresponds to a different triple. Partitioning the table's contents into multiple files improves querying speed--we can process each file in parallel.

# Instructions for deploying a query service locally



---

The knowledge base deployment consists of four main steps: 1) building index files, 2) merging degree statistics, 3) building vector index, and 4) starting query servers.

## Step 1: Building Index Files

Use `simple_wikidata_db/db_deploy/build_index.py` to build dictionary indices for fast in-memory query:

```bash
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir $PREPROCESS_DATA_DIR \
    --output_dir $INDEX_FILE_DIR \
    --num_chunks $NUM_CHUNKS \
    --num_workers $NUM_WORKERS \
    --chunk_idx $CHUNK_IDX
```

**Parameters:**
- `input_dir`: The preprocessed wikidata dump directory. It should be the output directory of the preprocessing job described above (e.g., `KB/`).
- `output_dir`: The directory where the generated index files will be stored. Usually a subfolder of `input_dir`, e.g., `KB/indices`.
- `num_chunks`: The number of chunks to split the data into. This allows parallel querying across multiple server processes.
- `num_workers`: Number of subprocesses for this job (default: 400).
- `chunk_idx`: Which chunk to build. Use `-1` (default) to build all chunks sequentially, or specify a chunk index to build a specific chunk.
  - **Important**: `chunk_idx` ranges from **0 to (num_chunks - 1)**. For example, if `num_chunks=6`, valid `chunk_idx` values are 0, 1, 2, 3, 4, 5.
  - You must build all chunks (0 through num_chunks-1) before proceeding to the next steps.

**Example:**
```bash
# Option 1: Build all chunks sequentially (simpler but slower)
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir KB/ \
    --output_dir KB/indices \
    --num_chunks 6 \
    --num_workers 400 \
    --chunk_idx -1

# Option 2: Build chunks in parallel on different machines/terminals (faster)
# If num_chunks=6, you need to run 6 commands with chunk_idx from 0 to 5:

# Terminal 1 or Machine 1
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir KB/ \
    --output_dir KB/indices \
    --num_chunks 6 \
    --chunk_idx 0

# Terminal 2 or Machine 2
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir KB/ \
    --output_dir KB/indices \
    --num_chunks 6 \
    --chunk_idx 1

# Terminal 3 or Machine 3
python -m simple_wikidata_db.db_deploy.build_index \
    --input_dir KB/ \
    --output_dir KB/indices \
    --num_chunks 6 \
    --chunk_idx 2

# ... Continue for chunk_idx 3, 4, 5
# All 6 chunks (0-5) must be completed before moving to Step 2
```

**Output Files:**
Each chunk produces the following pickle files in `output_dir`:
- `relation_entities_chunk_{i}.pickle`: Entity-relation mappings (head/tail)
- `tail_entities_chunk_{i}.pickle`: Entity pairs for given relation
- `tail_values_chunk_{i}.pickle`: Literal values for entity-relation pairs
- `external_ids_chunk_{i}.pickle`: External IDs for entities
- `mid_to_qid_chunk_{i}.pickle`: External ID to QID mappings

Additionally, temporary degree statistics are saved in `output_dir/../tmp/`:
- `label_degree_chunk_{i}.pickle`: In/out/attribute degrees for entities
- `p_degree_chunk_{i}.pickle`: Occurrence counts for relations

**Resource Requirements:**
- Memory: ~200GB per 1/10 chunk
- Time: ~30 minutes per 1/10 chunk with 400 workers

## Step 2: Merging Degree Statistics

After building all index chunks, use `simple_wikidata_db/db_deploy/merge_degrees.py` to merge the degree statistics from all chunks:

```bash
python -m simple_wikidata_db.db_deploy.merge_degrees \
    --input_dir $TEMP_DIR \
    --output_dir $DATA_DIR \
    --batch_size $BATCH_SIZE
```

**Parameters:**
- `input_dir`: Directory containing the temporary degree files, usually `KB/tmp/`
- `output_dir`: Directory to store merged degree files, usually the same as the main data directory `KB/`
- `batch_size`: Number of records per output JSONL file (default: 20000)

**Example:**
```bash
python -m simple_wikidata_db.db_deploy.merge_degrees \
    --input_dir KB/tmp \
    --output_dir KB
```

**Output Files:**
- `KB/label_degree/*.jsonl`: Merged entity degree statistics (in_degree, out_degree, attr_degree)
- `KB/p_degree/*.jsonl`: Merged relation degree statistics

## Step 3: Building Vector Index

The vector index is used for fuzzy entity recall in the entity linking process when exact label and alias matching fail. Build it using `simple_wikidata_db/db_deploy/build_vector_index.py`.

Environment Setup Recommendation:
If you want to use faiss gpu version, **it is highly recommended to create a separate Python environment for this step** because FAISS GPU version may have compatibility issues with other packages.

**Create a separate conda environment for FAISS GPU:**

```bash
# Create a new conda environment with Python 3.11
conda create -n faiss python=3.11 -y

# Activate the environment
conda activate faiss

# Install FAISS GPU version
conda install pytorch::faiss-gpu -y

# Install project dependencies
cd CoGOnGraph
pip install -r requirements.txt
```

After building the vector index, you can switch back to your main environment for running the query servers.

### Building the Index

```bash
python -m simple_wikidata_db.db_deploy.build_vector_index \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --device $GPU_DEVICES \
    --encode_batch_size $BATCH_SIZE \
    --faiss_use_gpu \
    --cached_sorted_data_path $CACHE_PATH
```

**Parameters:**
- `data_dir`: The main data directory containing preprocessed data (e.g., `KB/`)
- `model_name`: Sentence transformer model name (e.g., `BAAI/bge-large-en-v1.5`, `Qwen/Qwen3-Embedding-4B`)
- `device`: GPU device IDs, comma-separated (e.g., `0,1,2,3`)
- `encode_batch_size`: Batch size for encoding vectors (adjust based on model and GPU memory)
- `faiss_use_gpu`: Enable GPU for FAISS indexing operations
- `cached_sorted_data_path`: Path to cache sorted data for faster rebuilding (e.g. `KB/tmp/sorted_data.pkl`, optional but recommended)
- `training_samples`: Number of samples for training the index (default: 6,000,000)
- `add_batch_size`: Batch size for adding vectors to index (default: 1,000,000)
- `resume`: If resume from a partially built index

**Examples:**
```bash
# Using BAAI/bge-large-en-v1.5 model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m simple_wikidata_db.db_deploy.build_vector_index \
    --data_dir KB \
    --device 0,1,2,3 \
    --model_name BAAI/bge-large-en-v1.5 \
    --encode_batch_size 800 \
    --faiss_use_gpu

# Using Qwen embedding model with memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m simple_wikidata_db.db_deploy.build_vector_index \
    --data_dir KB \
    --device 0,1,2,3 \
    --model_name Qwen/Qwen3-Embedding-4B \
    --encode_batch_size 100 \
    --cached_sorted_data_path KB/tmp/sorted_data.pkl \
    --faiss_use_gpu
```

**Output Files:**
- `KB/indices/vector_index_{model_name}.faiss`: FAISS vector index
- `KB/indices/vector_index_map_{model_name}.pkl`: Mapping from index positions to entity QIDs

## Step 4: Starting Query Servers

### Prerequisites: Redis Setup

Before starting the query servers, you need to install and start Redis, which is used as an in-memory database for caching and serving the knowledge base data.

**Install Redis:**
```bash
# Ubuntu/Debian (recommended)
sudo snap install redis

# Or using apt
sudo apt update && sudo apt install redis-server
```

**Start Redis:**
```bash
# If installed via snap
sudo snap start redis

# If installed via apt
sudo systemctl start redis-server
```


### Starting Server Processes

Use `simple_wikidata_db/db_deploy/server.py` to start server processes. Each server loads one chunk of data:

```bash
python -m simple_wikidata_db.db_deploy.server \
    --data_dir $DATA_DIR \
    --chunk_number $CHUNK_NUMBER \
    --port $PORT \
    --host_ip $HOST_IP \
    --redis_db $REDIS_DB \
    --flush_redis
```

**Parameters:**
- `data_dir`: The main data directory containing both raw data and indices (e.g., `KB/`)
- `chunk_number`: Chunk index to serve (0 to num_chunks-1)
- `port`: Port number for this server to listen on (e.g., 23150, 23151, ...)
- `host_ip`: Host IP address for the server
- `redis_host`: Redis server host (default: `localhost`)
- `redis_port`: Redis server port (default: 6379)
- `redis_db`: Redis database number (use different DBs for different chunks, e.g., 0, 1, 2, ...)
- `flush_redis`: Flush Redis DB before loading data (use for first-time loading or data updates)
- `num_chunks`: Total number of chunks (default: 6). Must match the value used in Step 1.
- `vector_search_device`: GPU device for vector search (e.g., `cuda:0`). **Recommended: use default `None` for CPU inference** to avoid occupying GPU resources during server runtime.
- `vector_search_model`: Model name for vector search (default: `BAAI/bge-m3`). **Must match the model used in Step 3** to load the correct vector index files.

**Important Notes:**
- The first server (chunk_number=0) loads global data (labels, property labels, relation degrees)
- The last server (chunk_number=num_chunks-1) loads the vector index for semantic search functionality
- Each server process should use a different server port (via `--port`) and a different Redis database number (via `--redis_db`)
- Use `--flush_redis` on first run or when data is updated
- Make sure the vector index files exist before starting the last server, otherwise semantic search will fail
- **For `vector_search_model`**: Ensure it matches the model name used in Step 3. The server will look for `vector_index_{model_name}.faiss` in `KB/indices/`.
- **For `vector_search_device`**: CPU inference (default `None`) is recommended for production to save GPU resources. GPU acceleration is optional and only provides marginal speed improvements for individual queries.

**Examples:**
```bash
# Start the first server (chunk 0) - loads global labels and properties
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 0 \
    --port 23150 \
    --host_ip 127.0.0.1 \
    --redis_db 0 \
    --flush_redis > logs/server_log_0.log 2>&1 &

# Start additional servers in background
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 1 \
    --port 23151 \
    --host_ip 127.0.0.1 \
    --redis_db 1 \
    --flush_redis > logs/server_log_1.log 2>&1 &

python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 2 \
    --port 23152 \
    --host_ip 127.0.0.1 \
    --redis_db 2 \
    --flush_redis > logs/server_log_2.log 2>&1 &

# ... start more servers for remaining chunks

# Start the last server (chunk 5 if num_chunks=6) - loads vector index
python -m simple_wikidata_db.db_deploy.server \
    --data_dir KB/ \
    --chunk_number 5 \
    --port 23155 \
    --host_ip 127.0.0.1 \
    --redis_db 5 \
    --flush_redis \
    --vector_search_model BAAI/bge-m3 > logs/server_log_5.log 2>&1 &

```

**Resource Requirements:**
- **Memory**: For the April 2025 Wikidata dump with `num_chunks=6`, the total memory requirement across all servers is approximately **360GB** (~60GB per server chunk).
- **Loading Time**: First-time data loading takes approximately **45 minutes** for each server processes.

Note: These requirements scale with the size of the Wikidata dump and the number of chunks. Fewer chunks means larger chunk size and more memory per server.

**Service Architecture:**
- Uses XML-RPC protocol for client-server communication
- Each server listens on its specified port
- Clients query all servers in parallel and aggregate results locally

## Querying the Database

### Testing the Deployment

To verify that all servers are running correctly, use the provided test script in `db_deploy/client.py`:

```bash
python -m simple_wikidata_db.db_deploy.client --addr_list server_urls.txt
```

**Server URL File Format (`server_urls.txt`):**
```
http://127.0.0.1:23150
http://127.0.0.1:23151
http://127.0.0.1:23152
http://127.0.0.1:23153
http://127.0.0.1:23154
http://127.0.0.1:23155
```

**Available Query Methods:**
- `label2qid(label)`: Convert entity label to QID(s)
- `qid2label(qid)`: Convert QID to entity label
- `label2pid(label)`: Convert relation label to PID(s)
- `pid2label(pid)`: Convert PID to relation label
- `get_all_relations_of_an_entity(qid)`: Get all relations of an entity
- `get_tail_entities_given_head_and_relation(head_qid, relation_pid)`: Get tail entities
- `get_tail_values_given_head_and_relation(head_qid, relation_pid)`: Get literal values
- `qid2aliases(qid)`: Get entity aliases
- `qid2description(qid)`: Get entity description
- `get_label_degree(qid)`: Get entity degree statistics
- `get_p_degree(pid)`: Get relation degree statistics
- `find_similar_entities(name, k, nprobe)`: Semantic search for similar entities (requires vector index)

**Example Usage in Python:**
```python
from simple_wikidata_db.db_deploy.client import MultiServerWikidataQueryClient

# Initialize client with server URLs
server_urls = ["http://127.0.0.1:23150", "http://127.0.0.1:23151", ...]
client = MultiServerWikidataQueryClient(server_urls)

# Query examples
qids = client.query_all("label2qid", "Douglas Adams")
relations = client.query_all("get_all_relations_of_an_entity", "Q42")
similar = client.query_all("find_similar_entities", "famous writer", 5, 128)
```

For a single query, the client automatically sends the query to all relevant server nodes, retrieves results, and aggregates them locally.
