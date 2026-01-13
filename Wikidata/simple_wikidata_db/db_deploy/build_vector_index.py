import argparse
import os
import sys
from tqdm import tqdm
import itertools
import random
from collections import defaultdict
import pickle

# Add the parent directory to the Python path to allow importing 'vector_search'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_deploy.vector_search import VectorSearch
from db_deploy.utils import get_batch_files, jsonl_generator

def read_aliases_for_indexing(filename):
    texts = []
    qids = []
    for item in jsonl_generator(filename):
        texts.append(item["alias"])
        qids.append(item["qid"])
    return texts, qids

def read_labels_for_indexing(filename):
    texts = []
    qids = []
    for item in jsonl_generator(filename):
        texts.append(item["label"])
        qids.append(item["qid"])
    return texts, qids

def get_dynamic_encode_batch_size(max_len: int) -> int:
    if max_len > 1000:  return 100
    elif max_len > 100: return 200
    elif max_len > 50:  return 300
    else:   return 400

def run_memory_test(vector_search, unique_texts, add_batch_size, test_batches, test_sample_size):
    """
    Runs a memory test before training to ensure encoding doesn't cause OOM errors.
    Returns True if the tests pass, False otherwise.
    """
    print("--- Starting Memory Test Phase ---")
    
    for batch_idx in test_batches:
        # Assuming batch_idx is 1-based from user input.
        start_idx = (batch_idx - 1) * add_batch_size
        end_idx = start_idx + test_sample_size
        test_texts = unique_texts[start_idx:end_idx]

        if not test_texts:
            print(f"Batch {batch_idx}: Skipped (no data found for testing).")
            continue
            
        max_len = len(test_texts[0])
        dynamic_encode_batch_size = get_dynamic_encode_batch_size(max_len)
        
        print(f"Testing Batch {batch_idx}: {len(test_texts)} texts, max_len={max_len}, dynamic_batch_size={dynamic_encode_batch_size}")
        
        try:
            test_vectors = vector_search._encode(test_texts, dynamic_encode_batch_size, show_progress_bar=False)
            print(f"Batch {batch_idx}: Memory test passed ✓")
            del test_vectors
        except Exception as e:
            print(f"Batch {batch_idx}: Memory test failed ✗ - {e}")
            print("Consider adjusting batch sizes in get_dynamic_encode_batch_size() or reducing GPU load.")
            return False
            
    print("--- All memory tests passed. Proceeding to training. ---")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory (KB)")
    parser.add_argument("--model_name", type=str, default='BAAI/bge-m3', help="Sentence transformer model name")
    parser.add_argument("--device", type=str, default=None, help="GPU device IDs to use, comma-separated (e.g., '0,1,2'), not used for CPU encoding")
    parser.add_argument("--faiss_index_type", type=str, default='IVFPQ', help="Faiss index type (e.g., IVFPQ, IVFFlat)")
    parser.add_argument("--faiss_nlist", type=int, default=16384, help="Number of cells for IVF-based indexes")
    parser.add_argument("--faiss_m", type=int, default=32, help="Number of sub-quantizers for PQ")
    parser.add_argument("--faiss_nbits", type=int, default=8, help="Number of bits per sub-code for PQ")
    parser.add_argument("--encode_batch_size", type=int, default=1024, help="Batch size for encoding vectors")
    parser.add_argument("--training_samples", type=int, default=6_000_000, help="Number of unique texts to use for training the index")
    parser.add_argument("--add_batch_size", type=int, default=1_000_000, help="Number of unique texts to add to the index in one batch")
    parser.add_argument("--faiss_use_gpu", action="store_true", help="Use GPU for Faiss indexing operations if --device is also set.")
    parser.add_argument("--cached_sorted_data_path", type=str, default=None, help="Path to save/load cached sorted unique texts and QIDs.")
    parser.add_argument("--memory_test_batches", type=str, default="", help="Comma-separated list of 1-based batch indices to use for memory pre-testing.")
    parser.add_argument("--memory_test_samples", type=int, default=40000, help="Number of samples per batch for memory pre-testing.")
    parser.add_argument("--resume", action="store_true", help="Resume from a partially built index.")
    args = parser.parse_args()

    # --- Data Loading or Preparation ---
    if args.cached_sorted_data_path and os.path.exists(args.cached_sorted_data_path):
        print(f"Loading cached sorted data from {args.cached_sorted_data_path}...")
        with open(args.cached_sorted_data_path, 'rb') as f:
            unique_texts, qids_for_texts = pickle.load(f)
        total_unique_texts = len(unique_texts)
        print(f"Loaded {total_unique_texts} unique texts from cache.")
    else:
        if not args.data_dir or not os.path.isdir(args.data_dir):
            print(f"Error: Data directory not found at {args.data_dir}. It's required when not loading from a cache.")
            return

        # --- Data Reading and Deduplication ---
        print("Reading all files and deduplicating texts...")
        text_to_qids = defaultdict(set)
        alias_files = get_batch_files(os.path.join(args.data_dir, "aliases"))
        label_files = get_batch_files(os.path.join(args.data_dir, "labels"))
        all_files = alias_files + label_files
        
        for f in tqdm(all_files, desc="Reading and deduplicating"):
            if "aliases" in f:
                texts, qids = read_aliases_for_indexing(f)
            else:
                texts, qids = read_labels_for_indexing(f)
            for text, qid in zip(texts, qids):
                text_to_qids[text].add(qid)
        
        unique_texts = list(text_to_qids.keys())
        qids_for_texts = [list(qids) for qids in text_to_qids.values()]
        del text_to_qids # Free up memory
        
        total_unique_texts = len(unique_texts)
        print(f"Found {total_unique_texts} unique texts.")

        if not unique_texts:
            print("No unique texts found. Aborting.")
            return

        # Sort by text length to optimize encoding performance
        print("Sorting unique texts by length...")
        combined = sorted(zip(unique_texts, qids_for_texts), key=lambda x: len(x[0]), reverse=True)
        unique_texts, qids_for_texts = [list(t) for t in zip(*combined)]
        print("Sorting complete.")

        # --- Caching Sorted Data ---
        if args.cached_sorted_data_path:
            # Ensure the directory for the cache file exists
            cache_dir = os.path.dirname(args.cached_sorted_data_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            print(f"Saving sorted data to {args.cached_sorted_data_path}...")
            with open(args.cached_sorted_data_path, 'wb') as f:
                pickle.dump((unique_texts, qids_for_texts), f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved sorted data.")

    output_dir = os.path.join(args.data_dir, "indices") if args.data_dir else "indices"
    os.makedirs(output_dir, exist_ok=True)

    model_name_for_path = args.model_name.split("/")[-1]
    index_path = os.path.join(output_dir, f"vector_index_{model_name_for_path}.faiss")
    mapping_path = os.path.join(output_dir, f"vector_index_map_{model_name_for_path}.pkl")

    devices = [f'cuda:{gpu_id}' for gpu_id in args.device.split(',')] if args.device else None
    vector_search = VectorSearch(
        model_name=args.model_name, 
        device=devices,
        faiss_use_gpu=args.faiss_use_gpu
    )

    start_index = 0
    if args.resume and os.path.exists(index_path) and os.path.exists(mapping_path):
        print("Resuming from existing index...")
        vector_search.load_index(index_path, mapping_path)
        if vector_search.index:
            start_index = vector_search.index.ntotal
            print(f"Resuming from index with {start_index} items.")
        else:
            print("Failed to load index, starting from scratch.")

    # --- Memory Test Phase ---
    if args.memory_test_batches:
        test_batches = [int(b) for b in args.memory_test_batches.split(',')]
    else:
        test_batches = []
    
    try:
        vector_search.start_pool()
        if not run_memory_test(vector_search, unique_texts, args.add_batch_size, test_batches, args.memory_test_samples):
            print("Memory test failed. Aborting.")
            return
        
        # --- Training and Adding Phases ---
        if start_index == 0:
            # --- Training Phase (only if not resuming from a trained index) ---
            print(f"Starting training phase with up to {args.training_samples} samples.")
            if args.training_samples >= total_unique_texts:
                training_texts = unique_texts
            else:
                # Sample randomly to get a representative set for training
                training_indices = random.sample(range(total_unique_texts), args.training_samples)
                training_texts = [unique_texts[i] for i in training_indices]

            d = vector_search.model.get_sentence_embedding_dimension()
            vector_search.initialize_index(d, args.faiss_index_type, args.faiss_nlist, args.faiss_m, args.faiss_nbits)
            vector_search.train_index(training_texts, args.encode_batch_size)
            del training_texts # Free up memory
            print("Index training complete. Saving index...")
            vector_search.save_index(index_path, mapping_path)
            print("Index saved after training.")

        # --- Adding Phase ---
        print(f"Starting adding phase to build the full index...")
        
        # Adjust tqdm initial value to reflect progress when resuming
        initial_batch = start_index // args.add_batch_size
        
        for i in tqdm(range(start_index, total_unique_texts, args.add_batch_size), 
                      desc="Adding batches to index", 
                      initial=initial_batch, 
                      total=total_unique_texts // args.add_batch_size):
            batch_texts = unique_texts[i:i + args.add_batch_size]
            batch_qids = qids_for_texts[i:i + args.add_batch_size]
            
            if batch_texts:
                # Since texts are sorted by length in descending order, the first text is the longest.
                max_len = len(batch_texts[0])
                dynamic_encode_batch_size = get_dynamic_encode_batch_size(max_len)
                print(f"\nAdding batch from index {i} with {len(batch_texts)} texts, max_len={max_len}, dynamic_batch_size={dynamic_encode_batch_size}")
                vector_search.add_to_index(batch_texts, batch_qids, dynamic_encode_batch_size)
                
                print(f"Batch from index {i} added. Saving index...")
                vector_search.save_index(index_path, mapping_path)
                print(f"Index saved. Total indexed items: {vector_search.index.ntotal}")

        print(f"Finished adding all data. Total indexed items: {vector_search.index.ntotal}")

    finally:
        vector_search.stop_pool()
        print("Process pool stopped.")

    print("Unified index building complete.")

if __name__ == "__main__":
    main() 