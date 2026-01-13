import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import typing as tp


class VectorSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None, faiss_use_gpu=True):
        # device can be a string (e.g., 'cuda:0') or a list of strings (e.g., ['cuda:0', 'cuda:1'])
        self.devices = device if isinstance(device, list) else ([device] if device else None)
        
        # Don't specify a device for the main model if we're using multiple GPUs for encoding,
        # as the model will be distributed to worker processes.
        main_process_device = None
        if self.devices:
            if len(self.devices) > 1:
                # If using multiple GPUs, load the main model on CPU to avoid duplicate model on the first GPU.
                main_process_device = 'cpu'
            else:
                # If using a single GPU, specify it for the main model.
                main_process_device = self.devices[0]
        else:
            # If no device is specified, default to CPU.
            main_process_device = 'cpu'
        
        self.model = SentenceTransformer(model_name, device=main_process_device)
        self.index = None
        self.is_gpu_index = False
        self.idx_to_qids = []
        self.idx_to_text = []
        self.faiss_use_gpu = faiss_use_gpu
        self.pool = None

    def start_pool(self):
        if self.devices and len(self.devices) > 1:
            self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def stop_pool(self):
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None

    def _move_to_gpu(self, force_single_gpu=False):
        if not self.faiss_use_gpu or not self.devices or self.is_gpu_index or not self.index:
            return
        
        gpu_ids = [int(d.split(':')[-1]) for d in self.devices]

        # If only one GPU is available, or if single GPU is forced
        if len(gpu_ids) == 1 or force_single_gpu:
            target_gpu = gpu_ids[0] if len(gpu_ids) == 1 else gpu_ids[1]
            print(f"Moving index to single GPU device: cuda:{target_gpu}")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, target_gpu, self.index)
        else:
            print(f"Moving index to all GPUs available")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Recommended to use sharding mode for training and adding vectors

            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        self.is_gpu_index = True

    def _move_to_cpu(self):
        if not self.is_gpu_index or not self.index:
            return
        print("Moving index back to CPU...")
        self.index = faiss.index_gpu_to_cpu(self.index)
        self.is_gpu_index = False

    def _encode(self, texts: tp.List[str], encode_batch_size: int, show_progress_bar: bool = True):
        if self.pool:
            # A smaller value like 20000 can reduce memory footprint on each worker, but it will also increase process overhead.
            chunk_size = min(len(texts) // (len(self.devices) * 4), 40000)
            if chunk_size == 0 and len(texts) > 0:
                chunk_size = len(texts)

            vectors = self.model.encode_multi_process(
                texts,
                pool=self.pool,
                batch_size=encode_batch_size,
                chunk_size=chunk_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar
            )
        else:
            vectors = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
                batch_size=encode_batch_size
            )
        return vectors.astype('float32')

    def initialize_index(self, d: int, index_type='IVFPQ', nlist=16384, m=16, nbits=8):
        quantizer = faiss.IndexFlatIP(d)
        
        if index_type == 'IVFFlat':
            print(f"Using IVFFlat index with nlist={nlist}")
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        elif index_type == 'IVFPQ':
            print(f"Using IVFPQ index with nlist={nlist}, m={m}, nbits={nbits}")
            self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")
        print("Index initialized.")

    def train_index(self, sample_texts: tp.List[str], encode_batch_size: int):
        if self.index is None:
            raise RuntimeError("Index must be initialized before training.")
        
        print(f"Encoding {len(sample_texts)} sample texts for training...")
        training_vectors = self._encode(sample_texts, encode_batch_size, show_progress_bar=True)
        # training_vectors = np.random.rand(1_000_000, 768).astype('float32')
        
        self._move_to_gpu(force_single_gpu=True)
        
        print("Training the index...")
        self.index.train(training_vectors)
        print("Training complete.")

        self._move_to_cpu()

    def add_to_index(self, texts: tp.List[str], qids: tp.List[tp.List[str]], encode_batch_size: int):
        if self.index is None or not self.index.is_trained:
            raise RuntimeError("Index must be initialized and trained before adding vectors.")
        
        # Ensure index is on CPU before encoding to save VRAM
        self._move_to_cpu()
        
        print(f"Encoding {len(texts)} texts to add to index...")
        vectors = self._encode(texts, encode_batch_size, show_progress_bar=False) # Progress bar disabled for batch adding

        # Move index to GPU for fast adding
        self._move_to_gpu()

        print(f"Adding {len(vectors)} vectors to the index...")
        self.index.add(vectors)
        self.idx_to_qids.extend(qids)
        self.idx_to_text.extend(texts)
        print(f"Adding vectors complete. Total indexed items: {self.index.ntotal}")

        # Move index back to CPU to free VRAM for the next encoding batch
        self._move_to_cpu()

    def build_index(self, texts: tp.List[str], qids: tp.List[tp.List[str]], index_type='IVFPQ', nlist=16384, m=16, nbits=8, encode_batch_size=256):
        print("Encoding texts to vectors...")
        vectors = self._encode(texts, encode_batch_size, show_progress_bar=True)
        print(f"Encoding complete. Vector dimension: {vectors.shape[1]}")
        
        d = vectors.shape[1]
        self.initialize_index(d, index_type, nlist, m, nbits)

        # --- Training on single GPU ---
        print("Training the index on a single GPU...")
        self._move_to_gpu(force_single_gpu=True)
        self.index.train(vectors)
        print("Training complete.")

        # Move back to CPU before potentially moving to all GPUs for adding
        self._move_to_cpu()

        # --- Adding on all available GPUs ---
        print("Adding vectors to the index using all available GPUs...")
        self._move_to_gpu()
        self.index.add(vectors)
        print("Adding vectors complete.")
        self.idx_to_qids = qids
        self.idx_to_text = texts

        self._move_to_cpu()

    def search(self, query: str, k: int = 5, nprobe: int = 64) -> tp.List[tp.Dict[str, tp.Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # For inference, we move the index to GPU and keep it there for performance
        self._move_to_gpu()

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            print(f"Using nprobe={nprobe} for search.")

        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                dist = distances[0][i]
                if 0 <= idx < len(self.idx_to_text):
                    text = self.idx_to_text[idx]
                    qids = self.idx_to_qids[idx]
                    results.append({
                        "text": text,
                        "score": float(dist),
                        "qids": qids
                    })
        return results

    def save_index(self, index_path: str, mapping_path: str):
        print(f"Saving index to {index_path}")
        
        # Ensure index is on CPU before saving
        self._move_to_cpu()
        
        faiss.write_index(self.index, index_path)
        print(f"Saving mapping to {mapping_path}")
        with open(mapping_path, 'wb') as f:
            pickle.dump((self.idx_to_text, self.idx_to_qids), f)

    def load_index(self, index_path: str, mapping_path: str):
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            print(f"Index files not found, skipping vector search functionality: {index_path}, {mapping_path}")
            self.index = None
            self.idx_to_qids = []
            self.idx_to_text = []
            return

        print(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.is_gpu_index = False # Loaded index is always a CPU index
        print(f"Loading mapping from {mapping_path}")
        with open(mapping_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                self.idx_to_text, self.idx_to_qids = data
            else:
                self.idx_to_qids = data
                self.idx_to_text = []
                print("Warning: Loaded old index mapping format. Text will not be available in search results.") 