import os
import pickle
import time
import typing as tp
from multiprocessing import Pool
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer
import redis
from collections import defaultdict
from simple_wikidata_db.db_deploy.utils import (
    jsonl_generator,
    get_batch_files,
    read_entity_label,
    read_relation_label,
)
import ujson as json
from tqdm import tqdm
import math
from simple_wikidata_db.db_deploy.vector_search import VectorSearch

def read_aliases(filename):
    qid_to_aliases = defaultdict(list)
    alias_to_qids = defaultdict(list)
    for item in jsonl_generator(filename):
        qid = item["qid"]
        alias = item["alias"]
        qid_to_aliases[qid].append(alias)
        alias_to_qids[alias].append(qid)
    return qid_to_aliases, alias_to_qids

def read_descriptions(filename):
    qid_to_desc = {}
    for item in jsonl_generator(filename):
        qid_to_desc[item["qid"]] = item["description"]
    return qid_to_desc

def read_p_degrees(filename):
    pid_to_degree = {}
    for item in jsonl_generator(filename):
        pid_to_degree[item["pid"]] = item["degree"]
    return pid_to_degree

def read_label_degrees(filename):
    qid_to_degrees = {}
    for item in jsonl_generator(filename):
        qid_to_degrees[item["qid"]] = {
            "in": item["in_degree"],
            "out": item["out_degree"], 
            "attr": item["attr_degree"]
        }
    return qid_to_degrees

class WikidataQueryServer:
    def __init__(
        self,
        chunk_number: int,
        data_dir: str,
        redis_host: str,
        redis_port: int,
        redis_db: int,
        flush_redis: bool,
        num_workers: int = 400,
        num_chunks: int = 5,
        vector_search_device: tp.Optional[str] = None,
        vector_search_model: str = 'BAAI/bge-m3'
    ):
        self.num_workers = num_workers
        self.pool = Pool(processes=self.num_workers)
        self.redis_conn = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            retry=redis.retry.Retry(redis.backoff.ExponentialBackoff(), 3),
        )
        chunk_number = chunk_number + 1

        # Load vector search index only on the first chunk server
        self.vector_search = None
        if chunk_number == 6:
            self.vector_search = VectorSearch(model_name=vector_search_model, device=vector_search_device)
            model_name_for_path = vector_search_model.split('/')[-1]
            index_path = os.path.join(data_dir, "indices", f"vector_index_{model_name_for_path}.faiss")
            mapping_path = os.path.join(data_dir, "indices", f"vector_index_map_{model_name_for_path}.pkl")
            self.vector_search.load_index(index_path, mapping_path)
        
        if flush_redis:
            print("Flushing Redis DB...")
            self.redis_conn.flushdb()

        while int(self.redis_conn.info('persistence').get('rdb_bgsave_in_progress', 0)):
            print("An existing RDB save is in progress, waiting for it to complete...")
            time.sleep(5)

        if self.redis_conn.exists(f"wikidata_data_loaded_chunk_{chunk_number}"):
            print(f"Data for chunk {chunk_number} already loaded in Redis.")
            return

        original_save_config = self.redis_conn.config_get("save")
        original_aof_rewrite_config = self.redis_conn.config_get("auto-aof-rewrite-percentage")
        try:
            print(
                "Disabling Redis background saving and increasing query buffer for performance..."
            )
            self.redis_conn.config_set("save", "")
            self.redis_conn.config_set("auto-aof-rewrite-percentage", 0)

            if chunk_number == 1:
                self.files_index = {
                    "labels": get_batch_files(os.path.join(data_dir, "labels")),
                    "plabels": get_batch_files(os.path.join(data_dir, "plabels")),
                    "p_degree": get_batch_files(os.path.join(data_dir, "p_degree")),
                }

                print("Loading labels data into Redis...")
                print("Reading and storing relation labels...")
                pipe = self.redis_conn.pipeline(transaction=False)
                for output in tqdm(
                    self.pool.imap_unordered(
                        read_relation_label, self.files_index["plabels"], chunksize=1
                    )
                ):
                    pid_to_name, name_to_pid = output
                    for pid, name in pid_to_name.items():
                        pipe.set(f"pid_name:{pid}", name)
                    for name, pids in name_to_pid.items():
                        pipe.rpush(f"name_pid:{name}", *pids)
                pipe.execute()

                print("Reading and storing entity labels...")
                pipe = self.redis_conn.pipeline(transaction=False)
                for i, output in enumerate(
                    tqdm(
                        self.pool.imap_unordered(
                            read_entity_label, self.files_index["labels"], chunksize=1
                        )
                    )
                ):
                    qid_to_name, name_to_qid = output
                    for qid, name in qid_to_name.items():
                        pipe.set(f"qid_name:{qid}", name)
                    for name, qids in name_to_qid.items():
                        pipe.rpush(f"name_qid:{name}", *qids)
                    if (i + 1) % 50 == 0:
                        pipe.execute()
                pipe.execute()
                print("loading labels complete.")

                # Load merged p_degree data
                print("Reading and storing relation degrees...")
                pipe = self.redis_conn.pipeline(transaction=False)
                for pid_to_degree in tqdm(
                    self.pool.imap_unordered(
                        read_p_degrees, self.files_index["p_degree"], chunksize=1
                    ),
                    desc="Processing p_degree files"
                ):
                    for pid, degree in pid_to_degree.items():
                        pipe.set(f"relation_degree:{pid}", degree)
                pipe.execute()
                print("loading p_degree complete.")

            # Load aliases, descriptions and label_degree chunk by chunk
            self.files_index_chunked = {
                "aliases": get_batch_files(os.path.join(data_dir, "aliases")),
                "descriptions": get_batch_files(os.path.join(data_dir, "descriptions")),
                "label_degree": get_batch_files(os.path.join(data_dir, "label_degree")),
            }

            # Aliases
            alias_files = self.files_index_chunked["aliases"]
            chunk_size = math.ceil(len(alias_files) / num_chunks)
            start_index = (chunk_number - 1) * chunk_size
            end_index = min(start_index + chunk_size, len(alias_files))
            chunk_files = alias_files[start_index:end_index]
            
            print(f"Loading aliases data for chunk {chunk_number} into Redis...")
            pipe = self.redis_conn.pipeline(transaction=False)
            for i, output in enumerate(
                tqdm(
                    self.pool.imap_unordered(read_aliases, chunk_files, chunksize=1),
                    desc=f"Processing alias files for chunk {chunk_number}"
                )
            ):
                qid_to_aliases, alias_to_qids = output
                for qid, aliases in qid_to_aliases.items():
                    pipe.rpush(f"qid_aliases:{qid}", *aliases)
                for alias, qids in alias_to_qids.items():
                    pipe.rpush(f"alias_qids:{alias}", *qids)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()

            # Descriptions
            desc_files = self.files_index_chunked["descriptions"]
            chunk_size = math.ceil(len(desc_files) / num_chunks)
            start_index = (chunk_number - 1) * chunk_size
            end_index = min(start_index + chunk_size, len(desc_files))
            chunk_files = desc_files[start_index:end_index]

            print(f"Loading descriptions data for chunk {chunk_number} into Redis...")
            pipe = self.redis_conn.pipeline(transaction=False)
            for i, qid_to_desc in enumerate(
                tqdm(
                    self.pool.imap_unordered(read_descriptions, chunk_files, chunksize=1),
                    desc=f"Processing description files for chunk {chunk_number}"
                )
            ):
                for qid, desc in qid_to_desc.items():
                    pipe.set(f"qid_desc:{qid}", desc)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()

            # Label degrees
            label_degree_files = self.files_index_chunked["label_degree"]
            chunk_size = math.ceil(len(label_degree_files) / num_chunks)
            start_index = (chunk_number - 1) * chunk_size
            end_index = min(start_index + chunk_size, len(label_degree_files))
            chunk_files = label_degree_files[start_index:end_index]

            print(f"Loading label degrees data for chunk {chunk_number} into Redis...")
            pipe = self.redis_conn.pipeline(transaction=False)
            for i, qid_to_degrees in enumerate(
                tqdm(
                    self.pool.imap_unordered(read_label_degrees, chunk_files, chunksize=1),
                    desc=f"Processing label degree files for chunk {chunk_number}"
                )
            ):
                for qid, degrees in qid_to_degrees.items():
                    pipe.hset(f"entity_degree:{qid}", mapping=degrees)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()

            print(f"Loading index data for chunk {chunk_number} into Redis...")
            # 1. relation_entities
            pickle_path = f"{data_dir}/indices/relation_entities_chunk_{chunk_number}.pickle"
            print(f"Reading {pickle_path}")
            with open(pickle_path, "rb") as handle:
                data = pickle.load(handle)
            pipe = self.redis_conn.pipeline(transaction=False)
            print("Storing relation_entities...")
            for i, (qid, rels) in enumerate(tqdm(data.items())):
                if rels.get("head"):
                    relations_to_store = [
                        json.dumps({"pid": pid, "label": label, "counter": count})
                        for (pid, label), count in rels["head"].items()
                    ]
                    pipe.rpush(f"relations_head:{qid}", *relations_to_store)
                if rels.get("tail"):
                    relations_to_store = [
                        json.dumps({"pid": pid, "label": label, "counter": count})
                        for (pid, label), count in rels["tail"].items()
                    ]
                    pipe.rpush(f"relations_tail:{qid}", *relations_to_store)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()
            print('sample linked relations', list(data.keys())[:100])
            del data

            # 2. tail_entities
            pickle_path = f"{data_dir}/indices/tail_entities_chunk_{chunk_number}.pickle"
            print(f"Reading {pickle_path}")
            with open(pickle_path, "rb") as handle:
                data = pickle.load(handle)
            pipe = self.redis_conn.pipeline(transaction=False)
            print("Storing tail_entities...")
            for i, (key, entities) in enumerate(tqdm(data.items())):
                if entities.get("head"):
                    entities_to_store = [json.dumps(e) for e in entities["head"]]  # [(qid, label), ...]
                    pipe.rpush(f"rel_pairs:head:{key}", *entities_to_store)
                if entities.get("tail"):
                    entities_to_store = [json.dumps(e) for e in entities["tail"]]
                    pipe.rpush(f"rel_pairs:tail:{key}", *entities_to_store)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()
            print('sample tail entities', list(data.keys())[:100])
            del data

            # 3. tail_values
            pickle_path = f"{data_dir}/indices/tail_values_chunk_{chunk_number}.pickle"
            print(f"Reading {pickle_path}")
            with open(pickle_path, "rb") as handle:
                data = pickle.load(handle)
            pipe = self.redis_conn.pipeline(transaction=False)
            print("Storing tail_values...")
            for i, (key, values) in enumerate(tqdm(data.items())):
                if values:
                    pipe.rpush(f"tail_values:{key}", *values)
                if (i + 1) % 100 == 0:
                    pipe.execute()
            pipe.execute()
            print('sample tail values', list(data.keys())[:100])
            del data



            # # 4. external_ids
            # pickle_path = f"{data_dir}/indices/external_ids_chunk_{chunk_number}.pickle"
            # print(f"Reading {pickle_path}")
            # with open(pickle_path, "rb") as handle:
            #     data = pickle.load(handle)
            # pipe = self.redis_conn.pipeline()
            # print("Storing external_ids...")
            # for i, (key, values) in enumerate(tqdm(data.items())):
            #     if values:
            #         pipe.rpush(f"ext_ids:{key}", *values)
            #     if (i + 1) % 1000 == 0:
            #         pipe.execute()
            # pipe.execute()
            # del data

            # # 5. mid_to_qid
            # pickle_path = f"{data_dir}/indices/mid_to_qid_chunk_{chunk_number}.pickle"
            # print(f"Reading {pickle_path}")
            # with open(pickle_path, "rb") as handle:
            #     data = pickle.load(handle)
            # pipe = self.redis_conn.pipeline()
            # print("Storing mid_to_qid...")
            # for i, (mid, qids) in enumerate(tqdm(data.items())):
            #     if qids:
            #         pipe.rpush(f"mid_qid:{mid}", *qids)
            #     if (i + 1) % 1000 == 0:
            #         pipe.execute()
            # pipe.execute()
            # del data
            self.redis_conn.set(f"wikidata_data_loaded_chunk_{chunk_number}", "true")
            print("Index data loading complete.")

            print("Data loading finished. Triggering manual background save...")
            while int(self.redis_conn.info('persistence').get('rdb_bgsave_in_progress', 0)):
                print("An existing RDB save is in progress, waiting for it to complete...")
                time.sleep(5)
            self.redis_conn.bgsave()
        finally:
            print("Restoring original Redis save configuration...")
            if original_save_config and "save" in original_save_config:
                self.redis_conn.config_set("save", original_save_config["save"])
            if (
                original_aof_rewrite_config
                and "auto-aof-rewrite-percentage" in original_aof_rewrite_config
            ):
                self.redis_conn.config_set(
                    "auto-aof-rewrite-percentage",
                    original_aof_rewrite_config["auto-aof-rewrite-percentage"],
                )
            print("Redis configuration restored.")

    def label2qid(self, label: str) -> tp.List[str]:
        qids = self.redis_conn.lrange(f"name_qid:{label}", 0, -1)
        return qids if qids else "Not Found!"

    def label2pid(self, label: str) -> tp.List[str]:
        pids = self.redis_conn.lrange(f"name_pid:{label}", 0, -1)
        return pids if pids else "Not Found!"

    def qid2label(self, qid: str) -> str:
        return self.redis_conn.get(f"qid_name:{qid}") or "Not Found!"

    def pid2label(self, pid: str) -> str:
        return self.redis_conn.get(f"pid_name:{pid}") or "Not Found!"

    def mid2qid(self, mid: str) -> tp.List[str]:
        qids = self.redis_conn.lrange(f"mid_qid:{mid}", 0, -1)
        return qids if qids else "Not Found!"

    def get_all_relations_of_an_entity(self, entity_qid: str):
        pipe = self.redis_conn.pipeline()
        pipe.lrange(f"relations_head:{entity_qid}", 0, -1)
        pipe.lrange(f"relations_tail:{entity_qid}", 0, -1)
        raw_head, raw_tail = pipe.execute()

        if not raw_head and not raw_tail:
            return "Not Found!"

        relations = {'head': [], 'tail': []}
        if raw_head:
            # relations["head"] = [json.loads(r) for r in raw_head]
            relations["head"] = raw_head
        if raw_tail:
            # relations["tail"] = [json.loads(r) for r in raw_tail]
            relations["tail"] = raw_tail
        return relations

    def get_tail_entities_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.Dict[str, list]:
        key = f"{head_qid}@{relation_pid}"
        pipe = self.redis_conn.pipeline()
        pipe.lrange(f"rel_pairs:head:{key}", 0, -1)
        pipe.lrange(f"rel_pairs:tail:{key}", 0, -1)
        raw_head, raw_tail = pipe.execute()

        if not raw_head and not raw_tail:
            return "Not Found!"

        entities = {'head': [], 'tail': []}
        if raw_head:
            # entities["head"] = [json.loads(e) for e in raw_head]
            entities["head"] = raw_head
        if raw_tail:
            # entities["tail"] = [json.loads(e) for e in raw_tail]
            entities["tail"] = raw_tail
        return entities

    def get_tail_values_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        key = f"{head_qid}@{relation_pid}"
        values = self.redis_conn.lrange(f"tail_values:{key}", 0, -1)
        return values if values else "Not Found!"

    def get_external_id_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        key = f"{head_qid}@{relation_pid}"
        values = self.redis_conn.lrange(f"ext_ids:{key}", 0, -1)
        return values if values else "Not Found!"

    def qid2aliases(self, qid: str) -> tp.List[str]:
        aliases = self.redis_conn.lrange(f"qid_aliases:{qid}", 0, -1)
        return aliases if aliases else "Not Found!"

    def alias2qids(self, alias: str) -> tp.List[str]:
        qids = self.redis_conn.lrange(f"alias_qids:{alias}", 0, -1)
        return qids if qids else "Not Found!"

    def qid2description(self, qid: str) -> str:
        return self.redis_conn.get(f"qid_desc:{qid}") or "Not Found!"

    def get_label_degree(self, qid: str) -> tp.Dict[str, int]:
        degree = self.redis_conn.hgetall(f"entity_degree:{qid}")
        if not degree:
            return "Not Found!"
        return {k: int(v) for k, v in degree.items()}

    def get_p_degree(self, pid: str) -> int:
        count = self.redis_conn.get(f"relation_degree:{pid}")
        return int(count) if count is not None else "Not Found!"

    def find_similar_entities(self, name: str, k: int = 5, nprobe: int = 64) -> tp.List[tp.Dict[str, tp.Any]]:
        if self.vector_search:
            return self.vector_search.search(name, k, nprobe)
        return []


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


class XMLRPCWikidataQueryServer(WikidataQueryServer):
    def __init__(self, addr, server_args, requestHandler=RequestHandler):
        super().__init__(
            chunk_number=server_args.chunk_number,
            data_dir=server_args.data_dir,
            redis_host=server_args.redis_host,
            redis_port=server_args.redis_port,
            redis_db=server_args.redis_db,
            flush_redis=server_args.flush_redis,
            num_chunks=server_args.num_chunks,
            vector_search_device=server_args.vector_search_device,
            vector_search_model=server_args.vector_search_model,
        )
        self.server = SimpleXMLRPCServer(addr, requestHandler=requestHandler)
        self.server.register_introspection_functions()
        self.server.register_function(self.get_all_relations_of_an_entity)
        self.server.register_function(
            self.get_tail_entities_given_head_and_relation
        )
        self.server.register_function(self.label2pid)
        self.server.register_function(self.label2qid)
        self.server.register_function(self.pid2label)
        self.server.register_function(self.qid2label)
        self.server.register_function(
            self.get_tail_values_given_head_and_relation
        )
        self.server.register_function(
            self.get_external_id_given_head_and_relation
        )
        self.server.register_function(self.mid2qid)
        self.server.register_function(self.qid2aliases)
        self.server.register_function(self.alias2qids)
        self.server.register_function(self.qid2description)
        self.server.register_function(self.get_label_degree)
        self.server.register_function(self.get_p_degree)
        self.server.register_function(self.find_similar_entities)

    def serve_forever(self):
        self.server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--chunk_number", type=int, required=True, help="Chunk number"
    )
    parser.add_argument("--port", type=int, default=23546, help="Port number")
    parser.add_argument("--host_ip", type=str, required=True, help="Host IP")
    parser.add_argument(
        "--redis_host", type=str, default="localhost", help="Redis host"
    )
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis_db", type=int, default=0, help="Redis DB number")
    parser.add_argument(
        "--flush_redis",
        action="store_true",
        help="Flush Redis DB before loading data.",
    )
    parser.add_argument(
        "--num_chunks", type=int, default=6, help="Total number of chunks"
    )
    parser.add_argument(
        "--vector_search_device", type=str, default=None, help="Device for vector search model, e.g. 'cuda:0'"
    )
    parser.add_argument(
        "--vector_search_model", type=str, default='BAAI/bge-m3', help="Sentence transformer model for vector search"
    )
    args = parser.parse_args()
    print("Start with my program now!!!")
    server = XMLRPCWikidataQueryServer(
        addr=("0.0.0.0", args.port), server_args=args
    )
    # with open("server_urls.txt", "a") as f:
    #     f.write(f"http://{args.host_ip}:{args.port}\n")
    print(f"XMLRPC WDQS server ready and listening on 0.0.0.0:{args.port}", flush=True)
    server.serve_forever()
