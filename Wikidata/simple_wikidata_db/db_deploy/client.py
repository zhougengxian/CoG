import itertools
import xmlrpc.client
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
import ujson as json
from collections import Counter


class WikidataQueryClient:
    def __init__(self, url: str):
        self.url = url
        self.server = xmlrpc.client.ServerProxy(url)

    def label2qid(self, label: str) -> tp.List[str]:
        return self.server.label2qid(label)

    def label2pid(self, label: str) -> tp.List[str]:
        return self.server.label2pid(label)

    def pid2label(self, pid: str) -> str:
        return self.server.pid2label(pid)

    def qid2label(self, qid: str) -> str:
        return self.server.qid2label(qid)

    def get_all_relations_of_an_entity(
        self, entity_qid: str
    ) -> tp.Dict[str, tp.List]:
        return self.server.get_all_relations_of_an_entity(entity_qid)

    def get_tail_entities_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.Dict[str, tp.List]:
        return self.server.get_tail_entities_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_tail_values_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        return self.server.get_tail_values_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_external_id_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        return self.server.get_external_id_given_head_and_relation(
            head_qid, relation_pid
        )

    def mid2qid(self, mid: str) -> tp.List[str]:
        return self.server.mid2qid(mid)

    def qid2aliases(self, qid: str) -> tp.List[str]:
        return self.server.qid2aliases(qid)

    def alias2qids(self, alias: str) -> tp.List[str]:
        return self.server.alias2qids(alias)

    def qid2description(self, qid: str) -> str:
        return self.server.qid2description(qid)

    def get_label_degree(self, qid: str) -> tp.Dict[str, int]:
        return self.server.get_label_degree(qid)

    def get_p_degree(self, pid: str) -> int:
        return self.server.get_p_degree(pid)

    def find_similar_entities(self, name: str, k: int = 5, nprobe: int = 64) -> tp.List[tp.Dict[str, tp.Any]]:
        return self.server.find_similar_entities(name, k, nprobe)


import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.List[str]):
        self.clients = [WikidataQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(urls))
        # test connections
        start_time = time.perf_counter()
        self.test_connections()
        end_time = time.perf_counter()
        print(f"Connection testing took {end_time - start_time} seconds")

    def test_connections(self):
        def test_url(client):
            try:
                # Check if server provides the system.listMethods function.
                client.server.system.listMethods()
                return True
            except Exception as e:
                print(f"Failed to connect to {client.url}. Error: {str(e)}")
                return False

        # start_time = time.perf_counter()
        futures = [
            self.executor.submit(test_url, client) for client in self.clients
        ]
        results = [f.result() for f in futures]
        # end_time = time.perf_counter()
        # print(f"Testing connections took {end_time - start_time} seconds")
        # Remove clients that failed to connect
        self.clients = [
            client for client, result in zip(self.clients, results) if result
        ]
        if not self.clients:
            raise Exception("Failed to connect to all URLs")

    def query_all(self, method, *args):
        # start_time = time.perf_counter()
        clients = self.clients
        if method in ['get_p_degree', 'label2qid', 'pid2label', 'qid2label', 'label2pid']:
            clients = clients[:1]
        elif method == "find_similar_entities":
            clients = clients[-1:]
        futures = [
            self.executor.submit(getattr(client, method), *args)
            for client in clients
        ]
        results = [f.result() for f in futures]
        # end_time = time.perf_counter()
        # print(f"HTTP Queries took {end_time - start_time} seconds")

        # start_time = time.perf_counter()

        if method == "get_all_relations_of_an_entity":
            real_results = {"head": Counter(), "tail": Counter()}
            for res in results:
                if isinstance(res, str) and res == "Not Found!":
                    continue
                if res.get("head"):
                    for item_str in res["head"]:
                        item = json.loads(item_str)
                        real_results["head"][(item["pid"], item["label"])] += item["counter"]
                if res.get("tail"):
                    for item_str in res["tail"]:
                        item = json.loads(item_str)
                        real_results["tail"][(item["pid"], item["label"])] += item["counter"]
            
            final_results = {"head": [], "tail": []}
            for (pid, label), count in real_results["head"].items():
                final_results["head"].append({"pid": pid, "label": label, "counter": count})
            for (pid, label), count in real_results["tail"].items():
                final_results["tail"].append({"pid": pid, "label": label, "counter": count})
            return final_results

        if method == "find_similar_entities":
            # Only one result is expected from the first server
            res = results[0]
            if isinstance(res, list):
                return res
            return "Not Found!"

        if method == "get_tail_entities_given_head_and_relation":
            temp_head_set = set()
            temp_tail_set = set()
            for res in results:
                if isinstance(res, str) and res == "Not Found!":
                    continue
                if res.get("head"):
                    temp_head_set.update(res["head"])
                if res.get("tail"):
                    temp_tail_set.update(res["tail"])

            real_results = {"head": [], "tail": []}
            keys = ['qid', 'label']
            real_results['head'] = [dict(zip(keys, json.loads(e))) for e in temp_head_set]
            real_results['tail'] = [dict(zip(keys, json.loads(e))) for e in temp_tail_set]
            return real_results

        if method == "get_label_degree":
            real_results = Counter()
            for res in results:
                if isinstance(res, str) and res == "Not Found!":
                    continue
                if isinstance(res, dict):
                    real_results.update(res)
            return dict(real_results) if real_results else "Not Found!"

        if method == "get_p_degree":
            total_degree = 0
            found_any = False
            for res in results:
                if isinstance(res, int):
                    total_degree += res
                    found_any = True
            return total_degree if found_any else "Not Found!"
            
        real_results = set()
        for res in results:
            if res == "Not Found!":
                continue
            
            if isinstance(res, list):
                real_results.update(res)
            elif res is not None:
                real_results.add(res)
        
        # end_time = time.perf_counter()
        # print(f"Querying all took {end_time - start_time} seconds")

        return list(real_results) if len(real_results) > 0 else "Not Found!"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--addr_list",
        type=str,
        required=True,
        help="path to server address list",
    )
    args = parser.parse_args()

    with open(args.addr_list, "r") as f:
        server_addrs = f.readlines()
        server_addrs = [addr.strip() for addr in server_addrs]
    print(f"Server addresses: {server_addrs}")
    client = MultiServerWikidataQueryClient(server_addrs)
    print(
        f'MSFT\'s ticker code is  {client.query_all("get_tail_values_given_head_and_relation","Q2283","P249",)}'
    )
    print(
        f'relations of douglas adams is {client.query_all("get_all_relations_of_an_entity", "Q42")}'
    )
    print(
        f'tail entities of douglas adams educated at is {client.query_all("get_tail_entities_given_head_and_relation", "Q42", "P69")}'
    )
    print(
        f'label degree of douglas adams is {client.query_all("get_label_degree", "Q42")}'
    )
    print('\n--- Vector Search Example ---')
    similar_entities = client.query_all("find_similar_entities", "famous writer", 5, 128)
    if isinstance(similar_entities, list) and similar_entities:
        print(f'Entities similar to "famous writer":')
        for i, item in enumerate(similar_entities):
            print(f"Top{i+1}:")
            print(f"  Name: '{item['text']}'")
            print(f"  Score: {item['score']:.4f}")
            print(f"  Entity IDs: {item['qids']}")
    else:
        print('Could not find similar entities or an error occurred.')