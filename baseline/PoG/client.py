"""Minimal multi-server Wikidata XML-RPC client used by PoG."""

import json
import re
import typing as tp
import xmlrpc.client
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class WikidataQueryClient:
    def __init__(self, url: str):
        self.url = url
        self.server = xmlrpc.client.ServerProxy(url)

    def get_all_relations_of_an_entity(self, entity_qid: str):
        return self.server.get_all_relations_of_an_entity(entity_qid)

    def get_tail_entities_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ):
        return self.server.get_tail_entities_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_tail_values_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ):
        return self.server.get_tail_values_given_head_and_relation(
            head_qid, relation_pid
        )


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.Sequence[str]):
        if not urls:
            raise ValueError("At least one Wikidata server URL is required")
        self.clients = [WikidataQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(self.clients))
        self._test_connections()

    def _test_connections(self):
        def is_reachable(client):
            try:
                client.server.system.listMethods()
                return True
            except Exception as exc:
                print(f"Failed to connect to {client.url}: {exc}")
                return False

        futures = [
            self.executor.submit(is_reachable, client) for client in self.clients
        ]
        self.clients = [
            client
            for client, future in zip(self.clients, futures)
            if future.result()
        ]
        if not self.clients:
            raise ConnectionError("Failed to connect to all Wikidata servers")
        print(f"Connected to {len(self.clients)} Wikidata server(s).")

    def query_all(self, method, *args):
        sanitized_args = tuple(
            re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", arg)
            if isinstance(arg, str)
            else arg
            for arg in args
        )
        futures = [
            self.executor.submit(getattr(client, method), *sanitized_args)
            for client in self.clients
        ]
        results = [future.result() for future in futures]

        if method == "get_all_relations_of_an_entity":
            merged = {"head": Counter(), "tail": Counter()}
            for result in results:
                if result == "Not Found!":
                    continue
                for direction in ("head", "tail"):
                    for item_text in result.get(direction, []):
                        item = json.loads(item_text)
                        merged[direction][(item["pid"], item["label"])] += item[
                            "counter"
                        ]
            return {
                direction: [
                    {"pid": pid, "label": label, "counter": count}
                    for (pid, label), count in values.items()
                ]
                for direction, values in merged.items()
            }

        if method == "get_tail_entities_given_head_and_relation":
            merged = {"head": set(), "tail": set()}
            for result in results:
                if result == "Not Found!":
                    continue
                for direction in ("head", "tail"):
                    merged[direction].update(result.get(direction, []))
            keys = ("qid", "label")
            return {
                direction: [dict(zip(keys, json.loads(item))) for item in values]
                for direction, values in merged.items()
            }

        merged = set()
        for result in results:
            if result == "Not Found!":
                continue
            if isinstance(result, list):
                merged.update(result)
            elif result is not None:
                merged.add(result)
        return list(merged)
