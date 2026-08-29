"""Wikidata XML-RPC and Wikipedia clients for the CoG ToG-2 baseline.

Adapted from IDEA-FinAI/ToG-2 and modified for CoG integration.
"""

import xmlrpc.client
import typing as tp
import json
import requests
from bs4 import BeautifulSoup
from collections import Counter

def format_entity_name_for_wikipedia(entity_name):
    return entity_name.replace(' ', '_')


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

    def get_wikipedia_page(self, ent_dict, section: str = None) -> str:
        try:
            if ent_dict.get('name') and ent_dict['name'] != "Not Found!":
                entity_name = format_entity_name_for_wikipedia(ent_dict['name'])
            elif ent_dict['id'] != 'None':
                qid = ent_dict['id']
                entity_name = self.server.get_wikipedia_link(qid)
                entity_name = entity_name[0]
            else:
                return "Not Found!"

            if entity_name == "Not Found!":
                return "Not Found!"
            else:
                wikipedia_url = 'https://en.wikipedia.org/wiki/{}'.format(entity_name)
                # print('wikipedia_url  ' + wikipedia_url)
                # time.sleep(1)
                headers = {
                    'Connection': 'close',
                    'User-Agent': 'CoG-ToG-2/1.0 (https://github.com/IDEA-FinAI/ToG-2)'
                }
                response = requests.get(wikipedia_url, headers=headers, timeout=180)
                response.raise_for_status()  

                soup = BeautifulSoup(response.content, "html.parser")
                content_div = soup.find("div", {"id": "bodyContent"})
                if content_div is None:
                    return "Not Found!"

                # Remove script and style elements
                for script_or_style in content_div.find_all(["script", "style"]):
                    script_or_style.decompose()

                if section:
                    header = content_div.find(
                        lambda tag: tag.name == "h2" and section in tag.get_text()
                    )
                    if header:
                        content = ""
                        for sibling in header.find_next_siblings():
                            if sibling.name == "h2":
                                break
                            content += sibling.get_text()
                        return content.strip()
                    else:
                        return f"Section '{section}' not found."

                summary_content = ""
                for element in content_div.find_all(recursive=False): # may be False for summary only
                    if element.name == "h2":
                        break
                    summary_content += element.get_text()

                return summary_content.strip()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return "Not Found!"
            print(f"Error fetching Wikipedia page: {e}")
            return "Fetch Error!"
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Wikipedia page: {e}")
            return "Fetch Error!"



from concurrent.futures import ThreadPoolExecutor


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.List[str]):
        self.clients = [WikidataQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(urls))
       

    def test_connections(self):
        def test_url(client):
            try:
               
                client.server.system.listMethods()
                return True
            except Exception as e:
                print(f"Failed to connect to {client.url}. Error: {str(e)}")
                return False

        futures = [
            self.executor.submit(test_url, client) for client in self.clients
        ]
        results = [f.result() for f in futures]
       
        self.clients = [
            client for client, result in zip(self.clients, results) if result
        ]
        if not self.clients:
            raise Exception("Failed to connect to all URLs")

        print(f"Connected to {len(self.clients)} Wikidata server(s).")

    def get_wikipedia_page(self, entity, section=None):
        """Fetch Wikipedia through the first available client."""
        if not self.clients:
            raise RuntimeError("No available Wikidata client")
        return self.clients[0].get_wikipedia_page(entity, section=section)

   
    
    def query_all(self, method, *args):
        # start_time = time.perf_counter()
        clients = self.clients
        if method in ['get_p_degree', 'label2qid', 'pid2label', 'qid2label', 'label2pid', 'get_wikipedia_page']:
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
