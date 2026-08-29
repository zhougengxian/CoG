"""Load CoG datasets and merge their entity annotations."""

import json
from pathlib import Path

DATASET_FILES = {
    "cwq": ("cwq.json", None),
    "webqsp": ("webqsp_test.json", None),
    "qald": ("qald_10-en.json", None),
    "hotpot_e": ("hotpotadv_dev.json", "hotpotadv_entities_azure.json"),
    "2wiki": ("2wikimultihopqa.json", "2wikimultihopqa_entities_azure.json"),
    "KGQAGen": ("KGQAGen-10k.json", "KGQAGen-10k_entities_azure.json"),
    "musique": ("musique.json", "musique_entities_azure.json"),
}


def default_data_dir():
    return Path(__file__).resolve().parents[2] / "data"


def _read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_dataset(dataset_name, data_dir=None):
    if dataset_name not in DATASET_FILES:
        supported = ", ".join(DATASET_FILES)
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Choose from: {supported}")
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_file, entity_file = DATASET_FILES[dataset_name]
    records = _read_json(data_dir / data_file)
    entity_map = {}
    if entity_file:
        entity_records = _read_json(data_dir / entity_file)
        entity_map = {item["question"]: item.get("entities", {}) for item in entity_records}
    normalized = []
    for item in records:
        item = dict(item)
        question = item["question"]
        topic_entity = item.get("qid_topic_entity")
        if topic_entity is None:
            topic_entity = item.get("entities")
        if topic_entity is None:
            topic_entity = entity_map.get(question, {})
        item["topic_entity"] = topic_entity or {}
        normalized.append(item)
    return normalized, "question"
