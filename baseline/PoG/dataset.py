"""Load CoG datasets and merge their entity annotations for PoG."""

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


def data_dir():
    return Path(__file__).resolve().parents[2] / "data"


def _read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_dataset(dataset_name):
    if dataset_name not in DATASET_FILES:
        supported = ", ".join(DATASET_FILES)
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. Choose from: {supported}"
        )

    dataset_dir = data_dir()
    dataset_file, entity_file = DATASET_FILES[dataset_name]
    records = _read_json(dataset_dir / dataset_file)

    entity_map = {}
    if entity_file:
        annotations = _read_json(dataset_dir / entity_file)
        entity_map = {
            item["question"]: item.get("entities", {}) for item in annotations
        }

    normalized = []
    for record in records:
        record = dict(record)
        question = record["question"]
        topic_entity = record.get("qid_topic_entity")
        if topic_entity is None:
            topic_entity = record.get("topic_entity")
        if topic_entity is None:
            topic_entity = record.get("entities")
        if topic_entity is None:
            topic_entity = entity_map.get(question, {})
        record["topic_entity"] = topic_entity or {}
        normalized.append(record)

    return normalized
