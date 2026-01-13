from collections import defaultdict, Counter
from dataclasses import dataclass
from traitlets import default
import ujson as json
import os
import warnings


def a_factory():
    return {"head": set(), "tail": set()}


def b_factory():
    return {"head": Counter(), "tail": Counter()}

def entity_degree_factory():
    return {"in": 0, "out": 0, "attr": 0}

def jsonl_generator(fname):
    """Returns generator for jsonl file."""
    with open(fname, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.endswith(','):
                line = line[:-1]

            try:
                d = json.loads(line)
                yield d
            except json.JSONDecodeError:
                warnings.warn(f"Could not parse line {i} in {fname}: {line}")
                continue


def get_batch_files(fdir):
    """Returns paths to files in fdir."""
    filenames = os.listdir(fdir)
    filenames.sort()
    filenames = [os.path.join(fdir, f) for f in filenames]
    print(f"Fetched {len(filenames)} files from {fdir}, {filenames[:5]} ...")
    return filenames


# Build these 4 dictionaries
def read_entity_label(filename):
    qid_to_name = {}
    name_to_qid = defaultdict(list)
    for item in jsonl_generator(filename):
        qid_to_name[item["qid"]] = item["label"]
        name_to_qid[item["label"]].append(item["qid"])
    return qid_to_name, name_to_qid


def read_relation_label(filename):
    pid_to_name = {}
    name_to_pid = defaultdict(list)
    for item in jsonl_generator(filename):
        pid_to_name[item["pid"]] = item["label"]
        name_to_pid[item["label"]].append(item["pid"])
    return pid_to_name, name_to_pid
