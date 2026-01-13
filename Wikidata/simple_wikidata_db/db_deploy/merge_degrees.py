import argparse
import glob
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
import time

import ujson
from tqdm import tqdm


class JsonlWriter:
    """
    A class to write json objects to sharded .jsonl files.
    """

    def __init__(self, base_dir: Path, table_name: str, batch_size: int):
        self.table_dir = base_dir / table_name
        if self.table_dir.exists():
            shutil.rmtree(self.table_dir)
        self.table_dir.mkdir(parents=True, exist_ok=False)
        self.batch_size = batch_size
        self.index = 0
        self.current_size = 0
        self.f_out = None
        self._open_new_file()

    def _open_new_file(self):
        if self.f_out:
            self.f_out.close()
        file_path = self.table_dir / f"{self.index}.jsonl"
        self.f_out = open(file_path, "w")
        self.index += 1

    def write(self, data):
        if self.current_size >= self.batch_size:
            self._open_new_file()
            self.current_size = 0

        self.f_out.write(ujson.dumps(data, ensure_ascii=False) + "\n")
        self.current_size += 1

    def close(self):
        if self.f_out:
            self.f_out.close()


def merge_label_degrees(input_dir: str):
    merged_degrees = defaultdict(lambda: {"in": 0, "out": 0, "attr": 0})
    label_degree_files = glob.glob(
        os.path.join(input_dir, "label_degree_chunk_*.pickle")
    )
    print(f"Found {len(label_degree_files)} label_degree chunk files.")

    for file_path in tqdm(label_degree_files, desc="Merging label degrees"):
        with open(file_path, "rb") as f:
            chunk_data = pickle.load(f)
            for qid, degrees in chunk_data.items():
                merged_degrees[qid]["in"] += degrees["in"]
                merged_degrees[qid]["out"] += degrees["out"]
                merged_degrees[qid]["attr"] += degrees["attr"]
    return merged_degrees


def merge_p_degrees(input_dir: str):
    merged_counts = defaultdict(int)
    p_degree_files = glob.glob(os.path.join(input_dir, "p_degree_chunk_*.pickle"))
    print(f"Found {len(p_degree_files)} p_degree chunk files.")

    for file_path in tqdm(p_degree_files, desc="Merging p_degrees"):
        with open(file_path, "rb") as f:
            chunk_data = pickle.load(f)
            for pid, count in chunk_data.items():
                merged_counts[pid] += count
    return merged_counts


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Merge and write label degrees
    print("Processing label degrees...")
    merged_label_degrees = merge_label_degrees(str(input_dir))
    label_degree_writer = JsonlWriter(
        output_dir, "label_degree", args.batch_size
    )
    for qid, degrees in tqdm(
        merged_label_degrees.items(), desc="Writing label degrees"
    ):
        record = {
            "qid": qid,
            "in_degree": degrees["in"],
            "out_degree": degrees["out"],
            "attr_degree": degrees["attr"],
        }
        label_degree_writer.write(record)
    label_degree_writer.close()
    print("Finished writing label degrees.")

    # Merge and write p degrees
    print("\nProcessing p degrees...")
    merged_p_degrees = merge_p_degrees(str(input_dir))
    p_degree_writer = JsonlWriter(output_dir, "p_degree", args.batch_size)
    for pid, count in tqdm(merged_p_degrees.items(), desc="Writing p degrees"):
        record = {"pid": pid, "degree": count}
        p_degree_writer.write(record)
    p_degree_writer.close()
    print("Finished writing p degrees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge and store degree information from build_index chunks."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with chunked pickle files from build_index.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store merged and sharded degree files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20000,
        help="Number of items per jsonl file.",
    )
    args = parser.parse_args()
    start_time = time.time()
    main(args) 
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")