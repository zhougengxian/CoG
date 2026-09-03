"""CoG release entry point for the PoG baseline."""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from client import MultiServerWikidataQueryClient
from dataset import DATASET_FILES, load_dataset
from pog import answer_question


BASELINE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"


def _safe_filename_part(value):
    value = str(value).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "unnamed"


def output_filename(args):
    parts = ["PoG", args.model, args.dataset]
    if args.suffix:
        parts.append(args.suffix)
    return "_".join(_safe_filename_part(part) for part in parts) + ".jsonl"


def load_server_urls(path):
    with Path(path).open(encoding="utf-8") as file:
        urls = [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not urls:
        raise ValueError(f"No Wikidata server URL found in {path}")
    return urls


def load_completed_questions(output_path):
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {output_path} at line {line_number}"
                ) from exc
            question = record.get("question")
            if question is not None:
                completed.add(question)
    return completed


def append_result(output_path, question, answer):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"question": question, "answer": answer}
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def build_parser():
    parser = argparse.ArgumentParser(description="Run the CoG PoG baseline")
    parser.add_argument("--dataset", choices=list(DATASET_FILES), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature-exploration", type=float, default=None)
    parser.add_argument("--temperature-reasoning", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--remove-unnecessary-rel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--base-url", default="http://127.0.0.1:9038/v1")
    parser.add_argument(
        "--server-urls", type=Path, default=BASELINE_DIR / "server_urls.txt"
    )
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR / "results")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--semantic-prune-limit", type=int, default=70)
    parser.add_argument("--candidate-pool-limit", type=int, default=10000)
    parser.add_argument("--garbage-sample-limit", type=int, default=10)
    parser.add_argument("--max-topic-entities", type=int, default=100)
    parser.add_argument("--max-backtracks", type=int, default=5)
    return parser


def apply_model_defaults(args):
    if args.model == "Qwen3-32B":
        default_temperature = 0.6 if args.enable_thinking else 0.7
        if args.temperature_exploration is None:
            args.temperature_exploration = default_temperature
        if args.temperature_reasoning is None:
            args.temperature_reasoning = default_temperature
        if args.top_p is None:
            args.top_p = 0.95 if args.enable_thinking else 0.8
    else:
        if args.temperature_exploration is None:
            args.temperature_exploration = 0.3
        if args.temperature_reasoning is None:
            args.temperature_reasoning = 0.3
    return args


def validate_args(args):
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.samples is not None and args.samples < 0:
        raise ValueError("--samples must be non-negative")
    if args.depth < 1:
        raise ValueError("--depth must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.temperature_exploration < 0 or args.temperature_reasoning < 0:
        raise ValueError("temperatures must be non-negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    for name in (
        "semantic_prune_limit",
        "candidate_pool_limit",
        "garbage_sample_limit",
        "max_topic_entities",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_backtracks < 0:
        raise ValueError("--max-backtracks must be non-negative")


def run(args):
    apply_model_defaults(args)
    validate_args(args)
    records = load_dataset(args.dataset)
    start = min(args.start, len(records))
    end = len(records)
    if args.samples is not None:
        end = min(end, start + args.samples)

    output_path = args.output_dir / output_filename(args)
    completed = load_completed_questions(output_path)
    print(f"Dataset: {args.dataset}; selected rows: [{start}, {end})")
    print(f"Output: {output_path}")
    if start >= end:
        return

    llm_client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=3600,
    )
    wiki_client = MultiServerWikidataQueryClient(
        load_server_urls(args.server_urls)
    )
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for index in tqdm(range(start, end)):
        record = records[index]
        question = record["question"]
        if question in completed:
            print(f"[{index}] Skip completed question")
            continue
        print(f"\n[{index}] Question: {question}")
        answer = answer_question(
            record, args, wiki_client, llm_client, embedding_model
        )
        append_result(output_path, question, answer)
        completed.add(question)


if __name__ == "__main__":
    run(build_parser().parse_args())
