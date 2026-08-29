"""Dense retrieval and JSONL output helpers for the CoG ToG-2 baseline.

Adapted from IDEA-FinAI/ToG-2. Modified for CoG baseline integration.
"""

import json
import re
from contextlib import redirect_stderr
from pathlib import Path

def quiet_model_call(func, *args, **kwargs):
    with open("/dev/null", "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
        return func(*args, **kwargs)


def scores_rank(scores, texts):
    items = [{"score": float(score), "text": text} for score, text in zip(scores, texts)]
    return sorted(items, key=lambda item: item["score"], reverse=True)


def s2p_relevance_scores(texts, question, args, emb_model):
    """Rank texts with dense vectors from either bge-bi or bge-m3."""
    del args
    if not texts:
        return []
    query_embedding = quiet_model_call(emb_model.encode_queries, [question])
    passage_embeddings = quiet_model_call(emb_model.encode_passages, texts)
    return (query_embedding @ passage_embeddings.T)[0]


def split_paragraphs(text):
    paragraphs = []
    current = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))

    filtered = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\[\d+\]", "", paragraph).strip()
        rows = paragraph.splitlines()
        if (paragraph and not paragraph.startswith("^") and len(rows) > 1
                and len(paragraph) >= 50 and len(paragraph) / len(rows) >= 50):
            filtered.append(paragraph)
    return filtered


def split_sentences_windows(text, window_size=2, step_size=1):
    from blingfire import text_to_sentences_and_offsets

    if not text:
        return []
    offsets = text_to_sentences_and_offsets(text)[1]
    sentences = [text[start:end] for start, end in offsets]
    if window_size > 1 and len(sentences) >= window_size:
        sentences = [
            " ".join(sentences[index:index + window_size])
            for index in range(0, len(sentences) - window_size + 1, step_size)
        ]
    return sentences


def split_sentences(text):
    paragraphs = [line for line in text.splitlines() if len(line) >= 10]
    sentences = re.split(r"\.", ".".join(paragraphs))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 10]


def pages_embedding_search(question, related_passage, args, emb_model, top_k=3):
    if related_passage in {"Not Found!", "Fetch Error!", ""}:
        return "", []
    paragraphs = split_paragraphs(related_passage)
    if not paragraphs:
        return "", []
    paragraph_scores = s2p_relevance_scores(paragraphs, question, args, emb_model)
    ranked_paragraphs = scores_rank(paragraph_scores, paragraphs)
    paragraph = "".join(item["text"] for item in ranked_paragraphs[:3])
    sentences = split_sentences_windows(paragraph)
    sentence_scores = s2p_relevance_scores(sentences, question, args, emb_model)
    return paragraph, scores_rank(sentence_scores, sentences)[:top_k]


def pages_embedding_search_only_para(related_passage):
    if related_passage in {"Not Found!", "Fetch Error!", ""}:
        return []
    return split_paragraphs(related_passage)


def _safe_filename_part(value):
    value = str(value).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "unnamed"


def get_run_type(args):
    if args.self_consistency:
        return "sc"
    if args.gpt_only:
        return f"gpt_only_{args.gpt_only}"
    return "tog2"


def get_output_filename(args):
    parts = [get_run_type(args), args.dataset, args.model]
    if args.file_suffix:
        parts.append(args.file_suffix)
    return "__".join(_safe_filename_part(part) for part in parts) + ".jsonl"


def load_completed_questions(output_file):
    output_file = Path(output_file)
    if not output_file.exists():
        return set()
    completed = set()
    with output_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {output_file} at line {line_number}") from exc
            if "question" in item:
                completed.add(item["question"])
    return completed


def append_result(output_file, question, answer):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"question": question, "answer": answer}
    with output_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
