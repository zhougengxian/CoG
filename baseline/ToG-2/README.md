# ToG-2 baseline for CoG

This baseline is adapted from
[IDEA-FinAI/ToG-2](https://github.com/IDEA-FinAI/ToG-2), the implementation
of *Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with
Knowledge-guided Retrieval Augmented Generation*.

The copied source has been modified for the CoG release: it reads the shared
CoG datasets and entity annotations, uses the CoG Wikidata XML-RPC services,
writes resumable JSONL results, supports only `bge-bi` and `bge-m3` dense
retrieval, and does not collect token, timing, or document-usage statistics.

## Installation

Install CoG first, then install the baseline-specific dependencies:

```bash
pip install -r baseline/ToG-2/requirements.txt
```

The embedding weights are downloaded by FlagEmbedding on first use.

## Wikidata services

`server_urls.txt` contains the default local service addresses and is also the
configuration template. Edit its entries or pass another file with
`--server-urls`. Blank lines and lines beginning with `#` are ignored. One or
more reachable services are supported.

## Run

The default LLM is `Qwen3-32B`. Examples that initialize FlagEmbedding expose
only GPU 0 so it does not start workers on every visible GPU. Change
`CUDA_VISIBLE_DEVICES` when another GPU should be used. GPT-only modes do not
load FlagEmbedding and therefore do not need this environment variable.

Run standard ToG-2:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/ToG-2/main.py \
  --dataset 2wiki \
  --model Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --embedding-model bge-bi
```

GPT-only modes:

```bash
python baseline/ToG-2/main.py --dataset qald --model Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 --gpt-only cot
python baseline/ToG-2/main.py --dataset qald --model Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 --gpt-only io
```

Self-consistency mode:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/ToG-2/main.py --dataset qald --model Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 --self-consistency
```

To match the original ToG-2 code, `--self-consistency-threshold` defaults to
`-1` and `--max-length` defaults to `None`. Pass explicit values when a finite
generation limit or a stricter self-consistency gate is required.

Use `--file-suffix NAME` to distinguish runs with different experimental
settings. Output names follow:

```text
{run_type}__{dataset}__{model}[__{suffix}].jsonl
```

Examples:

```text
tog2__2wiki__Qwen3-32B.jsonl
gpt_only_cot__qald__Qwen3-32B__run1.jsonl
sc__cwq__Qwen3-32B.jsonl
```

The embedding model is intentionally not included in the filename. Use a
suffix when runs with different retrieval settings must coexist. Existing
questions in an output file are skipped when the command is resumed.

## Evaluate

```bash
cd eval
python eval.py \
  --dataset 2wiki \
  --output_file ../baseline/ToG-2/results/tog2__2wiki__Qwen3-32B.jsonl
```

## Supported datasets and retrieval models

Datasets: `cwq`, `webqsp`, `qald`, `hotpot_e`, `2wiki`, `KGQAGen`, and
`musique`. Retrieval models: `bge-bi` (default, `BAAI/bge-large-en-v1.5`) and
`bge-m3` (`BAAI/bge-m3`, dense vectors only).

## License and citation

The adapted ToG-2 code remains under the Apache License 2.0; see `LICENSE`.

```bibtex
@misc{ma2024thinkongraph20deepfaithful,
  title={Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation},
  author={Shengjie Ma and Chengjin Xu and Xuhui Jiang and Muzhi Li and Huaren Qu and Cehao Yang and Jiaxin Mao and Jian Guo},
  year={2024},
  eprint={2407.10805},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.10805}
}
```
