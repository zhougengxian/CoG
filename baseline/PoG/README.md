# PoG baseline for CoG

This baseline is adapted from
[liyichen-cly/PoG](https://github.com/liyichen-cly/PoG), the implementation of
*Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on
Knowledge Graphs*.

The release version preserves PoG's subobjective planning, relation selection,
entity pruning, memory update, reasoning, and backtracking flow. It reads the
datasets and entity annotations shipped with CoG, queries the CoG Wikidata
XML-RPC services, keeps per-question memory in process, writes resumable JSONL results.

## Installation

Install CoG first, then install the baseline-specific dependencies:

```bash
pip install -r baseline/PoG/requirements.txt
```

## Wikidata services

`server_urls.txt` contains the default local XML-RPC service addresses. Edit
the file or pass another file with `--server-urls`. Blank lines and lines
beginning with `#` are ignored. At least one reachable service is required.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/PoG/main.py \
  --dataset 2wiki \
  --model Qwen3-32B \
  --base-url http://127.0.0.1:9040/v1 \
  --api-key EMPTY
```

The default search depth is 4. Use `--start` to choose the first dataset row,
`--samples` to limit the number of selected rows, and `--suffix` to distinguish
runs with different settings. Existing questions in the output file are
skipped when a run is resumed.

For the model named exactly `Qwen3-32B`, `--enable-thinking` controls the
`chat_template_kwargs.enable_thinking` request field. Other model names do not
receive that field.

Output names follow:

```text
PoG_{model}_{dataset}[_{suffix}].jsonl
```

Each record contains only the fields required by the common evaluator:

```json
{"question": "...", "answer": "..."}
```

## Evaluate

```bash
cd eval
python eval.py \
  --dataset 2wiki \
  --output_file ../baseline/PoG/results/PoG_Qwen3-32B_2wiki.jsonl
```

## Supported datasets

`cwq`, `webqsp`, `qald`, `hotpot_e`, `2wiki`, `KGQAGen`, and `musique`.

## Citation and license

```bibtex
@inproceedings{chen2024pog,
  title={Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs},
  author={Chen, Liyi and Tong, Panrong and Jin, Zhongming and Sun, Ying and Ye, Jieping and Xiong, Hui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

The upstream README states that the project is licensed under Apache 2.0, but
the inspected source checkout does not contain a license file. Confirm the
upstream redistribution terms and add the applicable license before publishing
this adapted source.
