# Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy (CoG)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Note for Reviewers**: This repository is provided for anonymous peer review. 
> Full documentation and author information will be disclosed upon acceptance.

CoG is a novel multi-hop question answering framework that synergistically combines structured knowledge graphs (Wikidata) and unstructured text (Wikipedia) through an iterative cognitive process of planning, exploration, and reflection. By leveraging bidirectional graph-text synergy, CoG navigates massive knowledge spaces to answer complex questions requiring multi-step reasoning.

## 🌟 Key Features

- **Bidirectional Graph-Text Synergy**: Seamlessly integrates Wikidata knowledge graph and Wikipedia text for comprehensive information retrieval
- **Cognitive Reasoning Cycles**: Iterative planning, exploration, and reflection process mimicking human problem-solving
- **Adaptive Information Gathering**: Dynamically adjusts search strategies based on information utility assessment
- **Scalable Architecture**: Distributed Wikidata query service supporting parallel processing
- **Multi-Dataset Support**: Evaluated on 7 benchmark datasets including HotpotQA, MusiQue, 2WikiMultihopQA, and more

## 📁 Project Structure

```
CoGOnGraph/
├── CoG/                           # Main CoG framework implementation
│   ├── main.py                    # Entry point for running experiments
│   ├── plan/                      # Initial planning module
│   ├── kg/                        # Knowledge graph exploration module
│   ├── wiki/                      # Wikipedia retrieval module
│   ├── think/                     # Information synthesis and reflection module
│   ├── answer/                    # Answer generation module
│   ├── results/                   # Experiment results directory
│   └── README.md                  # Detailed usage documentation
├── Wikidata/                      # Wikidata service deployment
│   ├── simple_wikidata_db/        # Wikidata database library
│   └── README.md                  # Wikidata setup guide
├── eval/                          # Evaluation scripts
│   ├── eval.py                    # Evaluation script
│   └── README.md                  # Evaluation documentation
├── data/                          # Dataset directory
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

**Prerequisites:**
- Python 3.10 or higher
- CUDA-compatible GPU (optional, for faster processing)
- 500GB+ storage for Wikidata dump
- 500GB+ RAM for Wikidata processing and service deployment

**Install dependencies:**

```bash
cd CoGOnGraph
pip install -r requirements.txt
```

### 2. Deploy Wikidata Service

Before running experiments, you need to deploy the Wikidata query service. See detailed instructions in [`Wikidata/README.md`](Wikidata/README.md).

**Quick deployment steps:**
1. Download Wikidata dump
2. Preprocess the dump
3. Build index files
4. Start query servers

### 3. Run Experiments

Navigate to the CoG directory and run experiments on your chosen dataset:

```bash
cd CoG
CUDA_VISIBLE_DEVICES=0 python main.py --dataset hotpot_e
```
For detailed parameter descriptions and usage examples, see [`CoG/README.md`](CoG/README.md).

### 4. Evaluate Results

After running experiments, evaluate the results using the evaluation script:

```bash
cd eval
python eval.py \
    --experiment_id 260101_182426_musique_Qwen3-32B_bge-m3 \
    --dataset musique
```

For detailed evaluation instructions, see [`eval/README.md`](eval/README.md).

## 📊 Supported Datasets

CoG supports the following benchmark datasets:

| Dataset | Parameter | Description |
|---------|-----------|-------------|
| KGQAGen-10k | `KGQAGen` | High-quality Wikidata-based KGQA dataset with verified multi-hop reasoning instances |
| ComplexWebQuestions | `cwq` | Complex web questions |
| QALD10-en | `qald` | Multilingual QA dataset (English) |
| WebQuestionsSP | `webqsp` | Freebase-based simple questions |
| 2WikiMultihopQA | `2wiki` | Multi-hop QA dataset combining structured and unstructured data |
| AdvHotpotQA | `hotpot_e` | HotpotQA adversarial samples |
| MusiQue | `musique` | Multi-hop reasoning QA |


### Entity-Annotated Dataset Files

Some datasets `include *_entities_azure.json` files containing the original questions and their recognized entity annotations. These files are provided as auxiliary inputs for baseline models or evaluation pipelines that benefit from explicit entity information. By default, CoG uses only the standard dataset files.

## 🔧 System Architecture

CoG employs a cognitive reasoning cycle consisting of four main stages:

1. **Initial Planning**: Analyze the question and generate initial exploration queries
2. **Information Gathering**: 
   - **KG Exploration**: Navigate Wikidata to discover entity relations and facts
   - **Wiki Retrieval**: Retrieve relevant text and tables from Wikipedia
3. **Synthesis & Judgment**: Evaluate collected information and determine sufficiency
4. **Adaptive Planning**: 
   - If sufficient → Generate final answer
   - If useful but insufficient → Plan next exploration step
   - If useless → Reflect and adjust strategy


## 📖 Documentation

- **[CoG/README.md](CoG/README.md)**: Detailed usage guide for running experiments
- **[Wikidata/README.md](Wikidata/README.md)**: Complete guide for deploying Wikidata service
- **[eval/README.md](eval/README.md)**: Instructions for evaluating results

## 🐛 Troubleshooting

### Common Issues

1. **Wikidata Service Connection Failed**
   - Ensure all server instances are running
   - Check `server_urls.txt` configuration
   - Verify Redis is running: `sudo systemctl status redis`

2. **Wikipedia API Timeout**
   - System automatically retries failed questions (up to 3 times)
   - Use `--rerun_failed_only` to rerun failed questions

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- [Wikidata](https://www.wikidata.org/) for providing the knowledge graph
- [Wikipedia](https://www.wikipedia.org/) for providing the text corpus
- [ToG-2](https://github.com/DataArcTech/ToG-2) for the foundational Wikidata deployment codebase and evaluation methodology
