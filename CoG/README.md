# CoG (Cognition on Graph)

CoG is a multi-hop question answering framework that combines knowledge graphs and Wikipedia text, answering complex questions through a continuous process of planning, exploration, and reflection.

## Directory Structure

```
CoG/
├── main.py                    # Main program entry
├── utils.py                   # Data preparation and utility functions
├── utils_log.py               # Experiment logging
├── utils_run.py               # Run configuration utilities
├── server_urls.txt            # Wikidata service address configuration
├── plan/                      # Initial planning module
├── wiki/                      # Wikipedia retrieval module
├── kg/                        # Knowledge graph exploration module
├── answer/                    # Answer generation module
├── results/                   # Experiment results output directory
```

## Requirements

### 1. Python Dependencies

Install dependencies from the project root directory:

```bash
pip install -r requirements.txt
```

### 2. Service Configuration

#### LLM Service
The system uses a locally deployed OpenAI-compatible API service by default. Configuration required:
- **Default URL**: `http://127.0.0.1:9034/v1`
- **Default Model**: `Qwen3-32B`

You can configure API keys via environment variables or a `.env` file (if using cloud services).

#### Wikidata Service
A Wikidata query service is required. Service addresses are configured in `server_urls.txt`:

```
http://127.0.0.1:23150
http://127.0.0.1:23151
http://127.0.0.1:23152
...
```

Refer to the `Wikidata/` folder in the project root directory to start the service.

## Quick Start

### Basic Usage

Run experiments on a specified dataset:

```bash
cd CoGOnGraph/CoG
CUDA_VISIBLE_DEVICES=3 python main.py --dataset hotpot_e
```

### Usage Examples

#### 1. Using Custom Models and Parameters
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset hotpot_e \
    --model Qwen3-32B \
    --base_url http://127.0.0.1:9034/v1 \
    --embedding_model BAAI/bge-m3 \
    --tag my_experiment
```

#### 2. Resuming Interrupted Experiments
```bash
python main.py \
    --resume_run_id 20251225_143022_hotpot_e \
    --run_unfinished
```

## Entity-Annotated Data

The project includes `*_entities_azure.json` files for several datasets. These are entity-annotated versions of the original questions, provided mainly as auxiliary inputs for baseline models or evaluation pipelines that benefit from explicit topic/entity hints. The CoG main workflow reads the standard dataset files by default.

## Main Parameters

### Dataset Parameters
- `--dataset`: Select dataset, supports the following seven datasets:
  - `KGQAGen` - KGQAGen-10k
  - `cwq` - ComplexWebQuestions (CWQ)
  - `qald` - QALD10-en
  - `webqsp` - WebQuestionsSP (WebQSP)
  - `2wiki` - 2WikiMultihopQA
  - `hotpot_e` - AdvHotpotQA
  - `musique` - MusiQue
- `--num_sample`: Limit the number of questions to run
- `--start`: Starting question index
- `--run_specific_indices`: Run specific question indices, e.g., `--run_specific_indices 0 1 2 3`

### Model Parameters
- `--model`: Base LLM model (default: `Qwen3-32B`)
- `--base_url`: LLM service address (default: `http://127.0.0.1:9034/v1`)
- `--embedding_model`: Embedding model (default: `BAAI/bge-m3`)
- `--kg_model`: Dedicated model for KG module (defaults to `--model`)
- `--kg_base_url`: Service address for KG module (defaults to `--base_url`)
- `--wiki_model`: Dedicated model for Wiki module (defaults to `--model`)
- `--wiki_base_url`: Service address for Wiki module (defaults to `--base_url`)

### Generation Parameters
- `--max_length`: Maximum LLM output length
- `--max_length_entity_link`: Maximum output length for entity linking module
- `--max_length_plan`: Maximum output length for planning module
- `--max_length_relation_discovery`: Maximum output length for relation discovery module
- `--enable_thinking`: Enable thinking mode for Qwen3 models (disabled by default)

### Workflow Parameters
- `--max_turns`: Maximum iteration turns (default: 4)
- `--skip_tier1_fallback`: Skip tier-1 fallback strategy (default: False)
- `--fact_pruning_retries`: Number of retries for KG fact pruning/extraction (default: 5)
- `--plan_retries`: Number of retries for initial planning (default: 5)

### KG-Related Parameters
- `--kg_top_k`: Number of entity linking candidates (default: 20)
- `--max_display_facts`: Maximum number of facts to display for an entity (default: 1500)
- `--kg_prune_method`: KG fact processing method, options: `filter` for strict fact filtering and `extract` for text-style fact extraction
- `--kg_verbose_report`: Include KG candidate previews and reasoning in downstream reports (disabled by default to reduce context length)
- `--entity_link_method`: Entity linking method, options: `simple`, `advanced`, `analysis`
- `--entity_link_context`: Entity linking context, options: `question_only`, `query_only`, `question_query`, `instruct`
- `--server_urls`: Wikidata service configuration file (default: `server_urls.txt`)

### Wiki-Related Parameters
- `--wikipedia_method`: Wikipedia retrieval method, options:
  - `text_only`: Text retrieval only
  - `with_tables`: Include table support (default)
  - `full_section`: Full section retrieval
- `--max_page_retrieval_interactions`: Maximum interactions for Wiki page retrieval (default: 6)
- `--section_chunks`: Number of relevant chunks to retrieve per section (default: 3)
- `--wiki_verbose_report`: Include detailed Wikipedia retrieval rationales and traces in downstream reports (disabled by default to reduce context length)
- `--synthesis_method`: Synthesis prompt variant, options: `extract_and_judgment` and `evaluate_and_extract`

### Resume and Rerun Parameters
- `--resume_run_id`: Resume from a previous run ID
- `--run_unfinished`: Run unfinished questions (default: True)
- `--rerun_specific_indices`: Rerun specific question indices
- `--rerun_failed_only`: Only rerun questions that failed due to network issues

### Other Parameters
- `--tag`: Experiment tag for identification
- `--result_dir`: Results output directory (default: `results`)
- `--use_indent`: Enable text indentation for model inputs (default: True)

## Workflow Description

The CoG system employs an iterative multi-hop reasoning workflow:

1. **Initial Planning** (`plan/`): Analyze the question, generate initial query entities and sub-questions
2. **Information Gathering** (per iteration):
   - **KG Exploration** (`kg/`): Explore entity relations and facts in Wikidata
   - **Wiki Retrieval** (`wiki/`): Retrieve relevant text and tables from Wikipedia
3. **Synthesis and Judgment** (`think/`): Evaluate whether collected information is sufficient to answer the question
   - `SUFFICIENT`: Information is sufficient, generate final answer
   - `INSUFFICIENT_USEFUL`: Information is useful but insufficient, plan next step
   - `INSUFFICIENT_USELESS`: Information is useless, reflect and adjust strategy
4. **Answer Generation** (`answer/`): Generate final answer based on collected evidence

## Output Results

After completion, results are saved in the `results/` directory. Experiment folder naming format: `YYMMDD_HHMMSS_dataset_model_embedding__tag`

```
results/
└── 260101_182426_musique_Qwen3-235B-Instruct_bge-m3__Q235Ins/
    ├── metadata.json                  # Experiment metadata (start time, config, duration, etc.)
    ├── results_complete/              # Detailed results folder for each question
    │   ├── 0.json                     # Complete results for question 0
    │   ├── 1.json                     # Complete results for question 1
    │   └── ...
    ├── summary_light_results.json     # Lightweight summary of all questions
    └── summary_statistics.json        # Statistical data (success rate, turn distribution, etc.)
```

### Result File Descriptions

**1. Individual Question Result File** (`results_complete/X.json`)
Complete results for each question, containing:
- `question_id`: Question index
- `question_text`: Question text
- `ground_truth_answer`: Ground truth answer
- `final_answer`: Model-generated answer
- `reasoning_type`: Reasoning exit type, marking the exit reason:
  - `COMPLETED_SUCCESSFULLY` - Sufficient information, successfully generated final answer
  - `COMPLETED_BY_MAX_TURNS` - Reached maximum turns, generated answer via summary module
  - `COMPLETED_AFTER_FAILURE` - Module failure, generated answer via failure analysis module
  - `FALLBACK_COT_AFTER_MAX_TURNS` - Reached maximum turns and summary failed, fallback to direct CoT
  - `FALLBACK_COT_AFTER_FAILURE` - Module failure and analysis failed, fallback to direct CoT
- `reasoning_turns`: Number of reasoning turns used
- `final_notebook`: Final knowledge notebook content
- `full_interaction_history`: Complete interaction history (queries, judgments, planning for each turn)
- `run_stats`: Run metadata such as Wikipedia pages accessed and network failure flags

**2. Lightweight Summary File** (`summary_light_results.json`)
Simplified results list for all questions, used for subsequent evaluation of experiment results (refer to `eval/README.md` in the project root directory for evaluation methods).

**3. Statistical Summary File** (`summary_statistics.json`)
Contains overall statistical data:
- `total_questions_processed`: Total number of questions processed
- `performance_stats`: Reasoning turn statistics
- `reliability_stats`: Reliability statistics (Wiki/LLM API failure rates)
- `reasoning_flow_stats`: Reasoning flow statistics (exit type distribution)
- `wiki_resource_stats`: Wikipedia resource usage statistics

## Citation

If you use this system, please cite the relevant paper.

## License

Please refer to the LICENSE file in the project root directory.
