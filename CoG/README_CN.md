CoG是一个结合知识图谱和维基百科文本的多跳问答框架，通过规划、搜索和思考的迭代过程来回答复杂问题。

## 目录结构

```
CoG/
├── main.py                    # 主程序入口
├── utils.py                   # 数据准备和工具函数
├── utils_log.py               # 实验日志记录
├── utils_run.py               # 运行配置工具
├── server_urls.txt            # Wikidata服务地址配置
├── plan/                      # 初始规划模块
├── wiki/                      # 维基百科检索模块
├── kg/                        # 知识图谱探索模块
├── think/                     # 信息聚合与反思、规划模块
├── answer/                    # 答案生成模块
├── results/                   # 实验结果输出目录
```

## 环境要求

### 1. Python依赖

确保已安装项目根目录的依赖：

```bash
pip install -r requirements.txt
```

### 2. 服务配置

#### LLM服务
系统默认使用本地部署的OpenAI兼容API服务。需要配置：
- **默认地址**: `http://127.0.0.1:9034/v1`
- **默认模型**: `Qwen3-32B`

可通过环境变量或`.env`文件配置API密钥（如使用云端服务）。

#### Wikidata服务
需要运行Wikidata查询服务。服务地址在`server_urls.txt`中配置：

```
http://127.0.0.1:23150
http://127.0.0.1:23151
http://127.0.0.1:23152
...
```

可参考项目根目录的`Wikidata/`文件夹启动服务。

## 快速开始

### 基础运行

在指定数据集上运行实验：

```bash
cd CoGOnGraph/CoG
CUDA_VISIBLE_DEVICES=3 python main.py --dataset hotpot_e
```

### 运行示例

#### 1. 使用自定义模型和参数
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset musique \
    --model Qwen3-32B \
    --base_url http://127.0.0.1:9034/v1 \
    --embedding_model BAAI/bge-m3 \
    --max_turns 5 \
    --tag my_experiment
```

#### 2. 恢复中断的实验
```bash
python main.py \
    --resume_run_id 20251225_143022_hotpot_e \
    --run_unfinished
```

## 主要参数说明

### 数据集参数
- `--dataset`: 选择数据集，支持以下七个数据集：
  - `KGQAGen` - KGQAGen-10k
  - `cwq` - ComplexWebQuestions (CWQ)
  - `qald` - QALD10-en
  - `webqsp` - WebQuestionsSP (WebQSP)
  - `2wiki` - 2WikiMultihopQA
  - `hotpot_e` - AdvHotpotQA
  - `musique` - MusiQue
- `--num_sample`: 限制运行的问题数量
- `--start`: 起始问题索引
- `--run_specific_indices`: 指定运行特定索引的问题，例如：`--run_specific_indices 0 1 2 3`

### 模型参数
- `--model`: 基础LLM模型（默认：`Qwen3-32B`）
- `--base_url`: LLM服务地址（默认：`http://127.0.0.1:9034/v1`）
- `--embedding_model`: 嵌入模型（默认：`BAAI/bge-m3`）
- `--kg_model`: KG模块专用模型（默认与`--model`相同）
- `--kg_base_url`: KG模块服务地址（默认与`--base_url`相同）
- `--wiki_model`: Wiki模块专用模型（默认与`--model`相同）
- `--wiki_base_url`: Wiki模块服务地址（默认与`--base_url`相同）

### 生成参数
- `--max_length`: LLM输出最大长度
- `--max_length_entity_link`: 实体链接模块最大输出长度
- `--max_length_plan`: 规划模块最大输出长度
- `--max_length_relation_discovery`: 关系发现模块最大输出长度
- `--enable_thinking`: 启用Qwen3-32B的思考模式 (默认不启用)

### 工作流参数
- `--max_turns`: 最大迭代轮次（默认：4）
- `--skip_tier1_fallback`: 跳过一级回退策略（more：False）
- `--fact_pruning_retries`: KG事实剪枝重试次数（默认：5）
- `--plan_retries`: 初始规划重试次数（默认：5）

### KG相关参数
- `--kg_top_k`: 实体链接候选数量（默认：20）
- `--max_display_facts`: 实体事实显示最大数量（默认：1200）
- `--entity_link_method`: 实体链接方法，可选：`simple`, `advanced`, `analysis`
- `--entity_link_context`: 实体链接上下文，可选：`question_only`, `query_only`, `question_query`, `instruct`
- `--server_urls`: Wikidata服务配置文件（默认：`server_urls.txt`）

### Wiki相关参数
- `--wikipedia_method`: 维基百科检索方法，可选：
  - `text_only`: 仅文本检索
  - `with_tables`: 包含表格支持（默认）
  - `full_section`: 完整章节检索
- `--max_page_retrieval_interactions`: Wiki页面检索最大交互次数（默认：6）
- `--section_chunks`: 每个章节检索的相关块数量（默认：3）

### 恢复和重跑参数
- `--resume_run_id`: 恢复之前的运行ID
- `--run_unfinished`: 运行未完成的问题（默认：True）
- `--rerun_specific_indices`: 重跑特定索引的问题
- `--rerun_failed_only`: 仅重跑因网络问题失败的问题

### 其他参数
- `--tag`: 实验标签，用于标识实验
- `--result_dir`: 结果输出目录（默认：`results`）
- `--use_indent`: 启用模型输入的文本缩进（默认：True）

## 工作流程说明

CoG系统采用迭代式多跳推理流程：

1. **初始规划** (`plan/`)：分析问题，生成初始查询实体和子问题
2. **信息收集** (每轮迭代)：
   - **KG探索** (`kg/`)：在Wikidata中探索实体关系和事实
   - **Wiki检索** (`wiki/`)：在维基百科中检索相关文本和表格
3. **综合判断** (`think/`)：评估收集的信息是否足以回答问题
   - `SUFFICIENT`: 信息充足，生成最终答案
   - `INSUFFICIENT_USEFUL`: 信息有用但不充分，规划下一步
   - `INSUFFICIENT_USELESS`: 信息无用，反思并调整策略
4. **答案生成** (`answer/`)：基于收集的证据生成最终答案

## 输出结果

运行完成后，结果保存在`results/`目录下，实验文件夹命名格式为：`YYMMDD_HHMMSS_数据集_模型_嵌入模型__标签`

```
results/
└── 260101_182426_musique_Qwen3-235B-Instruct_bge-m3__Q235Ins/
    ├── metadata.json                  # 实验元数据（开始时间、配置、运行时长等）
    ├── results_complete/              # 每个问题的详细结果文件夹
    │   ├── 0.json                     # 问题0的完整结果
    │   ├── 1.json                     # 问题1的完整结果
    │   └── ...
    ├── summary_light_results.json     # 所有问题的轻量级汇总
    └── summary_statistics.json        # 统计数据（成功率、轮次分布等）
```

### 结果文件说明

**1. 单个问题结果文件** (`results_complete/X.json`)
每个问题的完整结果，包含：
- `question_id`: 问题索引
- `question_text`: 问题文本
- `ground_truth_answer`: 标准答案
- `final_answer`: 模型生成的答案
- `reasoning_type`: 推理结束类型，标记退出原因：
  - `COMPLETED_SUCCESSFULLY` - 信息充足，成功生成最终答案
  - `COMPLETED_BY_MAX_TURNS` - 达到最大轮次，通过总结模块生成答案
  - `COMPLETED_AFTER_FAILURE` - 模块失败，通过失败分析模块生成答案
  - `FALLBACK_COT_AFTER_MAX_TURNS` - 达到最大轮次且总结失败，回退到直接CoT
  - `FALLBACK_COT_AFTER_FAILURE` - 模块失败且分析失败，回退到直接CoT
- `reasoning_turns`: 使用的推理轮次
- `final_notebook`: 最终知识笔记本内容
- `full_interaction_history`: 完整的交互历史（每轮的查询、判断、规划等）
- `run_stats`: 运行统计（维基页面访问数、是否存在网络链接失败等）

**2. 轻量级汇总文件** (`summary_light_results.json`)
所有问题的简化结果列表，用于后续评估实验结果（评估方法请参考项目根目录的`eval/README.md`）。

**3. 统计摘要文件** (`summary_statistics.json`)
包含整体统计数据：
- `total_questions_processed`: 处理的问题总数
- `performance_stats`: 性能统计（每个问题的平均轮次、分布等）
- `reliability_stats`: 可靠性统计（维基/LLM API请求失败率）
- `reasoning_flow_stats`: 推理流程统计（结束类型分布）
- `wiki_resource_stats`: 维基资源使用统计


## 引用

我们的论文已被 EMNLP 2026 录用。作者为 Gengxian Zhou、Jian Xu、Zichen Tang、Shiming Xiang、Haihong E 和 Cheng-Lin Liu。Camera-ready 论文和会议论文集元数据公开后，本节将补充官方 BibTeX。

## 许可证

请参考项目根目录的LICENSE文件。
