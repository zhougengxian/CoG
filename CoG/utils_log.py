import json
import os
import subprocess
from datetime import datetime, timezone
import collections
import psutil

CST = timezone.utc


def get_process_start_time_iso(pid):
    """获取指定PID进程的启动时间，并返回ISO 8601格式的UTC字符串。"""
    try:
        process = psutil.Process(pid)
        start_time_utc = datetime.fromtimestamp(process.create_time(), tz=timezone.utc)
        return start_time_utc.isoformat()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # 如果进程不存在或无权访问，返回None
        return None


class ExperimentLogger:
    """
    管理实验结果的结构化日志记录。

    为每次实验运行创建一个唯一的目录。在运行期间，每个问题的详细结果
    都会被保存为一个单独的 JSON 文件。运行结束后，`finalize` 方法会生成
    一个轻量级的结果摘要和一个统计报告。
    """
    def __init__(self, base_dir="results", experiment_name="default_run",
                 model_config=None, dataset_name="unknown", description="", resume_from_id=None, tag=None):
        """
        初始化记录器。可以创建一个新实验或从现有实验恢复。

        Args:
            base_dir (str): 存储所有实验结果的根目录。
            experiment_name (str): 实验的描述性名称。
            model_config (dict): 包含模型配置的字典。
            dataset_name (str): 正在使用的数据集的名称。
            description (str): 对实验的人类可读描述。
            resume_from_id (str, optional): 如果提供，则恢复记录到具有此 ID 的现有
                                          实验目录。如果为 None，则创建新实验。
            tag (str, optional): 一个自定义标签，会附加到实验 ID 的末尾。
        """
        self.base_dir = base_dir
        if resume_from_id and self._try_resume(resume_from_id):
            return
        self._create_new_experiment(experiment_name, model_config, dataset_name, description, tag)

    def _try_resume(self, experiment_id):
        """尝试从给定的 experiment_id 恢复。"""
        run_dir_to_check = os.path.join(self.base_dir, experiment_id)
        metadata_path_to_check = os.path.join(run_dir_to_check, "metadata.json")

        if os.path.isdir(run_dir_to_check) and os.path.isfile(metadata_path_to_check):
            self.experiment_id = experiment_id
            self.run_dir = run_dir_to_check
            self.metadata_path = metadata_path_to_check
            self.results_dir = os.path.join(self.run_dir, "results_complete")
            os.makedirs(self.results_dir, exist_ok=True)
            
            with open(self.metadata_path, 'r+', encoding='utf-8') as f:
                metadata = json.load(f)
                self.start_time = datetime.fromisoformat(metadata['start_time_utc'])
                
                # 为恢复的进程更新 PID，并重置状态为 'running'
                metadata['pid'] = os.getpid()
                metadata['status'] = 'running'
                # 关键：更新“上次启动时间”和“进程启动时间”
                metadata['last_start_time_utc'] = datetime.now(CST).isoformat()
                metadata['process_start_time_utc'] = get_process_start_time_iso(os.getpid())
                f.seek(0)
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                f.truncate()
            
            print(f"成功恢复实验 '{self.experiment_id}' 于: {self.run_dir}")
            return True
        else:
            print(f"警告: 在 '{self.base_dir}' 中未找到实验 ID '{experiment_id}' 或其元数据文件。")
            print("将创建一个新实验。")
            return False

    def _create_new_experiment(self, experiment_name, model_config, dataset_name, description, tag):
        """为一次新运行创建目录和元数据。"""
        self.start_time = datetime.now(CST)
        timestamp = self.start_time.strftime('%y%m%d_%H%M%S')
        
        base_experiment_id = f"{timestamp}_{experiment_name}"
        if tag:
            # 清理 tag，使其适合作为目录名
            safe_tag = "".join(c if c.isalnum() else "_" for c in tag)
            self.experiment_id = f"{base_experiment_id}__{safe_tag}"
        else:
            self.experiment_id = base_experiment_id

        self.run_dir = os.path.join(self.base_dir, self.experiment_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.metadata_path = os.path.join(self.run_dir, "metadata.json")
        self.results_dir = os.path.join(self.run_dir, "results_complete")
        os.makedirs(self.results_dir, exist_ok=True)

        self._write_metadata(model_config, dataset_name, description)
        print(f"记录新实验 '{self.experiment_id}' 到: {self.run_dir}")

    def _get_git_commit_hash(self):
        """检索当前的 git 提交哈希以保证可复现性。"""
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            return commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "git_not_found"

    def _write_metadata(self, model_config, dataset_name, description):
        """收集并写入共享的实验元数据到 metadata.json。"""
        pid = os.getpid()
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_description": description,
            "start_time_utc": self.start_time.isoformat(),
            "last_start_time_utc": self.start_time.isoformat(), # 首次运行时与 start_time 相同
            "pid": pid,
            "process_start_time_utc": get_process_start_time_iso(pid), # 记录进程启动时间
            "status": "running",
            "dataset_name": dataset_name,
            "code_version": self._get_git_commit_hash(),
            "model_config": model_config or {}
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def log_result(self, question_id, question_text, ground_truth_answer,
                   structured_output, full_interaction_history, 
                   final_notebook, wiki_failure_flag, failure_details, llm_errors=None, accessed_wiki_pages=None):
        """
        将单个问题的完整结果保存到一个独立的 JSON 文件中。
        如果文件已存在，则会覆盖它。
        """
        end_time = datetime.now(CST)
        safe_filename = f"{str(question_id).replace('/', '_')}.json"

        complete_entry = {
            "question_id": question_id,
            "question_text": question_text,
            "ground_truth_answer": ground_truth_answer,
            "final_notebook": final_notebook,
            "llm_errors": llm_errors or [], # Save the list of LLM errors
            "run_stats": {
                "end_time_utc": end_time.isoformat(),
                "wiki_failure": wiki_failure_flag,
                "accessed_wiki_pages": accessed_wiki_pages or [],
                "num_accessed_wiki_pages": len(accessed_wiki_pages) if accessed_wiki_pages else 0
            },
            "structured_output": structured_output,
            "full_interaction_history": full_interaction_history,
            "failure_details": failure_details
        }
        
        result_path = os.path.join(self.results_dir, safe_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(complete_entry, f, indent=4, ensure_ascii=False)
            
    def mark_rerun_start(self):
        """Updates metadata to reflect that a re-run phase is starting."""
        print("Updating experiment status to 'running' for the re-run phase...")
        try:
            with open(self.metadata_path, 'r+', encoding='utf-8') as f:
                metadata = json.load(f)
                
                metadata['status'] = 'running'
                metadata['pid'] = os.getpid()
                metadata['last_start_time_utc'] = datetime.now(CST).isoformat()
                metadata['process_start_time_utc'] = get_process_start_time_iso(os.getpid())
                
                f.seek(0)
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                f.truncate()
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not update metadata for re-run: {e}")
            
    def finalize(self):
        """
        完成实验，生成摘要文件，并用结束时间和总时长更新元数据。
        """
        print("实验正在完成... 正在生成摘要文件...")
        
        # 1. 初始化用于聚合的数据结构
        light_results = []
        reasoning_type_counter = collections.Counter()
        
        turns = []
        wiki_failure_ids = []
        total_wiki_pages_accessed_list = []
        llm_error_summary = collections.defaultdict(lambda: {'count': 0, 'question_ids': set()})
        
        # 2. 遍历所有结果文件进行聚合
        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        for filename in result_files:
            file_path = os.path.join(self.results_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                structured_output = data.get("structured_output", {})
                run_stats = data.get("run_stats", {})
                interaction_history = data.get("full_interaction_history", [])
                
                # a. 构建 reasoning_path 和 stopping_module
                path = ["init_plan"]
                stopping_module = "Unknown"
                reasoning_type = structured_output.get("reasoning_type", "UNKNOWN")

                for turn in interaction_history:
                    judgment = turn.get("judgment")
                    if judgment == "INSUFFICIENT_USEFUL":
                        path.append("plan")
                    elif judgment == "INSUFFICIENT_USELESS":
                        path.append("recover")
                    elif judgment == "SUFFICIENT":
                        break # 路径在成功前结束

                if reasoning_type == "COMPLETED_SUCCESSFULLY":
                    path.append("success")
                    stopping_module = "Final Answer Generation"
                elif "MAX_TURNS" in reasoning_type:
                    path.append("max_turns")
                    stopping_module = "Max Turns Reached"
                elif "FAILURE" in reasoning_type:
                    path.append("fail")
                    # 直接从记录的 failure_details 中获取模块
                    failure_info = data.get("failure_details", {})
                    stopping_module = failure_info.get("module", "Unknown Module Failure")

                # b. 为 light_results 提取数据
                simple_entry = {
                    "id": data.get("question_id"),
                    "question": data.get("question_text"),
                    "ground_truth": data.get("ground_truth_answer"),
                    "answer": structured_output.get("final_answer"),
                    "reasoning_type": reasoning_type,
                    "stopping_module": stopping_module,
                    "reasoning_path": "->".join(path),
                    "reasoning_explanation": structured_output.get("reasoning_explanation"),
                    "turns": structured_output.get("reasoning_turns"),
                    "wiki_failure": run_stats.get("wiki_failure")
                }
                light_results.append(simple_entry)
                
                # c. 聚合统计数据
                if reasoning_type != "UNKNOWN":
                    reasoning_type_counter.update([reasoning_type])
                if structured_output.get("reasoning_turns") is not None:
                    turns.append(structured_output["reasoning_turns"])
                if run_stats.get("wiki_failure"):
                    wiki_failure_ids.append(data.get("question_id"))
                
                if run_stats.get("num_accessed_wiki_pages") is not None:
                    total_wiki_pages_accessed_list.append(run_stats["num_accessed_wiki_pages"])
                
                # d. Aggregate LLM errors
                if "llm_errors" in data and data["llm_errors"]:
                    question_id = data.get("question_id")
                    for error_str in data["llm_errors"]:
                        llm_error_summary[error_str]['count'] += 1
                        llm_error_summary[error_str]['question_ids'].add(question_id)

            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 无法读取或解析文件 {filename}: {e}")

        # 3. 写入轻量级结果文件为格式化的 JSON 数组
        light_results_path = os.path.join(self.run_dir, "summary_light_results.json")
        with open(light_results_path, 'w', encoding='utf-8') as f:
            json.dump(light_results, f, indent=4, ensure_ascii=False)
        print(f"轻量级结果已保存到: {light_results_path}")

        # 4. 计算并写入统计数据文件
        stats_path = os.path.join(self.run_dir, "summary_statistics.json")
        total_questions = len(light_results)
        
        # Helper to avoid division by zero
        def get_dist(counter):
            total = sum(counter.values())
            return {k: f"{(v / total * 100):.2f}%" for k, v in counter.items()} if total > 0 else {}

        statistics = {
            "total_questions_processed": total_questions,
            "performance_stats": {
                "turns_per_question": {
                    "average": f"{(sum(turns) / total_questions):.2f}" if total_questions > 0 else 0,
                    "min": min(turns) if turns else 0,
                    "max": max(turns) if turns else 0,
                    "distribution": dict(collections.Counter(turns))
                }
            },
            "reliability_stats": {
                "wiki_failures": {
                    "count": len(wiki_failure_ids),
                    "rate": f"{(len(wiki_failure_ids) / total_questions * 100):.2f}%" if total_questions > 0 else "0.00%",
                    "failed_question_ids": sorted(wiki_failure_ids) if wiki_failure_ids else []
                },
                "llm_failures": {
                    "total_errors": sum(info['count'] for info in llm_error_summary.values()),
                    "unique_error_types": len(llm_error_summary),
                    "error_details": {
                        error: {
                            "count": info['count'],
                            "question_ids": sorted(list(info['question_ids']))
                        } for error, info in llm_error_summary.items()
                    }
                }
            },
            "reasoning_flow_stats": {
                "exit_reason_counts": dict(reasoning_type_counter),
                "exit_reason_distribution": get_dist(reasoning_type_counter)
            },
            "wiki_resource_stats": {
                "accessed_documents": {
                    "total": sum(total_wiki_pages_accessed_list),
                    "average": f"{(sum(total_wiki_pages_accessed_list) / total_questions):.2f}" if total_questions > 0 else 0
                }
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=4, ensure_ascii=False)
        print(f"统计数据已保存到: {stats_path}")

        # 5. 更新元数据
        end_time = datetime.now(CST)
        
        with open(self.metadata_path, 'r+', encoding='utf-8') as f:
            metadata = json.load(f)
            metadata['end_time_utc'] = end_time.isoformat()
            metadata['status'] = 'completed'
            f.seek(0)
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            f.truncate()
            
        print(f"实验完成。")
        
        # 6. Combine wiki failures and LLM timeout failures for rerun list
        timeout_error_key = "APITimeoutError('Request timed out.')"
        rerun_ids = set(wiki_failure_ids)
        if timeout_error_key in llm_error_summary:
            timeout_ids = llm_error_summary[timeout_error_key]['question_ids']
            rerun_ids.update(timeout_ids)
            
        return sorted(list(rerun_ids))