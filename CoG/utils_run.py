import os
import json

def prepare_question_queue(dataset, logger, resume_run_id, run_unfinished=True, rerun_specific_indices=None, run_specific_indices=None, rerun_failed_only=False, limit=None, start=0):
    """
    根据运行模式准备要处理的问题队列。

    Args:
        dataset (list): 完整的数据集，每个元素是一个字典。
        logger (ExperimentLogger): 已初始化的实验记录器实例。
        resume_run_id (str or None): 要恢复的实验的ID。如果为 None，则为新运行。
        run_unfinished (bool): (恢复模式) 如果为 True，则将所有未完成的问题添加到队列中。
        rerun_specific_indices (list of int, optional): (恢复模式) 要强制重新运行的问题在数据集中的索引列表。
        run_specific_indices (list of int, optional): (新运行模式) 如果提供，则只运行这些指定索引的问题。
        rerun_failed_only (bool): (恢复模式) 如果为 True，则只运行先前失败的问题。
        limit (int, optional): 限制本次运行处理问题的最大数量。
        start (int, optional): The starting index of the sample to run.
    Returns:
        list: 将要运行的(索引, 问题)元组的列表。
    """
    # 确定初始的问题队列
    if not resume_run_id:
        print("Starting a new run.")
        if run_specific_indices:
            print(f"Running on a specific subset of {len(run_specific_indices)} questions.")
            final_indices = sorted([i for i in run_specific_indices if i < len(dataset)])
            initial_queue = [(i, dataset[i]) for i in final_indices]
        else:
            print("Preparing all questions from the dataset.")
            initial_queue = list(enumerate(dataset))
    else:
        # --- 恢复运行的逻辑 ---
        print(f"Resuming run '{resume_run_id}'.")
        
        # 如果是只重跑失败模式，则优先处理
        if rerun_failed_only:
            print("Mode: Rerun failed questions caused by network issues only.")
            stats_path = os.path.join(logger.run_dir, "summary_statistics.json")
            if not os.path.exists(stats_path):
                print(f"Warning: summary_statistics.json not found in {logger.run_dir}. Cannot determine failed questions. No questions will be run.")
                initial_queue = []
            else:
                try:
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    
                    reliability_stats = stats.get("reliability_stats", {})
                    
                    # Get wiki failure IDs
                    wiki_failed_ids = set(reliability_stats.get("wiki_failures", {}).get("failed_question_ids", []))
                    
                    # Get LLM timeout failure IDs
                    timeout_error_key = "APITimeoutError('Request timed out.')"
                    llm_failures = reliability_stats.get("llm_failures", {}).get("error_details", {})
                    timeout_failed_ids = set()
                    if timeout_error_key in llm_failures:
                        timeout_failed_ids = set(llm_failures[timeout_error_key].get("question_ids", []))
                    
                    # Combine them
                    failed_ids = sorted(list(wiki_failed_ids.union(timeout_failed_ids)))
                    
                    if not failed_ids:
                        print("No failed questions (Wiki or LLM Timeout) found in the previous run's summary.")
                        initial_queue = []
                    else:
                        print(f"Found {len(failed_ids)} failed questions to rerun.")
                        final_indices = sorted([i for i in failed_ids if i < len(dataset)])
                        initial_queue = [(i, dataset[i]) for i in final_indices]
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading or parsing summary_statistics.json: {e}. Cannot determine failed questions.")
        else:
            results_dir = os.path.join(logger.run_dir, "results_complete")
            completed_ids = set()
            if os.path.exists(results_dir):
                # 从文件名解析已完成的索引
                for filename in os.listdir(results_dir):
                    if filename.endswith('.json'):
                        try:
                            # 假设文件名为 "42.json"，我们提取 42
                            completed_index = int(filename.split('.')[0])
                            completed_ids.add(completed_index)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse index from filename '{filename}'. Skipping.")
                print(f"Found {len(completed_ids)} completed questions.")

            indices_to_run = set()

            # 1. 如果指定，添加要强制重跑的问题索引
            if rerun_specific_indices:
                indices_to_run.update(rerun_specific_indices)
                print(f"Added {len(rerun_specific_indices)} specific question indices to the queue for re-running.")

            # 2. 如果开启，添加所有未完成的问题索引
            if run_unfinished:
                print("Searching for unfinished questions to add to the queue.")
                if limit is not None and limit >= 0:
                    search_end = min(len(dataset), start + limit)
                    candidate_indices = range(start, search_end)
                    print(f"Restricting unfinished search to chunk range [{start}, {search_end}).")
                else:
                    candidate_indices = range(start, len(dataset))
                    if start > 0:
                        print(f"Restricting unfinished search to indices >= {start}.")
                unfinished_count = 0
                for i in candidate_indices:
                    if i not in completed_ids:
                        indices_to_run.add(i)
                        unfinished_count += 1
                print(f"Found and added {unfinished_count} unfinished questions.")
            
            if not indices_to_run:
                print("No questions to run based on the current configuration.")
                initial_queue = []
            else:
                # 3. 根据最终的索引集合，生成问题列表，并按索引排序
                final_indices = sorted(list(indices_to_run))
                initial_queue = [(i, dataset[i]) for i in final_indices if i < len(dataset)]

                # 过滤掉不存在的索引
                if len(initial_queue) != len(final_indices):
                    print("Warning: Some specified indices were out of bounds for the dataset and have been ignored.")

    # Apply start index filter
    if start > 0:
        # We assume initial_queue is sorted by index
        original_size = len(initial_queue)
        initial_queue = [(i, item) for i, item in initial_queue if i >= start]
        print(f"Applying start index: Starting from question index {start}. Removed {original_size - len(initial_queue)} questions.")

    # 最后，应用 limit 参数
    if limit is not None and limit >= 0:
        print(f"Applying limit: The run will be limited to the first {limit} questions in the queue.")
        return initial_queue[:limit]
    
    return initial_queue
