import concurrent.futures
from functools import partial

from wiki.workflow_text import retrieve_info_wikipedia
from wiki.format import format_wikipedia_retrieval


def run_parallel_retrieval(question: str, analysis: str, queries: list, entities: list, args, embeddings):
    """
    使用线程池并行执行所有查询的维基百科检索任务，并返回格式化后的结果。

    Args:
        question (str): 原始问题.
        analysis (str): LLM 生成的分析计划.
        queries (list): 要执行的查询字符串列表.
        entities (list): 与查询对应的实体列表.
        args: 包含模型配置的参数.
        embeddings: 用于 RAG 的嵌入模型.

    Returns:
        list[str]: 一个包含所有查询检索结果的、格式化后的长字符串列表。
    """
    print(f"--- Processing {len(queries)} queries in parallel ---")

    # 使用 functools.partial 来预先填充在所有调用中都相同的参数
    # 'query' 和 'entity' 将由 executor.map 动态提供
    retrieval_task = partial(retrieve_info_wikipedia, 
                             question=question, 
                             analysis=analysis, 
                             args=args, 
                             embeddings=embeddings)

    raw_results = []
    # 创建线程池来并行执行任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # executor.map 会将 queries 和 entities 的元素一一对应地作为参数传给 retrieval_task
        # 它会按顺序返回所有任务的结果
        # 使用 list() 来确保所有任务完成后再继续
        raw_results = list(executor.map(retrieval_task, queries, entities))

    print("--- WikiPedia Retrieval Finish ---")
    # 格式化所有收集到的结果
    all_formatted_results = []
    for i, result in enumerate(raw_results):
        formatted_output = format_wikipedia_retrieval(
            query=queries[i], 
            entity=entities[i], 
            result=result)
        all_formatted_results.append(formatted_output)
        
    return all_formatted_results