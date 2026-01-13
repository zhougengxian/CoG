import re
import ast
import time
from string import Template
from utils import run_llm, generate_process

from plan.prompt import init_plan_prompt, filter_sequential_prompt


def extract_plan_result(result_text):
    try:
        analysis = re.search(r"Analysis:(.*?)\nQuery:", result_text, re.DOTALL).group(1).strip()
        queries_str = re.search(r"\nQuery:(.*?)\nEntities:", result_text, re.DOTALL).group(1).strip()
        entities_str = re.search(r"\nEntities:(.*)", result_text, re.DOTALL).group(1).strip()
        
        queries = ast.literal_eval(queries_str)
        entities = ast.literal_eval(entities_str)
        
        if len(queries) != len(entities):
            print(f"Length of queries and entities do not match. queries: {len(queries)}, entities: {len(entities)}")
            return None
        
        return {
            "analysis": analysis,
            "queries": queries,
            "entities": entities
        }
    except Exception as e:
        print(f"Could not parse the result string: {result_text}\nerror: {e}")
        return None
    

def generate_init_plan(question, args, max_retries=3):
    """
    Represents one step of generating and parsing a plan from the LLM, with retries.
    """
    for attempt in range(max_retries):
        print(f"--- Generating Plan for Question (Attempt {attempt + 1}/{max_retries}) ---\n")
        prompt = init_plan_prompt.substitute(question=question)
        result_text = run_llm(prompt, args)
        if not result_text:
            print(f"Attempt {attempt + 1} failed: LLM call returned no result. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        result = extract_plan_result(result_text)
        if result:
            analysis, queries, entities = result
            print("Analysis:", analysis)
            print("Query:", queries)
            print("Entities:", entities)
            return True, analysis, queries, entities
        else:
            print(f"Attempt {attempt + 1} failed: Could not parse LLM output. Generated Result:")
            print(result_text)
            time.sleep(5)
            
    print(f"Failed to generate and parse a valid plan after {max_retries} attempts.")
    return False, None, None, None


def run_init_planning(question, args, max_retries=3):
    """
    执行一个完整的计划生成和精炼工作流。

    该工作流包括两个主要步骤：
    1. 生成初步计划：根据用户问题创建一个初始的、可能包含序贯步骤的计划。
    2. 精炼计划：通过修正幻觉并移除依赖步骤，从初步计划中提取可以立即执行的、并行的第一步查询。
    """
    print('=' * 25 + " Question " + '=' * 25 + "\n" + f"{question}")
    
    # --- 步骤 1: 生成初步计划 ---
    initial_plan_result = generate_process(
        step_name="Generate Initial Plan",
        prompt_template=init_plan_prompt,
        template_inputs={'question': question},
        parsing_function=extract_plan_result,
        args=args,
        module='main',
        max_retries=max_retries,
        max_tokens=args.max_length_plan
    )
    
    if not initial_plan_result:
        reason = "LLM failed to generate a valid init plan after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    # 从返回的字典中解包数据
    initial_analysis = initial_plan_result['analysis']
    initial_queries = initial_plan_result['queries']
    initial_entities = initial_plan_result['entities']

    print("\n--- Initial Plan Generated ---")
    print(f"Analysis: {initial_analysis}")
    print(f"Query: {initial_queries}")
    print(f"Entities: {initial_entities}\n")

    # --- 步骤 2: 精炼计划 ---
    # 这一步旨在解决初步计划中可能存在的幻觉问题，并过滤掉当前无法执行的依赖步骤。
    # 我们只保留那些基于原始问题可立即执行的并行查询。
    refine_template_inputs = {
        'question': question,
        'analysis': initial_analysis,
        'query_list': str(initial_queries),
        'entity_list': str(initial_entities)
    }

    final_plan_result = generate_process(
        step_name="Refine Plan",
        prompt_template=filter_sequential_prompt,
        template_inputs=refine_template_inputs,
        parsing_function=extract_plan_result,  # 复用同一个解析函数
        args=args,
        module='main',
        max_retries=max_retries,
        max_tokens=args.max_length_plan
    )

    if not final_plan_result:
        reason = "LLM failed to refine the initial plan after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
        
    print("\n--- Final Refined Plan ---")
    print(f"Analysis: {final_plan_result['analysis']}")
    print(f"Query: {final_plan_result['queries']}")
    print(f"Entities: {final_plan_result['entities']}\n")
    
    return 'SUCCESS', final_plan_result