import json
import copy
import argparse
import time
import collections
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from utils import prepare_dataset, E5InstructEmbeddings
from plan.workflow import run_init_planning
from wiki.workflow_text import retrieve_info_wikipedia
from wiki.workflow_table import retrieve_info_wikipedia_with_tables
from wiki.workflow_fullsection import retrieve_info_wikipedia_full_section
from wiki.format import format_wikipedia_retrieval
from kg.client import MultiServerWikidataQueryClient
from kg.workflow import run_full_kg_exploration
from think.synthesis import run_synthesis_and_judgment
from think.plan import run_next_step_planning
from think.recover import run_recovery_planning
from answer.complete import run_answer_generation
from answer.max_turn import run_max_turns_conclusion
from answer.fail import run_failure_conclusion
from answer.fallback import run_direct_cot_answer
from utils_run import prepare_question_queue
from utils_log import ExperimentLogger

load_dotenv()

def prepare_model_interaction_history(interaction_history):

    if not interaction_history:
        return []

    history_for_model = copy.deepcopy(interaction_history)
    for entry in history_for_model:
        entry.pop('evidence_blocks', None)
    return history_for_model


def run_full_workflow(question, args, client, embeddings):
    """
    执行完整的问答工作流，从初始规划到最终答案生成。

    参数:
        question (str): 用户提出的问题。
        args (argparse.Namespace): 包含所有配置参数的对象。
        client (MultiServerWikidataQueryClient): Wikidata 查询客户端实例。
        embeddings (HuggingFaceEmbeddings): 嵌入模型实例。

    返回:
        tuple: 包含以下元素的元组:
            - structured_output (dict): 包含最终答案和推理过程的结构化字典。
            - interaction_history (list): 包含每个回合详细信息的列表。
            - failure_details (dict): 如果工作流失败，包含失败模块和原因的字典。
            - wiki_failure_flag (bool): 指示维基百科 API 是否曾调用失败。
            - workflow_status (str): 工作流的最终状态。
    """
    MAX_TURNS = args.max_turns
    skip_tier1_fallback = args.skip_tier1_fallback
    # Initialize data structures for the interaction loop
    notebook = ""
    candidate_entities_pool = []
    interaction_history = []
    failure_details = {} # To store details of a workflow failure
    wiki_failure_flag = False # Flag to track Wikipedia API failures
    accessed_wiki_pages = set() # Track unique Wikipedia pages accessed
    # It starts as IN_PROGRESS and can transition to FAILED or COMPLETED_SUFFICIENT.
    workflow_status = "IN_PROGRESS"
    
    # Create a temporary set on the args object to collect unique LLM errors
    llm_errors_this_question = set()
    args.current_llm_errors = llm_errors_this_question

    # --- Stage 0: Initial Planning ---
    print("--- Running Initial Planning ---")
    plan_status, planning_result = run_init_planning(question, args, args.plan_retries)

    if plan_status != "SUCCESS":
        workflow_status = "FAILED"
        failure_details = {"module": "Initial Planning", "reason": planning_result.get("reason", "Unknown reason")}
        print("!!! [WORKFLOW FAILED] Initial planning failed. Aborting workflow. !!!")
    else:
        # Planning was successful, setup for the iterative loop
        current_queries = planning_result['queries']
        current_entities = planning_result['entities']
        current_analysis = planning_result['analysis']

        for turn in range(MAX_TURNS):
            print(f"--- Turn {turn+1} ---")
            
            # --- Information Gathering ---
            # Collect results from all sub-queries for this turn
            turn_results = []
            for query, entity in zip(current_queries, current_entities):
                print(f"\nGathering evidence for query: '{query}'")
                # 1. KG Exploration
                kg_report = run_full_kg_exploration(
                    question=question,
                    analysis=current_analysis,
                    query=query,
                    entity_mention=entity,
                    client=client,
                    embeddings=embeddings,
                    args=args
                )
                
                # 2. Wikipedia Retrieval
                if args.wikipedia_method == "text_only":
                    retrieval_result = retrieve_info_wikipedia(query, entity, question, current_analysis, args, embeddings)
                elif args.wikipedia_method == "with_tables":
                    retrieval_result = retrieve_info_wikipedia_with_tables(query, entity, question, current_analysis, args, embeddings)
                elif args.wikipedia_method == "full_section":
                    retrieval_result = retrieve_info_wikipedia_full_section(query, entity, question, current_analysis, args)
                else:
                    # default to with_tables version
                    retrieval_result = retrieve_info_wikipedia_with_tables(query, entity, question, current_analysis, args, embeddings)
                if retrieval_result.get("status") == "WIKI_API_ERROR" or retrieval_result.get("partial_wiki_error") is True:
                    wiki_failure_flag = True
                
                # Count accessed pages
                if retrieval_result.get("page_title"):
                    accessed_wiki_pages.add(retrieval_result.get("page_title"))
                
                formatted_wiki_result = format_wikipedia_retrieval(query, entity, retrieval_result, args, verbose=getattr(args, 'wiki_verbose_report', True))
                
                turn_results.append({
                    "query": query,
                    "entity": entity,
                    "kg_result": kg_report,
                    "wiki_result": formatted_wiki_result
                })
            
            # --- Stage 1: Information Consolidation & Judgment ---
            synth_status, synthesis_result = run_synthesis_and_judgment(
                question=question,
                notebook=notebook,
                analysis=current_analysis,
                current_queries=current_queries,
                current_entities=current_entities,
                turn_results=turn_results,
                args=args
            )
            
            if synth_status != "SUCCESS":
                workflow_status = "FAILED"
                failure_details = {"module": "Synthesis and Judgment", "reason": synthesis_result.get("reason", "Unknown reason")}
                print("Synthesis and judgment failed for this turn. Aborting.")
                break

            judgment = synthesis_result['judgment']
            extracted_content = synthesis_result['extracted_content']
            thought_process = synthesis_result['thought_process']

            # Append current turn to history BEFORE making a decision for the next turn
            interaction_history.append({
                'turn': turn + 1,
                'analysis': current_analysis,
                'queries': current_queries,
                'entities': current_entities,
                'judgment': judgment,
                'extracted_content': extracted_content,
                'synthesis_thought_process': thought_process,
                'evidence_blocks': synthesis_result['evidence_blocks']
            })
            
            # --- Stage 2: Planning & Action ---
            if judgment == "SUFFICIENT":
                print("\n[CONCLUSION] Sufficient information found. Proceeding to generate final answer.")
                # Generate the final answer
                answer_status, final_answer_result = run_answer_generation(
                    question=question,
                    analysis=current_analysis,
                    notebook=notebook,
                    queries=current_queries,
                    entities=current_entities,
                    extracted_content=extracted_content,
                    thought_process=thought_process,
                    args=args
                )

                if answer_status == "SUCCESS":
                    workflow_status = "COMPLETED_SUFFICIENT"
                    final_result = final_answer_result
                    interaction_history[-1]['final_thought_process'] = final_answer_result['final_thought_process']
                    interaction_history[-1]['final_answer'] = final_answer_result['final_answer']
                else:
                    workflow_status = "FAILED"
                    failure_details = {"module": "Final Answer Generation", "reason": final_answer_result.get("reason", "Unknown reason")}
                    print("Final answer generation failed. Aborting workflow.")
                break # Exit the main loop

            elif judgment == "INSUFFICIENT_USEFUL":
                print("\n[PLAN] Information is useful but insufficient. Planning next step...")
                plan_status, planning_result = run_next_step_planning(
                    question=question, notebook=notebook, analysis=current_analysis,
                    queries=current_queries, entities=current_entities,
                    extracted_content=extracted_content, thought_process=thought_process,
                    candidate_entities_pool=candidate_entities_pool, args=args
                )
                if plan_status == "SUCCESS":
                    interaction_history[-1]['next_step_planning_rationale'] = planning_result['thought_process']
                    # Update all states for the next iteration
                    notebook = planning_result['updated_notebook']
                    current_analysis = planning_result['updated_analysis']
                    candidate_entities_pool = planning_result['updated_candidate_pool']
                    current_queries = planning_result['next_queries']
                    current_entities = planning_result['next_entities']
                else:
                    workflow_status = "FAILED"
                    failure_details = {"module": "Next Step Planning", "reason": planning_result.get("reason", "Unknown reason")}
                    print("Next step planning failed. Aborting workflow.")
                    break # Exit loop if planning fails

            elif judgment == "INSUFFICIENT_USELESS":
                print("\n[PLAN] Information is useless. Reflecting and recovering...")
                plan_status, planning_result = run_recovery_planning(
                    question=question, notebook=notebook, analysis=current_analysis,
                    queries=current_queries, entities=current_entities,
                    extracted_content=extracted_content, thought_process=thought_process,
                    interaction_history=prepare_model_interaction_history(interaction_history),
                    candidate_entities_pool=candidate_entities_pool, args=args
                )
                if plan_status == "SUCCESS":
                    interaction_history[-1]['next_step_planning_rationale'] = planning_result['thought_process']
                    # Update state for next turn. Notebook is NOT updated.
                    current_analysis = planning_result['updated_analysis']
                    candidate_entities_pool = planning_result['updated_candidate_pool']
                    current_queries = planning_result['next_queries']
                    current_entities = planning_result['next_entities']
                else:
                    workflow_status = "FAILED"
                    failure_details = {"module": "Recovery Planning", "reason": planning_result.get("reason", "Unknown reason")}
                    print("Recovery planning failed. Aborting workflow.")
                    break # Exit loop if recovery fails

    # --- Final Workflow Conclusion ---
    print("\n" + "="*20 + " Workflow Finished " + "="*20)
    structured_output = {
        "final_answer": None,
        "reasoning_type": None,
        "reasoning_explanation": None,
        "reasoning_turns": len(interaction_history),
        "wiki_failure": wiki_failure_flag  # Add the flag to the output
    }

    if workflow_status == "COMPLETED_SUFFICIENT":
        print("[Status] Success: A definitive answer was generated.")
        structured_output["reasoning_type"] = "COMPLETED_SUCCESSFULLY"
        structured_output["final_answer"] = final_result.get('final_answer', "Error: Answer not found in final result.")
        structured_output["reasoning_explanation"] = final_result.get('final_thought_process', "Error: Thought process not found in final result.")
        
    else:
        # This block handles non-ideal exits: reaching max turns or a module failure.
        # It first attempts a sophisticated conclusion, then falls back to a simple one.
        conclusion_func = None
        conclusion_args = {}

        if workflow_status == "IN_PROGRESS": # Loop completed without reaching SUFFICIENT
            print(f"[Status] Reached MAX_TURNS ({MAX_TURNS}) without a definitive answer.")
            conclusion_func = run_max_turns_conclusion
            conclusion_args = {
                "question": question, "notebook": notebook,
                "interaction_history": prepare_model_interaction_history(interaction_history), "max_turns": MAX_TURNS, "args": args
            }
            print("\n[Status] Failed: Could not generate a final conclusion.")

        elif workflow_status == "FAILED":
            print(f"[Status] Failed: The workflow was terminated due to an internal error in module: {failure_details.get('module', 'Unknown')}.")
            print("Attempting to generate an intelligent failure analysis report...")
            conclusion_func = run_failure_conclusion
            conclusion_args = {
                "question": question, "notebook": notebook,
                "interaction_history": prepare_model_interaction_history(interaction_history), "failure_details": failure_details, "args": args
            }
            print("\n[Status] Critical Failure: The analysis report could not be generated.")

        # Tier 1: Attempt the more sophisticated conclusion function.
        # This can be skipped by setting skip_tier1_fallback=True.
        if not skip_tier1_fallback and conclusion_func:
            conclusion_status, conclusion_result = conclusion_func(**conclusion_args)
        else:
            # Force failure to trigger Tier 2 fallback.
            if skip_tier1_fallback:
                print("[INFO] Skipping Tier 1 conclusion as requested, proceeding to Tier 2 fallback.")
            conclusion_status = "FAILED"
            conclusion_result = {}
        
        if conclusion_status == "SUCCESS":
            structured_output["final_answer"] = conclusion_result.get('final_answer', "Error: Answer not found in conclusion.")
            if workflow_status == "IN_PROGRESS":
                structured_output["reasoning_type"] = "COMPLETED_BY_MAX_TURNS"
                structured_output["reasoning_explanation"] = conclusion_result.get('analysis_of_incompletion', "Analysis of incompletion not available.")
            elif workflow_status == "FAILED":
                structured_output["reasoning_type"] = "COMPLETED_AFTER_FAILURE"
                structured_output["reasoning_explanation"] = conclusion_result.get('root_cause_analysis', "Root cause analysis not available.")
        else:
            # Tier 2: Fallback to a simple direct answer generation
            print("Falling back to direct generation using the model's internal knowledge...")
            if workflow_status == "IN_PROGRESS":
                print("\n[Status] Failed: Could not generate a final conclusion.")
                structured_output["reasoning_type"] = "FALLBACK_COT_AFTER_MAX_TURNS"
                structured_output["reasoning_explanation"] = f"Reached max turns ({MAX_TURNS}) and the final summarization failed. This answer is a direct fallback."
            elif workflow_status == "FAILED":
                print("\n[Status] Critical Failure: The analysis report could not be generated.")
                structured_output["reasoning_type"] = "FALLBACK_COT_AFTER_FAILURE"
                structured_output["reasoning_explanation"] = f"Workflow failed at module '{failure_details.get('module', 'Unknown')}' and the recovery process also failed. This answer is a direct fallback."

            structured_output["final_answer"] = run_direct_cot_answer(question, args)

    print("\n--- Structured Output ---")
    print(json.dumps(structured_output, indent=4, ensure_ascii=False))

    # Clean up the temporary collectors from the args object
    if hasattr(args, 'current_llm_errors'):
        del args.current_llm_errors

    return structured_output, interaction_history, failure_details, wiki_failure_flag, notebook, workflow_status, list(llm_errors_this_question), list(accessed_wiki_pages)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str,
                        default="hotpot_e", help="choose the dataset.")
    parser.add_argument("--base_url", type=str,
                        default='http://127.0.0.1:9040/v1',
                        help="if the LLM_type is qwen, you need add your own openai api keys.")
    # ['Qwen3-32B', 'gemini-2.5-flash', 'gpt-4.1']
    parser.add_argument("--model", type=str, default='Qwen3-32B', help="base LLM model.")
    parser.add_argument("--kg_model", type=str, default=None, help="Model for KG module. Defaults to --model.")
    parser.add_argument("--kg_base_url", type=str, default=None, help="Base URL for KG module. Defaults to --base_url.")
    parser.add_argument("--wiki_model", type=str, default=None, help="Model for Wiki module. Defaults to --model.")
    parser.add_argument("--wiki_base_url", type=str, default=None, help="Base URL for Wiki module. Defaults to --base_url.")
    parser.add_argument("--embedding_model", type=str, default='BAAI/bge-m3', help="base embedding model.")
    parser.add_argument("--max_length", type=int, help="the max length of LLMs output.")
    parser.add_argument("--max_length_entity_link", type=int, default=None, help="the max length of entity link LLMs output.")
    parser.add_argument("--max_length_plan", type=int, default=None, help="the max length of plan LLMs output.")
    parser.add_argument("--max_length_relation_discovery", type=int, default=None, help="the max length of relation discovery LLMs output.")
    parser.add_argument("--max_display_facts", type=int, default=1500, help="Maximum number of facts to display for an entity during KG fact pruning.")
    parser.add_argument("--fact_pruning_retries", type=int, default=5, help="the max number of retries for fact pruning/extraction in kg exploration.")
    parser.add_argument("--kg_prune_method", type=str, default="filter", choices=["filter", "extract"], help="Method for processing retrieved KG facts: 'filter' for strict JSON filtering (original), 'extract' for text-based information extraction.")
    parser.add_argument("--kg_verbose_report", action=argparse.BooleanOptionalAction, default=False, help="Whether to include debug info (candidates, reasoning, summary) in the KG report.")
    parser.add_argument("--wiki_verbose_report", action=argparse.BooleanOptionalAction, default=False, help="Include detailed LLM rationales and traces in Wikipedia outputs.")
    parser.add_argument("--plan_retries", type=int, default=5, help="the max number of retries for initial planning.")
    parser.add_argument("--kg_top_k", type=int, default=20, help="the top k candidates for entity linking in kg exploration.")
    parser.add_argument("--entity_link_method", type=str, default="analysis", choices=["simple", "advanced", "analysis"], help="Version of prompt for entity linking.")
    parser.add_argument("--entity_link_context", type=str, default="question_query",
                        choices=["question_only", "query_only", "question_query", "instruct"],
                        help="Method to construct the context for entity linking scoring.")
    # --enable_thinking, --no-enable_thinking
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False, help="For Qwen3, enable thinking mode.")
    parser.add_argument("--max_turns", type=int, default=4, help="the max number of turns.")
    parser.add_argument("--skip_tier1_fallback", action=argparse.BooleanOptionalAction, default=False, help="skip tier 1 fallback as practiced in ToG2.")
    parser.add_argument("--resume_run_id", type=str, default=None, help="resume run id.")
    parser.add_argument("--run_unfinished", action=argparse.BooleanOptionalAction, default=True, help="For resuming runs: run unfinished questions.")
    parser.add_argument("--rerun_specific_indices", type=int, nargs='+', default=None, help="For resuming runs: A list of specific question indices to rerun, e.g., --rerun_specific_indices 5 12 42.")
    parser.add_argument("--rerun_failed_only", action="store_true", help="For resuming runs: Only rerun questions that failed due to network issues in the previous run (wiki_failures).")
    parser.add_argument("--run_specific_indices", type=int, nargs='+', default=None, help="For new runs: A list of specific question indices to run, e.g., --run_specific_indices 0 1 2 3 4.")
    parser.add_argument("--num_sample", type=int, default=None, help="Limit the number of questions to run in this session.")
    parser.add_argument("--start", type=int, default=0, help="The starting index of the sample to run.")
    parser.add_argument("--server_urls", type=str, default="server_urls.txt", help="The address of the Wikidata service.")
    parser.add_argument("--wikipedia_method", type=str, default="with_tables", 
                        choices=["text_only", "with_tables", "full_section"], 
                        help="Choose the version of retrieve_info_wikipedia function: 'text_only' for basic version, 'with_tables' for version with table support (default).")
    parser.add_argument("--synthesis_method", type=str, default="extract_and_judgment", 
                        choices=["extract_and_judgment", "evaluate_and_extract"], 
                        help="Choose the version of prompt for synthesis: 'extract_and_judgment' for basic verbatim extraction (default), 'evaluate_and_extract' for the advanced evaluation and extraction version.")
    parser.add_argument("--max_page_retrieval_interactions", type=int, default=6, help="the max number of interactions for retrieve_wiki_page.")
    parser.add_argument("--section_chunks", type=int, default=3, help="The number of top relevant chunks to retrieve from each section.")
    parser.add_argument("--tag", type=str, default=None, help="A custom tag to append to the experiment ID for easy identification.")
    parser.add_argument("--use_indent", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable text indentation for model inputs.")
    parser.add_argument("--result_dir", type=str, default="results", help="The base directory to store experiment results.")
    
    args = parser.parse_args()
    
    dataset, question_string = prepare_dataset(args.dataset)

    encode_kwargs = {"normalize_embeddings": True}
    embed_model_name = args.embedding_model
    
    if "e5" in embed_model_name.lower() and "instruct" in embed_model_name.lower():
        embeddings = E5InstructEmbeddings(model_name=embed_model_name, encode_kwargs=encode_kwargs)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, encode_kwargs=encode_kwargs)
    print(f"Embedding model {embed_model_name} initialized.")

    with open(args.server_urls, "r") as f:
        server_addrs = f.readlines()
        server_addrs = [addr.strip() for addr in server_addrs]
    client = MultiServerWikidataQueryClient(server_addrs)

    # --- Run on Full Dataset and Log Results ---

    # 1. Initialize the logger
    # You can resume a previous run by providing its ID
    resume_run_id = args.resume_run_id

    run_unfinished = args.run_unfinished
    rerun_specific_indices = args.rerun_specific_indices
    run_specific_indices = args.run_specific_indices
    rerun_failed_only = args.rerun_failed_only
    limit = args.num_sample
    start = args.start

    logger = ExperimentLogger(
        experiment_name=f"{args.dataset}_{args.model}_{args.embedding_model.split('/')[-1]}",
        model_config=vars(args),
        dataset_name=args.dataset,
        description="Full run of the CoG agent on the development set. KG display truncated.",
        resume_from_id=resume_run_id,
        tag=args.tag,
        base_dir=args.result_dir
    )

    # 2. Identify questions to run
    # If resuming, filter out already completed questions
    questions_to_run_with_indices = prepare_question_queue(
        dataset=dataset,
        logger=logger,
        resume_run_id=resume_run_id,
        run_unfinished=run_unfinished,
        rerun_specific_indices=rerun_specific_indices,
        run_specific_indices=run_specific_indices,
        rerun_failed_only=rerun_failed_only,
        limit=limit,
        start=start
    )
    print(f"Total questions to process in this run: {len(questions_to_run_with_indices)}")


    # 3. Main loop over the dataset
    for original_index, item in questions_to_run_with_indices:
        question = item[question_string]
        # 使用原始索引作为 question_id
        question_id = original_index
        ground_truth_answer = item.get('answer', 'N/A')

        print("\n" + "="*50)
        print(f"Processing Question (Index: {question_id})")
        print(f"Question: {question}")
        print("="*50 + "\n")

        # Call the encapsulated workflow function
        structured_output, interaction_history, failure_details, wiki_failure_flag, final_notebook, _, llm_errors, accessed_wiki_pages = run_full_workflow(
            question=question,
            args=args,
            client=client,
            embeddings=embeddings
        )
        
        # Log the result, passing the index as the ID
        logger.log_result(
            question_id=question_id,
            question_text=question,
            ground_truth_answer=ground_truth_answer,
            structured_output=structured_output,
            full_interaction_history=interaction_history,
            final_notebook=final_notebook,
            wiki_failure_flag=wiki_failure_flag,
            failure_details=failure_details,
            llm_errors=llm_errors,
            accessed_wiki_pages=accessed_wiki_pages
        )

    # 5. Finalize the logger to generate summary files
    failed_ids_for_rerun = logger.finalize()

    # 6. Rerun failed questions caused by network issues, up to 3 times
    max_reruns = 3
    if failed_ids_for_rerun:
        # Update metadata to reflect that the re-run process is starting
        logger.mark_rerun_start()
        # Create a map for quick lookup once
        dataset_map = {idx: item for idx, item in enumerate(dataset)}

        for i in range(max_reruns):
            if not failed_ids_for_rerun:
                print("No more questions with wiki failures to rerun.")
                break

            print("\n" + "="*20 + f" Rerunning Network Failures (Attempt {i+1}/{max_reruns}) " + "="*20)
            print(f"Found {len(failed_ids_for_rerun)} questions that failed due to Wikipedia or LLM timeout issues. Rerunning them...")

            rerun_questions_with_indices = []
            for failed_id in sorted(failed_ids_for_rerun): # Sort for predictable order
                if failed_id in dataset_map:
                    rerun_questions_with_indices.append((failed_id, dataset_map[failed_id]))
                else:
                    print(f"Warning: Could not find question with index {failed_id} in the original dataset for rerunning.")

            for original_index, item in rerun_questions_with_indices:
                question = item[question_string]
                question_id = original_index
                ground_truth_answer = item.get('answer', 'N/A')

                print("\n" + "="*50)
                print(f"Rerunning Question (Index: {question_id})")
                print(f"Question: {question}")
                print("="*50 + "\n")

                # Call the encapsulated workflow function again
                structured_output, interaction_history, failure_details, wiki_failure_flag, final_notebook, _, llm_errors, accessed_wiki_pages = run_full_workflow(
                    question=question,
                    args=args,
                    client=client,
                    embeddings=embeddings
                )
                
                # Log the new result, overwriting the previous one
                logger.log_result(
                    question_id=question_id,
                    question_text=question,
                    ground_truth_answer=ground_truth_answer,
                    structured_output=structured_output,
                    full_interaction_history=interaction_history,
                    final_notebook=final_notebook,
                    wiki_failure_flag=wiki_failure_flag,
                    failure_details=failure_details,
                    llm_errors=llm_errors,
                    accessed_wiki_pages=accessed_wiki_pages
                )
            
            print(f"\n--- Rerun attempt {i+1} complete. Finalizing results again to update summaries. ---")
            failed_ids_for_rerun = logger.finalize()

    if failed_ids_for_rerun:
        print(f"\nWarning: After {max_reruns} retries, there are still {len(failed_ids_for_rerun)} questions with wiki failures.")
        print(f"Final failed question IDs: {sorted(failed_ids_for_rerun)}")

    print("\n--- All questions processed. Experiment complete. ---")

if __name__ == "__main__":
    main()