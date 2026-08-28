import re
import ast
from string import Template
from utils import generate_process, indent_text

# =================================================================
# 1. Prompt Template for Next Step Planning
# =================================================================

prompt_next_step_planning = Template('''\
### ROLE
You are a master AI strategist leading a multi-hop question-answering mission. Your task is to plan the next step of the investigation after an information-gathering turn that was useful but insufficient.

### CONTEXT
- **Original Question:** ${question}
- **Current Notebook (accumulated facts):**
${notebook}
- **Previous Overall Plan (Analysis):** ${analysis}
- **Previous Sub-Queries:** ${queries}
- **Previous Core Entities:** ${entities}
- **Global Candidate Entities Pool (unexplored leads with reasons):**
${candidate_entities_pool}

### NEW FINDINGS FROM LAST TURN
- **Thought Process:** ${thought_process}
- **Extracted Content:**
${extracted_content}

### YOUR TASK
Based on all the context and the latest findings, you must plan the next round of investigation. The previous round was deemed INSUFFICIENT_USEFUL, which means we are on the right track and must now dig deeper. Your goal is to build on this progress by generating a new set of queries that bridges the specific information gaps you identified, moving us closer to the final answer.

1.  **Update Notebook:** Combine the "Current Notebook" with the "Extracted Content" to create a new, comprehensive notebook. Your goal is to produce a single, coherent summary of all known facts. Crucially, you must retain all unique and relevant information from the original notebook while integrating the new findings. You can merge, rephrase, or restructure facts for clarity, but you must not drop any important details.
2.  **Update Analysis:** Revise the "Previous Overall Plan" to reflect the new state of your understanding. Synthesize the new information with your existing understanding to formulate a new strategy. Clearly explain what the immediate next step should focus on and why.
3.  **Plan Next Queries and Entities:** Define a new set of sub-queries and corresponding entities for the next round of information gathering. These should be based on your updated analysis and aim to fill the identified knowledge gaps. Each entity in the `Next Entities` list must be the **most specific and complete entity name** from the corresponding query. This is crucial for accurate knowledge graph linking.
    - Append disambiguation terms for ambiguous entities (e.g., "Apple (company)", "Lincoln (film)"). Never repeat an ambiguous entity name that previously led to irrelevant results.
4.  **Manage Candidate Pool:** This is your long-term memory for promising but deferred exploration paths.
    a. **Review Existing Leads:** Examine the current `Global Candidate Entities Pool`. Decide if any of them have become high-priority and should be promoted to `Next Queries` in this turn.
    b. **Identify New Leads:** Scan the `NEW FINDINGS FROM LAST TURN` for any new entities or concepts that are promising for future steps but exceed the 5-query limit for the current turn.
    c. **Update and Maintain the Pool:** Generate the new `Candidate Pool` list by following these three rules:
        - **Carry Over:** Keep all leads from the old pool that were not promoted to a query this turn.
        - **Add New:** Add the new leads you identified in step (b), ensuring each has an `entity` and a `reason`.
        - **Remove Promoted:** Remove any lead from the old pool that you used to create a `Next Query` in this turn.

### CONSTRAINTS
- The number of `Next Queries` must not exceed 5.
- `Next Queries` and `Next Entities` lists must have the exact same number of items.
- All outputs must be derived from the provided context and new findings.
- Only output the queries/entities for the **immediate next step**. Do not plan, describe, or allude to any subsequent steps.
- The `Updated Candidate Pool` must be a list of dictionaries, where each dictionary has "entity" and "reason" keys.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Thought Process: Your step-by-step reasoning on how you will approach each part of the task.
Updated Notebook: Your updated notebook content.
Updated Analysis: Your new, revised analysis and plan.
Next Queries: [A Python-style list of concise and effective search query strings.]
Next Entities: [A Python-style list of the **most specific**, core entities in next queries, corresponding one-to-one with the Next Queries list.]
Updated Candidate Pool: [A Python-style list of dictionaries, e.g., [{"entity": "Name", "reason": "Why it's a promising lead."}]]

### EXAMPLE OF A VALID RESPONSE
Thought Process: 1. The last turn confirmed Satya Nadella is CEO and also brought up "Microsoft Azure" as a key product. The notebook should be updated to reflect both facts. 2. The main goal is still high-level company understanding. So, the top priorities are getting overall financial data and leadership background. A deep dive into a specific product line like Azure is a secondary priority right now. 3. Based on this, I'll create queries for "Microsoft revenue 2023" and "Satya Nadella early life and education". 4. Since Azure is important but not urgent, it's a perfect candidate for the pool. I'll add it with a clear reason so we can explore it in a future turn if needed.
Updated Notebook: - Satya Nadella is the CEO of Microsoft.
- Microsoft Azure is a key product line of Microsoft.
Updated Analysis: We have identified Satya Nadella as the CEO of Microsoft and noted that Microsoft Azure is a key growth area under his leadership. The next step is to find Microsoft's revenue for the fiscal year 2023 and find Satya Nadella's background for more context.
Next Queries: ["Microsoft revenue 2023", "Satya Nadella early life and education"]
Next Entities: ["Microsoft", "Satya Nadella"]
Updated Candidate Pool: [{"entity": "Microsoft Azure", "reason": "A key product line, its revenue could be relevant for deep analysis."}]

### YOUR RESPONSE:
''')

'''
For each query, you must then extract the **most specific subject** to act as its corresponding entity. This is critical for accurate downstream processing.
'''

# =================================================================
# 2. Parsing Function for the LLM Output (MODIFIED for new order)
# =================================================================

def parse_next_step_planning_result(result_text: str):
    """
    Parses the LLM's output for the next step planning, matching the updated prompt format.
    """
    try:
        # 按照新的 Prompt 输出顺序调整正则表达式
        thought_process_match = re.search(r"Thought Process:(.*?)Updated Notebook:", result_text, re.DOTALL)
        updated_notebook_match = re.search(r"Updated Notebook:(.*?)Updated Analysis:", result_text, re.DOTALL)
        updated_analysis_match = re.search(r"Updated Analysis:(.*?)Next Queries:", result_text, re.DOTALL)
        next_queries_match = re.search(r"Next Queries:(.*?)Next Entities:", result_text, re.DOTALL)
        next_entities_match = re.search(r"Next Entities:(.*?)Updated Candidate Pool:", result_text, re.DOTALL)
        updated_candidate_pool_match = re.search(r"Updated Candidate Pool:(.*)", result_text, re.DOTALL)

        if not all([thought_process_match, updated_notebook_match, updated_analysis_match, next_queries_match, next_entities_match, updated_candidate_pool_match]):
            print("Parsing Error: Could not find one or more required fields in the planning output.")
            print(f"Full Response:\n{result_text}")
            return None

        thought_process = thought_process_match.group(1).strip()
        updated_notebook = updated_notebook_match.group(1).strip()
        updated_analysis = updated_analysis_match.group(1).strip()
        
        # 使用 ast.literal_eval 安全地解析 Python 列表
        next_queries = ast.literal_eval(next_queries_match.group(1).strip())
        next_entities = ast.literal_eval(next_entities_match.group(1).strip())
        updated_candidate_pool = ast.literal_eval(updated_candidate_pool_match.group(1).strip())
        
        if len(next_queries) != len(next_entities):
            print(f"Parsing Error: Mismatch between number of queries ({len(next_queries)}) and entities ({len(next_entities)}).")
            return None

        # 验证 candidate pool 的结构
        if not isinstance(updated_candidate_pool, list) or not all(isinstance(item, dict) and 'entity' in item and 'reason' in item for item in updated_candidate_pool):
            print(f"Parsing Error: 'Updated Candidate Pool' is not a list of dictionaries with 'entity' and 'reason' keys.")
            print(f"Parsed Pool: {updated_candidate_pool}")
            return None

        return {
            "thought_process": thought_process,
            "updated_notebook": updated_notebook,
            "updated_analysis": updated_analysis,
            "next_queries": next_queries,
            "next_entities": next_entities,
            "updated_candidate_pool": updated_candidate_pool,
        }

    except (SyntaxError, ValueError) as e:
        print(f"Parsing Error: Failed to parse lists from the output. Error: {e}")
        print(f"Full Response:\n{result_text}")
        return None
    except Exception as e:
        print(f"An unexpected exception occurred during planning parsing: {e}\nInput text was: {result_text}")
        return None

# =================================================================
# 3. Main Workflow Function to Run the Planning Step
# =================================================================

def run_next_step_planning(question, notebook, analysis, queries, entities, extracted_content, thought_process, candidate_entities_pool, args, max_retries=3):
    """
    Runs the next step planning stage when information is insufficient but useful.
    """
    print("--- Running Next Step Planning ---")

    # Indent the notebook content for proper prompt formatting.
    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."

    template_inputs = {
        'question': question,
        'notebook': indented_notebook,
        'analysis': analysis,
        'queries': queries,
        'entities': entities,
        'extracted_content': extracted_content,
        'thought_process': thought_process,
        'candidate_entities_pool': str(candidate_entities_pool) if candidate_entities_pool else "[]"
    }

    planning_result = generate_process(
        step_name="Plan Next Step",
        prompt_template=prompt_next_step_planning,
        template_inputs=template_inputs,
        parsing_function=parse_next_step_planning_result,
        args=args,
        module='main',
        max_retries=max_retries,
    )
    if not planning_result:
        reason = "LLM failed to generate a valid next step plan after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    print("\n--- Next Step Plan Generated ---")
    print(f"Thought Process: {planning_result['thought_process']}")
    print(f"Updated Notebook: {planning_result['updated_notebook']}")
    print(f"Updated Analysis: {planning_result['updated_analysis']}")
    print(f"Next Queries: {planning_result['next_queries']}")
    print(f"Next Entities: {planning_result['next_entities']}")
    print(f"Updated Candidate Pool: {planning_result['updated_candidate_pool']}")

    return "SUCCESS", planning_result