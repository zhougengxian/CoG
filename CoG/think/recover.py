import re
import ast
from string import Template
from utils import generate_process, indent_text

# =================================================================
# 1. Prompt Template for Recovery Planning
# =================================================================

prompt_recovery_planning = Template('''\
### ROLE
You are a master AI strategist leading a multi-hop question-answering mission. Your task is to recover from a failed information-gathering turn where the retrieved evidence was useless.

### CONTEXT
- **Original Question:** ${question}
- **Current Notebook (accumulated facts):**
${notebook}
- **Global Candidate Entities Pool (unexplored leads with reasons):**
${candidate_entities_pool}
- **Interaction History (what has been tried so far):**
${interaction_history}

### FAILED TURN DETAILS
The last turn was judged as INSUFFICIENT_USELESS. Here's what went wrong:
- **Previous Overall Plan (Analysis):** ${analysis}
- **Failed Sub-Queries:** ${queries}
- **Failed Core Entities:** ${entities}
- **Extracted Content:** ${extracted_content}
- **Reasoning for Failure (Thought Process):** ${thought_process}

### YOUR TASK
The previous approach has led to a dead end. You must critically reflect on the entire interaction history, diagnose the strategic error and devise a new plan. Your goal is to pivot your strategy by generating a completely new set of queries to get the investigation back on track.

1.  **Critical Reflection:** Based on the `Interaction History` and `FAILED TURN DETAILS`, critically analyze why the previous approach failed. Was the initial plan flawed? Were the entities wrong? Were the queries poorly formulated? Summarize the core strategic error and derive actionable **guiding principles** for the next attempt.
2.  **Update Analysis:** Based on your reflection, propose a fundamentally new plan or direction. Your new plan should leverage insights from your reflection, potentially by promoting an entity from the `Candidate Pool`, re-examining the original question for missed keywords, or formulating queries from an entirely different angle.
3.  **Plan Next Queries and Entities:** Based on your new analysis, define a new set of sub-queries and corresponding entities. These must represent a clear change in direction from the failed queries.
    - If you suspect the previous failure was due to entity ambiguity, you MUST update the Next Entities to include specific disambiguation terms (e.g., "Apple (company)", "Lincoln (film)").
4.  **Manage Candidate Pool:** Review the `Global Candidate Entities Pool` and the `FAILED TURN DETAILS`.
    a. Decide if any existing leads in the pool are now high-priority for your new strategy.
    b. Add any new promising leads if any were incidentally discovered even in the failed turn.
    c. Generate the `Updated Candidate Pool` by carrying over unused leads, adding new ones, and removing any that are now being promoted to active queries.

### CONSTRAINTS
- The number of `Next Queries` must not exceed 5.
- `Next Queries` and `Next Entities` lists must have the exact same number of items.
- The `Updated Candidate Pool` must be a list of dictionaries, where each dictionary has "entity" and "reason" keys.
- Your primary goal is to change the direction of the investigation. Avoid queries that are only minor variations of the failed ones.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Thought Process: Your step-by-step critical reflection on the failure and your reasoning for the new plan.
Updated Analysis: Your new, revised analysis and plan.
Next Queries: [A Python-style list of concise and effective search query strings.]
Next Entities: [A Python-style list of the core entities in next queries, corresponding one-to-one with the Next Queries list.]
Updated Candidate Pool: [A Python-style list of dictionaries, e.g., [{"entity": "Name", "reason": "Why it's a promising lead."}]]

### YOUR RESPONSE:
''')

# =================================================================
# 2. Parsing Function for the LLM Output
# =================================================================

def parse_recovery_planning_result(result_text: str):
    """
    Parses the LLM's output for the recovery planning.
    """
    try:
        thought_process_match = re.search(r"Thought Process:(.*?)Updated Analysis:", result_text, re.DOTALL)
        updated_analysis_match = re.search(r"Updated Analysis:(.*?)Next Queries:", result_text, re.DOTALL)
        next_queries_match = re.search(r"Next Queries:(.*?)Next Entities:", result_text, re.DOTALL)
        next_entities_match = re.search(r"Next Entities:(.*?)Updated Candidate Pool:", result_text, re.DOTALL)
        updated_candidate_pool_match = re.search(r"Updated Candidate Pool:(.*)", result_text, re.DOTALL)

        if not all([thought_process_match, updated_analysis_match, next_queries_match, next_entities_match, updated_candidate_pool_match]):
            print("Parsing Error: Could not find one or more required fields in the recovery planning output.")
            print(f"Full Response:\n{result_text}")
            return None

        thought_process = thought_process_match.group(1).strip()
        updated_analysis = updated_analysis_match.group(1).strip()
        
        next_queries = ast.literal_eval(next_queries_match.group(1).strip())
        next_entities = ast.literal_eval(next_entities_match.group(1).strip())
        updated_candidate_pool = ast.literal_eval(updated_candidate_pool_match.group(1).strip())
        
        if len(next_queries) != len(next_entities):
            print(f"Parsing Error: Mismatch between number of queries ({len(next_queries)}) and entities ({len(next_entities)}).")
            return None

        if not isinstance(updated_candidate_pool, list) or not all(isinstance(item, dict) and 'entity' in item and 'reason' in item for item in updated_candidate_pool):
            print(f"Parsing Error: 'Updated Candidate Pool' is not a list of dictionaries with 'entity' and 'reason' keys.")
            print(f"Parsed Pool: {updated_candidate_pool}")
            return None

        return {
            "thought_process": thought_process,
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
        print(f"An unexpected exception occurred during recovery parsing: {e}\nInput text was: {result_text}")
        return None

# =================================================================
# 3. Main Workflow Function to Run the Recovery Step
# =================================================================

def run_recovery_planning(question, notebook, analysis, queries, entities, extracted_content, thought_process, interaction_history, candidate_entities_pool, args, max_retries=3):
    """
    Runs the recovery planning stage when information is insufficient and useless.
    """
    print("--- Running Recovery Planning ---")

    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."

    template_inputs = {
        'question': question,
        'notebook': indented_notebook,
        'analysis': analysis,
        'queries': queries,
        'entities': entities,
        'extracted_content': extracted_content,
        'thought_process': thought_process,
        'interaction_history': str(interaction_history) if interaction_history else "[]",
        'candidate_entities_pool': str(candidate_entities_pool) if candidate_entities_pool else "[]"
    }

    planning_result = generate_process(
        step_name="Plan Recovery",
        prompt_template=prompt_recovery_planning,
        template_inputs=template_inputs,
        parsing_function=parse_recovery_planning_result,
        args=args,
        module='main',
        max_retries=max_retries,
    )
    if not planning_result:
        reason = "LLM failed to generate a valid recovery plan after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    print("\n--- Recovery Plan Generated ---")
    print(f"Thought Process: {planning_result['thought_process']}")
    print(f"Updated Analysis: {planning_result['updated_analysis']}")
    print(f"Next Queries: {planning_result['next_queries']}")
    print(f"Next Entities: {planning_result['next_entities']}")
    print(f"Updated Candidate Pool: {planning_result['updated_candidate_pool']}")

    return "SUCCESS", planning_result