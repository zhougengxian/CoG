from string import Template
import re
from utils import generate_process
from kg.client import MultiServerWikidataQueryClient
import ast

prompt_select_relations = Template('''\
### ROLE
You are an expert Knowledge Graph analyzer. Your task is to select the most relevant relations (properties) of a given entity that will help answer a specific question.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Query:** "${query}"
- **Entity in Focus:** ${entity_label} (${entity_qid}): ${entity_description}

### AVAILABLE RELATIONS
The entity "${entity_label}" is connected to ${total_relations_count} relations (Outgoing: ${outgoing_count}, Incoming: ${incoming_count}) in the Knowledge Graph.
Below is a list of all relations connected to the entity. They are separated into Outgoing (the entity is the subject) and Incoming (the entity is the object) relations. Each relation includes its label, PID, and a frequency indicating how many other entities in the graph are linked via this relation.

**Outgoing Relations:**
${outgoing_relations}

**Incoming Relations:**
${incoming_relations}

### YOUR TASK
Based on the **CONTEXT**, your task is to select relations that will help find the answer. This involves a two-pronged approach:
1.  **Direct Relations**: Select relations that seem to directly contain the answer.
2.  **Exploratory Relations**: If no direct relations exist, select relations that could lead to intermediate entities, which might then contain the answer.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary:

Reasoning: Your brief thought process explaining why you chose these specific relations based on the question and the entity.
Selected PIDs: [A Python-style list of strings, where each string is the PID of a selected relation (e.g., ["P31", "P17", "P570"]). If no relations seem relevant, output an empty list []]

### YOUR RESPONSE:
''')


def format_relations_for_prompt(all_relations: dict) -> tuple[str, str]:
    """Formats the relations dictionary into a readable string for the LLM prompt."""
    
    def format_single_direction(relations: list) -> str:
        if not relations:
            return "  - None"
        sorted_relations = sorted(relations, key=lambda x: x.get('counter', 0), reverse=True)
        return "\n".join([f"  - {rel['label']} ({rel['pid']}), frequency: {rel.get('counter', 0)}" for rel in sorted_relations])

    outgoing_str = format_single_direction(all_relations.get("head", []))
    incoming_str = format_single_direction(all_relations.get("tail", []))
    
    return outgoing_str, incoming_str


def parse_relation_selection_result(result_text: str, all_relations: dict):
    """
    Parses the LLM's output to extract reasoning and a list of PIDs.
    It also validates that the selected PIDs exist in the provided relations.
    """
    try:
        reasoning_match = re.search(r"Reasoning:(.*?)(?=\nSelected PIDs:)", result_text, re.DOTALL)
        pids_match = re.search(r"Selected PIDs:(.*)", result_text, re.DOTALL)

        if not reasoning_match or not pids_match:
            print(f"Parsing Error: Could not find 'Reasoning' or 'Selected PIDs' in the output.")
            return None

        reasoning = reasoning_match.group(1).strip()
        pids_str = pids_match.group(1).strip()
        
        selected_pids = ast.literal_eval(pids_str)
        
        if not isinstance(selected_pids, list):
            print(f"Parsing Error: 'Selected PIDs' is not a list.")
            return None
            
        # ***[NEW VALIDATION LOGIC]***
        # Create a set of all available PIDs for efficient lookup.
        outgoing_pids = {rel['pid'] for rel in all_relations.get('head', [])}
        incoming_pids = {rel['pid'] for rel in all_relations.get('tail', [])}
        available_pids = outgoing_pids.union(incoming_pids)

        # Check if all PIDs selected by the LLM are valid.
        invalid_pids = [pid for pid in selected_pids if pid not in available_pids]

        if invalid_pids:
            # If any invalid PIDs are found, print a warning and return None to trigger a retry.
            print(f"Parsing Warning: LLM returned PIDs that were not in the available list: {invalid_pids}. This might be a hallucination. Will attempt retry.")
            return None

        # If all checks pass, return the successfully parsed and validated data.
        return {
            "reasoning": reasoning,
            "selected_pids": selected_pids
        }
        
    except Exception as e:
        print(f"An exception occurred during parsing: {e}\nInput text was: {result_text}")
        return None
    
def run_relation_discovery(
    question: str, 
    analysis: str, 
    query: str, 
    linked_entity: dict, 
    client: MultiServerWikidataQueryClient, 
    args,
    max_retries: int = 3
) -> tuple[str, dict]:
    """
    Executes Stage 1: Relation Discovery and Coarse-Grained Filtering.
    Uses an LLM to select the most relevant relations for a given entity.

    Returns:
        A tuple (status, result_dict):
        - status (str): "SUCCESS" or "FAILED".
        - result_dict (dict): 
            - On SUCCESS: Contains keys like 'total_relations_count', 'selected_pids', 
                          'reasoning', and 'all_relations'.
            - On FAILED: Contains a 'reason' key explaining the failure.
    """
    entity_qid = linked_entity.get('qid')
    entity_label = linked_entity.get('label')
    entity_description = linked_entity.get('description', 'N/A')
    
    if not entity_qid:
        reason = "Input `linked_entity` must contain 'qid'."
        print(f"--- [Discovery Failed] {reason} ---")
        return "FAILED", {"reason": reason}
    
    print(f"--- Stage 1: Relation Discovery for '{entity_label}' ({entity_qid}) ---")
    all_relations = client.query_all("get_all_relations_of_an_entity", entity_qid)
    
    if not isinstance(all_relations, dict) or (not all_relations.get("head") and not all_relations.get("tail")):
        print(f"No relations found for entity {entity_qid}. Proceeding with empty selection.")
        return "SUCCESS", {
            "total_relations_count": 0,
            "selected_pids": [],
            "reasoning": "No relations were found in the knowledge graph for this entity.",
            "all_relations": {"head": [], "tail": []}
        }

    outgoing_relations = all_relations.get("head", [])
    incoming_relations = all_relations.get("tail", [])
    outgoing_count = len(outgoing_relations)
    incoming_count = len(incoming_relations)
    total_relations_count = outgoing_count + incoming_count
    
    print(f"Found {total_relations_count} total relations (Outgoing: {outgoing_count}, Incoming: {incoming_count}).")

    outgoing_relations_str, incoming_relations_str = format_relations_for_prompt(all_relations)

    template_inputs = {
        'question': question,
        'analysis': analysis,
        'query': query,
        'entity_qid': entity_qid,
        'entity_label': entity_label,
        'entity_description': entity_description,
        'total_relations_count': total_relations_count,
        'outgoing_count': outgoing_count,
        'incoming_count': incoming_count,
        'outgoing_relations': outgoing_relations_str,
        'incoming_relations': incoming_relations_str,
    }

    selection_result = generate_process(
        step_name=f"Select Relations for '{entity_label}'",
        prompt_template=prompt_select_relations,
        template_inputs=template_inputs,
        parsing_function=lambda text: parse_relation_selection_result(text, all_relations),
        args=args,
        module='kg',
        max_retries=max_retries,
        max_tokens=args.max_length_relation_discovery
    )
    
    if not selection_result:
        reason = f"LLM failed to select relations for '{entity_label}' after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
        
    selected_pids = selection_result.get("selected_pids", [])
    reasoning = selection_result.get("reasoning", "No reasoning provided.")
    
    print(f"LLM selected {len(selected_pids)} relations out of {total_relations_count}.")
    print(f"LLM Reasoning: {reasoning}")
    print(f"Selected PIDs: {selected_pids}")
    
    final_result = {
        "total_relations_count": total_relations_count,
        "selected_pids": selected_pids,
        "reasoning": reasoning,
        "all_relations": all_relations
    }
    return "SUCCESS", final_result

