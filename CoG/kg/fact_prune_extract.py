import json
import re
from string import Template
import functools
import math

from utils import generate_process
from kg.client import MultiServerWikidataQueryClient

# =========================================================================================
# STAGE 2: Fact Retrieval & Information Extraction
# =========================================================================================

# --- 2.1: Prompt Template for Unified Fact Extraction ---

prompt_extract_facts_unified = Template('''\
### ROLE
You are an expert assistant specializing in knowledge graph analysis for question answering. Your mission is to intelligently extract key information from a list of retrieved facts, guiding a multi-step exploration process.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Query:** "${query}"
- **Entity in Focus:** ${entity_label}

### AVAILABLE FACTS
Below is a structured list of all facts retrieved from the Knowledge Graph for the entity "${entity_label}". The facts are categorized by relation type (Outgoing or Incoming).

**Fact Format:**
Each line presents a single relation and its associated facts in the following format:
- **Relation Label (Relation ID):** [List of fact labels]

- **Relation Label:** The human-readable name of the relation (e.g., "country").
- **Relation ID:** The unique identifier for the relation (e.g., "P17").
- **List of fact labels:** A JSON array of strings, where each string is a value (like dates or numbers) or an entity label associated with the main entity through that relation.

**Retrieved Facts:**
${formatted_facts}

### YOUR TASK
Your goal is to read the retrieved Knowledge Graph facts and extract the most valuable information. Evaluate the facts using a two-pronged approach:
1. **Direct Information**: Extract any facts that directly answer or help answer the **Current Sub-Query**.
2. **Promising Clues**: Extract key facts that, while not direct answers, provide crucial context or promising leads for solving the Original Question. These will serve as stepping stones for the next phase of investigation.
Ignore irrelevant or noisy facts. If no facts help answer the query or provide clues for the original question, do not force an extraction; simply state that no relevant information was found.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary before or after it. Each key must be on a new line.

Rationale: Briefly explain your thought process. What relevant information did you find in the facts? How does it help answer the sub-query or provide clues for the original question? If no relevant information was found, state that clearly.
Extracted Info: A synthesized paragraph or bulleted list of the key information and promising clues you extracted from the facts. Adhere strictly to the information present in the "AVAILABLE FACTS". If no relevant information is found, you MUST state exactly "None". Do not include any explanations or reasoning here.

### YOUR RESPONSE:
''')



# --- 2.2: Helper Functions for Formatting, Parsing, and Reporting ---

def format_facts_for_extraction_prompt(
    retrieved_facts_by_pid: dict, entity_label: str, 
    max_display_facts: int, min_facts_per_group: int = 50
) -> str:
    """
    Formats the retrieved facts into a structured, readable string for the LLM pruning prompt.
    Limits the total number of entities to max_display_facts using a hybrid allocation strategy:
    1. Guarantees a minimum number of slots (min_facts_per_group) for each relation direction.
    2. Distributes the remaining slots proportionally based on the remaining fact counts.
    This prevents large relations from completely starving smaller but potentially relevant ones.
    """
    outgoing_lines = [f"**Outgoing Relations (Entity '{entity_label}' is the subject):**"]
    incoming_lines = [f"**Incoming Relations (Entity '{entity_label}' is the object):**"]

    # 1. Collect all fact groups and total count
    all_fact_groups = []
    total_facts_count = 0
    original_facts = {}

    for pid, data in retrieved_facts_by_pid.items():
        outgoing_facts = (data.get('outgoing_entities', []) or []) + (data.get('literal_values', []) or [])
        incoming_facts = data.get('incoming_entities', []) or []
        
        original_facts[pid] = {'outgoing': outgoing_facts, 'incoming': incoming_facts}

        if outgoing_facts:
            all_fact_groups.append({'pid': pid, 'direction': 'outgoing', 'facts': outgoing_facts})
            total_facts_count += len(outgoing_facts)
        if incoming_facts:
            all_fact_groups.append({'pid': pid, 'direction': 'incoming', 'facts': incoming_facts})
            total_facts_count += len(incoming_facts)

    # 2. If total exceeds limit, calculate limits using the "guaranteed minimum + proportional" strategy
    limits = {} # Key: (pid, direction), Value: limit
    if total_facts_count > max_display_facts:
        # Stage 1: Assign base minimum allocation
        base_assigned_slots = 0
        remaining_pool_groups = []
        
        for group in all_fact_groups:
            pid, direction, facts = group['pid'], group['direction'], group['facts']
            num_facts = len(facts)
            
            base_allocation = min(num_facts, min_facts_per_group)
            limits[(pid, direction)] = base_allocation
            base_assigned_slots += base_allocation
            
            if num_facts > min_facts_per_group:
                remaining_pool_groups.append({
                    'pid': pid, 'direction': direction,
                    'count': num_facts - min_facts_per_group
                })

        # Stage 2: Proportionally distribute remaining slots
        remaining_slots_to_allocate = max_display_facts - base_assigned_slots
        total_remaining_facts_count = sum(item['count'] for item in remaining_pool_groups)
        
        if remaining_slots_to_allocate > 0 and total_remaining_facts_count > 0:
            for group in remaining_pool_groups:
                pid, direction = group['pid'], group['direction']
                remaining_count = group['count']
                
                proportion = remaining_count / total_remaining_facts_count
                extra_slots = math.ceil(remaining_slots_to_allocate * proportion)
                
                # Add extra slots to the base allocation
                # Ensure we don't exceed the actual number of remaining facts
                original_total_facts = len(original_facts[pid][direction])
                current_limit = limits.get((pid, direction), 0)
                
                # The total extra slots we can add is total_facts - current_limit
                max_addable_slots = original_total_facts - current_limit
                
                # Add the smaller of the calculated extra slots or the max addable slots
                limits[(pid, direction)] += min(extra_slots, max_addable_slots)
    
    # 3. Format the facts, applying truncation and adding markers if necessary
    for pid, data in retrieved_facts_by_pid.items():
        label = data.get('label', pid)
        
        outgoing_facts = original_facts[pid]['outgoing']
        incoming_facts = original_facts[pid]['incoming']
        
        # Format outgoing facts
        if outgoing_facts:
            limit = limits.get((pid, 'outgoing'), len(outgoing_facts))
            display_facts = outgoing_facts[:limit]

            if display_facts:  # Only add if there are facts to show after truncation
                outgoing_str = json.dumps(display_facts)
                if len(outgoing_facts) > limit:
                    hidden_count = len(outgoing_facts) - limit
                    outgoing_str += f" ... ({hidden_count} more)"
                outgoing_lines.append(f"- {label} ({pid}): {outgoing_str}")

        # Format incoming facts
        if incoming_facts:
            limit = limits.get((pid, 'incoming'), len(incoming_facts))
            display_facts = incoming_facts[:limit]
            
            if display_facts: # Only add if there are facts to show after truncation
                incoming_str = json.dumps(display_facts)
                if len(incoming_facts) > limit:
                    hidden_count = len(incoming_facts) - limit
                    incoming_str += f" ... ({hidden_count} more)"
                incoming_lines.append(f"- {label} ({pid}): {incoming_str}")

    # If no lines added (only headers), append - None
    if len(outgoing_lines) == 1:
        outgoing_lines.append("- None")
    if len(incoming_lines) == 1:
        incoming_lines.append("- None")

    return "\n".join(outgoing_lines) + "\n\n" + "\n".join(incoming_lines)


def parse_extraction_result(result_text: str, retrieved_facts_by_pid: dict = None):
    """
    Parses the LLM's output for fact extraction.
    """
    try:
        # Extract Rationale
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Info:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."
        
        # Extract Info
        info_match = re.search(r"Extracted Info:(.*)", result_text, re.DOTALL)
        extracted_info_raw = info_match.group(1).strip() if info_match else ""
        extracted_info = None if extracted_info_raw.lower() == 'none' else extracted_info_raw
        
        if not rationale_match or not info_match:
            print(f"Parsing Error: Missing required markers in the output.\nInput text was: {result_text}")
            return None
            
        print("Successfully parsed the extracted facts from LLM.")
        return {
            "reasoning": rationale,
            "extracted_info": extracted_info
        }
    except Exception as e:
        print(f"Parsing Error: Failed to parse LLM output. Error: {e}\nInput text was: {result_text}")
        return None


def format_exploration_report(
    entity_mention: str,
    linked_candidates: list,
    linked_entity: dict,
    extracted_info: str,
    total_relations_count: int,
    selected_pids_count: int,
    total_facts_count: int,
    relation_selection_reasoning: str,
    fact_extraction_reasoning: str,
    max_candidates_display: int = 4,
    verbose: bool = False
) -> str:
    """
    Generates the final, human-readable structured report string.
    """
    # 1. 报告主标题
    report_parts = [f'KG Exploration for Mention: "{entity_mention}"']

    # 2. 构建 [Entity Linking] 模块   
    linking_section = ["\n[Entity Linking]"]
    linking_section.append(f'- Linked Entity: {linked_entity.get("label")} ({linked_entity.get("qid")})')
    linking_section.append(f'- Description: {linked_entity.get("description", "N/A")}')
    
    # 2.1 添加带截断的候选列表
    if verbose and linked_candidates:
        display_k = min(len(linked_candidates), max_candidates_display)
        linking_section.append(f'- Candidates (top-{display_k} preview):')
        for candidate in linked_candidates[:max_candidates_display]:
            desc = candidate.get("description", "N/A")
            if len(desc) > 100:
                desc = desc[:100] + "..."
            linking_section.append(f'  - {candidate.get("label")} ({candidate.get("qid")}): {desc}')
        
        remaining_count = len(linked_candidates) - max_candidates_display
        if remaining_count > 0:
            linking_section.append(f'  - ... (and {remaining_count} more)')

    # 2.2 添加链接理由
    rationale = linked_entity.get("analysis")
    if verbose and rationale:
        if not isinstance(rationale, str):
            rationale = json.dumps(rationale, ensure_ascii=False)
        rationale_single_line = rationale.replace('\n', ' ')
        linking_section.append(f'- Linking Rationale: {rationale_single_line}')
    
    report_parts.append("\n".join(linking_section))
    
    # 3. 构建 [Retrieved Facts for "..."] 模块 (保留空行)
    report_parts.append(f'\n[Retrieved Facts for "{linked_entity.get("label")}"]')
    report_parts.append(extracted_info)

    # Add the summary
    if verbose:
        summary_section = [
            "\n[Filtering Summary]",
            f"- Relations explored: {selected_pids_count} of {total_relations_count} total",
            f"- Total facts retrieved: {total_facts_count}"
        ]
        report_parts.append("\n".join(summary_section))
        
        # Add the reasoning section
        # First, replace newlines to avoid f-string syntax errors and for cleaner output
        # Handle cases where reasoning might be a dict or list (LLM structured output)
        if not isinstance(relation_selection_reasoning, str):
            relation_selection_reasoning = json.dumps(relation_selection_reasoning, ensure_ascii=False)
        rel_reason_single_line = relation_selection_reasoning.replace('\n', ' ')
    
        if not isinstance(fact_extraction_reasoning, str):
            fact_extraction_reasoning = json.dumps(fact_extraction_reasoning, ensure_ascii=False)
        fact_reason_single_line = fact_extraction_reasoning.replace('\n', ' ')
        
        reasoning_section = [
            "\n[Reasoning]",
            f"- Relation Selection: {rel_reason_single_line}",
            f"- Fact Extraction: {fact_reason_single_line}"
        ]
        report_parts.append("\n".join(reasoning_section))
    
    return "\n".join(report_parts)


# --- 2.3: Main Orchestrator Function for Stage 2 ---

def run_fact_retrieval_and_extraction(
    question: str, 
    analysis: str, 
    query: str, 
    entity_mention: str,
    linked_candidates: list,
    linked_entity: dict, 
    discovery_result: dict,
    client: MultiServerWikidataQueryClient, 
    args,
    max_retries: int = 3
) -> tuple[str, str]:
    """
    Executes Stage 2: Fact Retrieval, Information Extraction, and Report Generation.
    """
    entity_qid = linked_entity.get('qid')
    entity_label = linked_entity.get('label')
    selected_pids = discovery_result.get('selected_pids', [])
    all_relations = discovery_result.get('all_relations', {})
    
    print(f"\n--- Stage 2: Fact Retrieval & Extraction for '{entity_label}' ({len(selected_pids)} relations) ---")
    
    if not selected_pids:
        print("No relations were selected in Stage 1. Skipping fact retrieval.")
        # Create an empty report
        report = format_exploration_report(
            entity_mention=entity_mention,
            linked_candidates=linked_candidates,
            linked_entity=linked_entity,
            extracted_info="None",
            total_relations_count=discovery_result.get('total_relations_count', 0),
            selected_pids_count=0,
            total_facts_count=0,
            relation_selection_reasoning=discovery_result.get('reasoning', 'No relations to select.'),
            fact_extraction_reasoning="No facts to extract.",
            verbose=getattr(args, 'kg_verbose_report', False)
        )
        return "SUCCESS", report

    # === 2.1: Sequential & Batched Fact Retrieval ===
    retrieved_facts_by_pid = {}
    all_relations_map = {rel['pid']: rel for rel in all_relations.get('head', []) + all_relations.get('tail', [])}

    for pid in selected_pids:
        # Initialize the structure for this PID
        retrieved_facts_by_pid[pid] = {
            "label": all_relations_map.get(pid, {}).get('label', 'N/A'),
            "outgoing_entities": [],
            "incoming_entities": [],
            "literal_values": []
        }
        
        # API Call 1: Get connected entities
        entities_result = client.query_all('get_tail_entities_given_head_and_relation', entity_qid, pid)
        if isinstance(entities_result, dict):
            retrieved_facts_by_pid[pid]['incoming_entities'].extend([e['label'] for e in entities_result.get('head', [])])
            retrieved_facts_by_pid[pid]['outgoing_entities'].extend([e['label'] for e in entities_result.get('tail', [])])
            
        # API Call 2: Get literal values
        values_result = client.query_all('get_tail_values_given_head_and_relation', entity_qid, pid)
        if isinstance(values_result, list):
            retrieved_facts_by_pid[pid]['literal_values'].extend(values_result)

        current_facts = retrieved_facts_by_pid[pid]
        current_pid_fact_count = len(current_facts['outgoing_entities']) + len(current_facts['incoming_entities']) + len(current_facts['literal_values'])
        print(f"Retrieving facts for relation: {all_relations_map.get(pid, {}).get('label', pid)} ({pid}), total facts: {current_pid_fact_count}")

    total_facts_count = sum(len(v) for data in retrieved_facts_by_pid.values() for k, v in data.items() if k != 'label')
    print(f"Retrieved a total of {total_facts_count} raw facts.")

    # === 2.2: Unified Fact Extraction via LLM ===
    if total_facts_count == 0:
        print("No facts were retrieved for the selected relations. Skipping extraction.")
        extraction_result = {
            "reasoning": "No facts to extract.",
            "extracted_info": "None"
        }
    else:
        formatted_facts_str = format_facts_for_extraction_prompt(retrieved_facts_by_pid, entity_label, max_display_facts=args.max_display_facts)
        
        template_inputs = {
            'question': question,
            'analysis': analysis,
            'query': query,
            'entity_label': entity_label,
            'formatted_facts': formatted_facts_str
        }
        
        # Use functools.partial to pass the retrieved_facts_by_pid to the parser for validation
        parser_with_context = functools.partial(parse_extraction_result, retrieved_facts_by_pid=retrieved_facts_by_pid)

        extraction_result = generate_process(
            step_name=f"Extract Facts for '{entity_label}'",
            prompt_template=prompt_extract_facts_unified,
            template_inputs=template_inputs,
            parsing_function=parser_with_context,
            args=args,
            module='kg',
            max_retries=max_retries,
            # response_format={"type": "json_object"}
            # max_tokens=2048 # Allow more tokens for the JSON output
        )
    
    if not extraction_result:
        reason = f"LLM failed to extract facts for '{entity_label}' after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", reason
        
    print(f"LLM Reasoning for extraction: {extraction_result.get('reasoning', 'N/A')}")
        
    # === 2.3: Integration & Report Generation ===
    extracted_info = extraction_result.get('extracted_info', 'None') or 'None'
    
    final_report = format_exploration_report(
        entity_mention=entity_mention,
        linked_candidates=linked_candidates,
        linked_entity=linked_entity,
        extracted_info=extracted_info,
        total_relations_count=discovery_result.get('total_relations_count', 0),
        selected_pids_count=len(selected_pids),
        total_facts_count=total_facts_count,
        relation_selection_reasoning=discovery_result.get('reasoning', 'N/A'),
        fact_extraction_reasoning=extraction_result.get('reasoning', 'N/A'),
        verbose=getattr(args, 'kg_verbose_report', False)
    )

    return "SUCCESS", final_report