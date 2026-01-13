import json
from string import Template
import functools
import math

from utils import generate_process
from kg.client import MultiServerWikidataQueryClient

# =========================================================================================
# STAGE 2: Fact Retrieval & Fine-Grained Pruning
# =========================================================================================

# --- 2.1: Prompt Template for Unified Fact Pruning ---

prompt_prune_facts_unified = Template('''\
### ROLE
You are an expert assistant specializing in knowledge graph analysis for question answering. Your mission is to intelligently prune a list of facts, guiding a multi-step exploration process.

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
Your goal is to decide which facts to keep. This is a crucial step to guide the next phase of our investigation. Evaluate the facts using a two-pronged approach:
1. **Directly Relevant Facts**: Keep any facts that directly help answer the **Current Sub-Query**.
2. **Promising Intermediate Entities for Further Exploration**: Keep facts that represent entities which, while not direct answers, are promising stepping stones for the investigation. A fact is "promising" if exploring it further is highly likely to:
  - Lead to the answer for the **Current Sub-Query**.
  - Provide crucial context or clues for solving the **Original Question**.
Discard facts that are clearly irrelevant, noisy, or are just general information.

### OUTPUT FORMAT
Your output **MUST** be a single, valid JSON object, with no additional commentary before or after it.
The JSON object must have the following structure:
{
  "reasoning": "Your thought process, structured in three parts. 1) Briefly summarize the types of facts retrieved for the entity. 2) Explicitly justify kept facts by grouping them into (a) Directly Relevant to the Current Sub-Query and (b) Promising Intermediate Entities for further exploration, explaining how they could lead to the answer or add crucial context for the Original Question. 3) State why other facts were discarded (irrelevant/noisy/general).,
  "pruned_facts": {
    "outgoing": {
      "P17": ["list of kept fact labels"],
      "...": []
    },
    "incoming": {
      "P802": ["list of kept fact labels"],
      "...": []
    }
  }
}

- The "pruned_facts" object must contain two keys: "outgoing" and "incoming", corresponding to the categories in the "AVAILABLE FACTS" section.
- Each of these keys holds an object where:
  - The keys are the Relation ID from the "AVAILABLE FACTS" section (e.g., "P17", "P802").
  - The values are lists of strings, where each string is a fact label you chose to keep.
  - If you decide to discard all facts for a specific relation, you can either omit the relation ID key entirely or keep the key with an empty list [] as its value.
  - You MUST only include fact labels that were originally provided for that specific PID and direction.

### YOUR RESPONSE:
''')



# --- 2.2: Helper Functions for Formatting, Parsing, and Reporting ---

def format_facts_for_pruning_prompt(
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


def parse_pruning_result(result_text: str, retrieved_facts_by_pid: dict):
    """
    Parses the LLM's JSON output for fact pruning, including robust validation and anti-hallucination checks.
    """
    try:
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            print(f"Parsing Error: Could not find a valid JSON object in the output.\nInput text was: {result_text}")
            return None
        
        # 1. Parse JSON
        json_str = result_text[start_idx : end_idx + 1]
        data = json.loads(json_str)
        
        # 2. Structure Validation
        if not isinstance(data, dict) or not all(k in data for k in ["reasoning", "pruned_facts"]):
            print("Parsing Error: Missing 'reasoning' or 'pruned_facts' keys in the root.")
            return None
        
        pruned_facts = data["pruned_facts"]
        # if not isinstance(pruned_facts, dict) or not all(k in pruned_facts for k in ["outgoing", "incoming"]):
        if not isinstance(pruned_facts, dict):
            print("Parsing Error: Invalid pruned facts.")
            return None
            
    except json.JSONDecodeError:
        print(f"Parsing Error: Failed to decode JSON from LLM output.\nInput text was: {result_text}")
        return None
        
    # 3. Content Validation (Anti-Hallucination)
    for direction in ["outgoing", "incoming"]:
        # 安全地获取方向字典，如果LLM省略了空的方向，则视为空字典
        pids_dict = pruned_facts.get(direction, {})
        if not isinstance(pids_dict, dict):
            print(f"Parsing Error: pruned_facts['{direction}'] is not a dictionary.")
            return None

        for pid, kept_facts in pids_dict.items():
            if pid not in retrieved_facts_by_pid:
                print(f"Parsing Warning: LLM returned a PID '{pid}' that was not in the original selection. This is a hallucination.")
                return None
            
            if not isinstance(kept_facts, list):
                print(f"Parsing Error: The value for PID '{pid}' is not a list.")
                return None
            
            # 为当前处理的PID和方向，创建一个特定的、小范围的有效事实集合
            original_pid_facts = retrieved_facts_by_pid[pid]
            valid_facts_for_direction = set()
            if direction == "outgoing":
                valid_facts_for_direction.update(original_pid_facts.get('outgoing_entities', []))
                valid_facts_for_direction.update(original_pid_facts.get('literal_values', []))
            elif direction == "incoming":
                valid_facts_for_direction.update(original_pid_facts.get('incoming_entities', []))
                
            # 逐一检查保留的事实是否在正确的原始集合中
            for fact in kept_facts:
                if fact not in valid_facts_for_direction:
                    facts_list = list(valid_facts_for_direction)
                    max_display_facts = 200
                    if len(facts_list) > max_display_facts:
                        display_facts = facts_list[:max_display_facts]
                        remaining_count = len(facts_list) - max_display_facts
                        valid_facts_str = f"{display_facts} ... ({remaining_count} more)"
                    else:
                        valid_facts_str = str(facts_list)
                    print(f"Parsing Warning: LLM returned a fact '{fact}' for PID '{pid}' in the '{direction}' direction that was not in the original retrieved list. This is a hallucination. The valid facts for this direction are:\n{valid_facts_str}")
                    return None
                    
    # 4. If all checks pass, return the successfully parsed and validated data
    print("Successfully parsed and validated the pruned facts from LLM.")
    return data


def format_exploration_report(
    entity_mention: str,
    linked_candidates: list,
    linked_entity: dict,
    final_facts: dict,
    total_relations_count: int,
    selected_pids_count: int,
    total_facts_count: int,
    kept_facts_count: int,
    relation_selection_reasoning: str,
    fact_pruning_reasoning: str,
    max_candidates_display: int = 4
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
    if linked_candidates:
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
    if rationale:
        rationale_single_line = rationale.replace('\n', ' ')
        linking_section.append(f'- Linking Rationale: {rationale_single_line}')
    
    report_parts.append("\n".join(linking_section))
    
    # 3. 构建 [Retrieved Facts for "..."] 模块 (保留空行)
    report_parts.append(f'\n[Retrieved Facts for "{linked_entity.get("label")}"]')

    # Helper to format each section
    def format_section(title, data):
        lines = [f"[{title}]"]
        if not data:
            lines.append("- None")
            return "\n".join(lines)
            
        for label, values in data.items():
            lines.append(f"- {label}: {str(values)}")
            # lines.append(f"- {label}: {', '.join(map(str, values))}")
        return "\n".join(lines)

    report_parts.append(format_section("Attributes", final_facts["attributes"]))
    report_parts.append(format_section("Outgoing", final_facts["outgoing"]))
    report_parts.append(format_section("Incoming", final_facts["incoming"]))

    # Add the summary
    summary_section = [
        "[Filtering Summary]",
        f"- Relations explored: {selected_pids_count} of {total_relations_count} total",
        f"- Facts kept: {kept_facts_count} of {total_facts_count} total"
    ]
    report_parts.append("\n".join(summary_section))
    
    # Add the reasoning section
    # First, replace newlines to avoid f-string syntax errors and for cleaner output
    rel_reason_single_line = relation_selection_reasoning.replace('\n', ' ')
    fact_reason_single_line = fact_pruning_reasoning.replace('\n', ' ')
    
    reasoning_section = [
        "[Reasoning]",
        f"- Relation Selection: {rel_reason_single_line}",
        f"- Fact Pruning: {fact_reason_single_line}"
    ]
    report_parts.append("\n".join(reasoning_section))
    
    return "\n".join(report_parts)


# --- 2.3: Main Orchestrator Function for Stage 2 ---

def run_fact_retrieval_and_pruning(
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
    Executes Stage 2: Fact Retrieval, Fine-Grained Pruning, and Report Generation.
    """
    entity_qid = linked_entity.get('qid')
    entity_label = linked_entity.get('label')
    selected_pids = discovery_result.get('selected_pids', [])
    all_relations = discovery_result.get('all_relations', {})
    
    print(f"\n--- Stage 2: Fact Retrieval & Pruning for '{entity_label}' ({len(selected_pids)} relations) ---")
    
    if not selected_pids:
        print("No relations were selected in Stage 1. Skipping fact retrieval.")
        # Create an empty report
        empty_final_facts = {"attributes": {}, "outgoing": {}, "incoming": {}}
        report = format_exploration_report(
            entity_mention=entity_mention,
            linked_candidates=linked_candidates,
            linked_entity=linked_entity,
            final_facts=empty_final_facts,
            total_relations_count=discovery_result.get('total_relations_count', 0),
            selected_pids_count=0,
            total_facts_count=0,
            kept_facts_count=0,
            relation_selection_reasoning=discovery_result.get('reasoning', 'No relations to select.'),
            fact_pruning_reasoning="No facts to prune."
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

    # === 2.2: Unified Fact Pruning via LLM ===
    if total_facts_count == 0:
        print("No facts were retrieved for the selected relations. Skipping pruning.")
        pruning_result = {
            "reasoning": "No facts to prune.",
            "pruned_facts": {"outgoing": {}, "incoming": {}}
        }
    else:
        formatted_facts_str = format_facts_for_pruning_prompt(retrieved_facts_by_pid, entity_label, max_display_facts=args.max_display_facts)
        
        template_inputs = {
            'question': question,
            'analysis': analysis,
            'query': query,
            'entity_label': entity_label,
            'formatted_facts': formatted_facts_str
        }
        
        # Use functools.partial to pass the retrieved_facts_by_pid to the parser for validation
        parser_with_context = functools.partial(parse_pruning_result, retrieved_facts_by_pid=retrieved_facts_by_pid)

        pruning_result = generate_process(
            step_name=f"Prune Facts for '{entity_label}'",
            prompt_template=prompt_prune_facts_unified,
            template_inputs=template_inputs,
            parsing_function=parser_with_context,
            args=args,
            module='kg',
            max_retries=max_retries,
            # response_format={"type": "json_object"}
            # max_tokens=2048 # Allow more tokens for the JSON output
        )
    
    if not pruning_result:
        reason = f"LLM failed to prune facts for '{entity_label}' after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", reason
        
    print(f"LLM Reasoning for pruning: {pruning_result.get('reasoning', 'N/A')}")
        
    # === 2.3: Integration & Report Generation ===
    pruned_facts = pruning_result.get('pruned_facts', {})
    final_facts = {"attributes": {}, "outgoing": {}, "incoming": {}}
    kept_facts_count = 0
    
    # 新的、更健壮的分类逻辑
    for direction, pids_dict in pruned_facts.items():
        if direction not in ['outgoing', 'incoming']:
            continue
            
        for pid, kept_values in pids_dict.items():
            if not kept_values:
                continue
            
            # 累加总数
            kept_facts_count += len(kept_values)
            relation_label = retrieved_facts_by_pid[pid]['label']
            
            # 分类逻辑的核心
            if direction == 'incoming':
                # 入度关系下的事实总是实体，直接归类
                final_facts['incoming'][relation_label] = kept_values
            
            elif direction == 'outgoing':
                # 对于出度关系，需要根据事实本身的类型细分
                original_pid_facts = retrieved_facts_by_pid[pid]
                original_literals = set(original_pid_facts.get('literal_values', []))
                
                # 将保留的事实分为字面量(attributes)和实体(outgoing entities)
                kept_literals = [val for val in kept_values if val in original_literals]
                kept_entities = [val for val in kept_values if val not in original_literals]
                
                # 如果有保留的字面量，放入 'attributes'
                if kept_literals:
                    final_facts['attributes'][relation_label] = kept_literals
                
                # 如果有保留的实体，放入 'outgoing'
                if kept_entities:
                    final_facts['outgoing'][relation_label] = kept_entities
    
    print(f"Kept {kept_facts_count} facts after pruning.")

    final_report = format_exploration_report(
        entity_mention=entity_mention,
        linked_candidates=linked_candidates,
        linked_entity=linked_entity,
        final_facts=final_facts,
        total_relations_count=discovery_result.get('total_relations_count', 0),
        selected_pids_count=len(selected_pids),
        total_facts_count=total_facts_count,
        kept_facts_count=kept_facts_count,
        relation_selection_reasoning=discovery_result.get('reasoning', 'N/A'),
        fact_pruning_reasoning=pruning_result.get('reasoning', 'N/A')
    )

    return "SUCCESS", final_report