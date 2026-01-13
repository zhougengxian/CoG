import re
import functools
from string import Template

from utils import generate_process

prompt_entity_link = Template('''\
### ROLE
You are an expert in Knowledge Graph entity linking. Your task is to disambiguate an entity mention from a user's question by matching it to the correct entity in a knowledge graph, using the surrounding context.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Query:** "${query}"
- **Entity Mention to Link:** "${entity}"

### CANDIDATE ENTITIES
Here are the top candidate entities in the Knowledge Graph, sorted by a preliminary relevance score. Each candidate includes its Wikidata QID, label, description, aliases, popularity, some of its neighbors, and relevance score.

${formatted_candidates}

### YOUR TASK
Your primary goal is to perform **Identity Linking**. You must determine which of the candidate entities is the most accurate real-world representation of the entity mention "${entity}", given the context of the question.

Your output MUST be a single line containing ONLY the QID of the entity that is the best match for the question's context.

If you are certain that NONE of the candidates are a good match, output the single word: NO_MATCH.

### YOUR RESPONSE:
''')



prompt_entity_link_advanced = Template('''\
### ROLE
You are an expert in Knowledge Graph entity linking. Your task is to disambiguate an entity mention from a user's question by matching it to the correct entity in a knowledge graph, using the surrounding context.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Query:** "${query}"
- **Entity Mention to Link:** "${entity}"

### CANDIDATE ENTITIES
Here are the top candidate entities in the Knowledge Graph, sorted by a preliminary relevance score. Each candidate includes its Wikidata QID, label, description, aliases, popularity, some of its neighbors, and relevance score.

${formatted_candidates}

### YOUR TASK
Your task is to critically evaluate the candidate entities based on the provided CONTEXT. Your goal is to either identify the single correct entity OR determine that no suitable match exists.

1.  **Analyze the Context:** Carefully review the 'Original Question' and 'Overall Plan'. What are the key details about the entity "${entity}" (e.g., their time period, relationships, role)?
2.  **Evaluate Each Candidate:** For each candidate, compare its `Description`, `Aliases`, and `Neighborhood` against the context. A correct match should be consistent with the context.
3.  **Make a Decision:**
    - **If, and only if,** you find one candidate that is a confident and accurate match for "${entity}" based on all available information, your output should be its QID on a single line.
    - **If none of the candidates are a confident match,** or if their key details (description, relationships, etc.) contradict the context, you MUST output the single word: NO_MATCH. Do not guess or select a "closest" but incorrect option.

**Output Format:**
- If a confident match is found, your entire output MUST be only the QID of that entity (e.g., Q12345).
- If no confident match is found, your entire output MUST be the single word: NO_MATCH.
- Your output must not contain any other text, explanation, or reasoning.

### YOUR RESPONSE:
''')


prompt_entity_link_analysis = Template('''\
### ROLE
You are an expert in Knowledge Graph entity linking. Your task is to disambiguate an entity mention from a user's question by matching it to the correct entity in a knowledge graph, using the surrounding context.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Query:** "${query}"
- **Entity Mention to Link:** "${entity}"

### CANDIDATE ENTITIES
Here are the top candidate entities in the Knowledge Graph, sorted by a preliminary relevance score. Each candidate includes its Wikidata QID, label, description, aliases, popularity, some of its neighbors, and relevance score.

${formatted_candidates}

### YOUR TASK
Your task is to critically evaluate the candidate entities based on the provided CONTEXT. Your goal is to either identify the single correct entity OR determine that no suitable match exists.

1.  **Analyze the Context:** Carefully review the 'Original Question' and 'Overall Plan' to understand the key characteristics of the entity "${entity}" being sought.
2.  **Summarize Candidates:** Briefly summarize the types of entities present in the candidate list.
3.  **Justify Your Decision:**
    - **If you find a match:** Provide a detailed justification for your choice. Explain how the chosen entity's details align with the context and why it is superior to other alternatives. You do not need to analyze every rejected candidate individually.
    - **If no match is found:** Explain why the most promising candidates are unsuitable or contradict the context, leading to your `NO_MATCH` decision. Remember: Do not guess or select a "closest" but incorrect option.
4.  **State Your Final Decision:** In the `Decision` field of your output, provide either the QID of the matched entity or the single word NO_MATCH.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary before or after it. Each key must be on a new line.

Analysis: Your brief reasoning. First, analyze the context to determine the key characteristics of the entity being sought. Second, briefly describe what types of candidates are available. Third, if you found a match, explain specifically why the selected entity fits the context and mention. If no match was found, explain what key information is missing or conflicting.
Decision: The QID of the single best entity (e.g., Q12345) OR the single word NO_MATCH if no entity is a confident match.

### YOUR RESPONSE:
''')


def format_candidate_entities(candidates):
    """
    Formats a list of candidate entities into a structured string for LLM processing,
    following the specified format.
    """
    if not candidates:
        return "No candidates found."

    formatted_texts = []
    for cand in candidates:
        # Entity section
        aliases = cand.get('alias')
        alias_str = ', '.join(aliases) if aliases else 'N/A'
        
        degree = cand.get('degree', {})
        
        entity_section = [
            "[Entity]",
            f"  - QID: {cand.get('qid', 'N/A')}",
            f"  - Label: {cand.get('label', 'N/A')}",
            f"  - Description: {cand.get('description', 'N/A')}",
            f"  - Aliases: {alias_str}",
            f"  - Popularity (Degree): In: {degree.get('in', 0)}, Out: {degree.get('out', 0)}, Attr: {degree.get('attr', 0)}"
        ]

        # Neighborhood section
        neighborhood_section = ["[Neighborhood]"]
        triplets = cand.get('triplets', {})
        head_triplets = triplets.get('head', {})
        tail_triplets = triplets.get('tail', {})

        if head_triplets:
            neighborhood_section.append("  - Outgoing Relations (as Head):")
            for rel, objs in head_triplets.items():
                if objs:
                    obj_str = ", ".join(map(str, objs))
                    neighborhood_section.append(f"    - {rel}: {obj_str}")
        
        if tail_triplets:
            neighborhood_section.append("  - Incoming Relations (as Tail):")
            for rel, subjs in tail_triplets.items():
                if subjs:
                    subj_str = ", ".join(map(str, subjs))
                    neighborhood_section.append(f"    - {rel}: {subj_str}")
        
        if not head_triplets and not tail_triplets:
            neighborhood_section.append("  - No relations found.")

        # Scores section
        scores = cand.get('score_details', {})
        scores_section = [
            "[Scores]",
            f"  - Final: {cand.get('final_score', 0.0):.4f} (Name similarity={scores.get('name_sim', 0.0):.3f}, Popularity={scores.get('pop_score', 0.0):.3f}, Description={scores.get('desc_context_sim', 0.0):.3f}, Neighbor={scores.get('triplet_context_sim', 0.0):.3f})"
        ]

        # Combine all sections
        full_text = "\n".join(entity_section + neighborhood_section + scores_section)
        formatted_texts.append(full_text)
    
    return "---\n" + "\n---\n".join(formatted_texts) + "\n---"


def format_candidates_compact(candidates):
    """
    Formats a list of candidate entities into a compact, structured, and clear string for LLM processing.
    """
    if not candidates:
        return "No candidates found."

    formatted_texts = []
    for cand in candidates:
        # Line 1: Identity
        identity_line = f"[{cand.get('label', 'N/A')}]({cand.get('qid', 'N/A')})"

        # Line 2: Description
        desc_line = f"Desc: {cand.get('description', 'N/A')}"

        # Line 3: Aliases
        aliases = cand.get('alias', [])
        alias_str = ', '.join(aliases) if aliases else 'N/A'
        aliases_line = f"Aliases: {alias_str}"

        # Line 4: Popularity
        degree = cand.get('degree', {})
        pop_line = (
            f"Popularity: In-degree={degree.get('in', 0)}, "
            f"Out-degree={degree.get('out', 0)}, "
            f"Attributes={degree.get('attr', 0)}"
        )

        # Line 5: Scores
        scores = cand.get('score_details', {})
        score_line = (
            f"Score: {cand.get('final_score', 0.0):.4f} "
            f"(Name-Sim:{scores.get('name_sim', 0.0):.3f}, Pop-Score:{scores.get('pop_score', 0.0):.3f}, "
            f"Desc-Sim:{scores.get('desc_context_sim', 0.0):.3f}, Neighbor-Sim:{scores.get('triplet_context_sim', 0.0):.3f})"
        )

        # Lines 6 & 7: Neighborhood
        triplets = cand.get('triplets', {})
        
        head_triplets = triplets.get('head', {})
        outgoing_rels = []
        if head_triplets:
            for rel, objs in head_triplets.items():
                if objs:
                    obj_str = ", ".join(map(str, objs))
                    outgoing_rels.append(f"{rel}: [{obj_str}]")
        outgoing_line = "Outgoing: " + ("; ".join(outgoing_rels) if outgoing_rels else "N/A")

        tail_triplets = triplets.get('tail', {})
        incoming_rels = []
        if tail_triplets:
            for rel, subjs in tail_triplets.items():
                if subjs:
                    subj_str = ", ".join(map(str, subjs))
                    incoming_rels.append(f"{rel}: [{subj_str}]")
        incoming_line = "Incoming: " + ("; ".join(incoming_rels) if incoming_rels else "N/A")
        
        # Combine all parts
        full_text = "\n".join([
            identity_line, desc_line, aliases_line, pop_line, score_line,
            outgoing_line, incoming_line
        ])
        formatted_texts.append(full_text)
    
    return "---\n" + "\n---\n".join(formatted_texts) + "\n---"


# --- 第 1 步: 定义智能的解析函数 ---
def parse_and_find_entity(result_text: str, candidates: list):
    """
    Parses the LLM's raw text output. If a valid QID is found, it immediately
    searches for the corresponding full entity details in the candidates list.

    This ensures that a hallucinated QID is treated as a parsing failure,
    allowing `generate_process` to retry.

    Args:
        result_text (str): The raw output from the LLM.
        candidates (list): The list of candidate entities shown to the LLM.

    Returns:
        - The full dictionary of the selected entity if successful.
        - The string "NO_MATCH" if the LLM explicitly states no match.
        - None if parsing fails or the QID is not in the candidate list (to trigger a retry).
    """
    stripped_text = result_text.strip().upper()
    
    # Case 1: LLM explicitly returns NO_MATCH
    if stripped_text == "NO_MATCH":
        return "NO_MATCH"
    
    # Case 2: LLM returns a QID
    match = re.match(r'^(Q\d+)$', stripped_text)
    if match:
        selected_qid = match.group(1)
        # Find the full entity details from the original candidate list
        final_entity_details = next((cand for cand in candidates if cand['qid'] == selected_qid), None)
        if final_entity_details is None:
            print(f"Parsing Warning: LLM returned a QID '{selected_qid}' that was not in the candidate list. Will attempt retry.")
        return final_entity_details
            
    # Case 3: LLM output is malformed
    print(f"Could not parse QID from disambiguation result: {result_text}")
    return None


def parse_entity_link_analysis_result(result_text: str, candidates: list):
    """
    Parses the LLM's output for entity linking (analysis version), which includes reasoning and a decision.
    Validates that the selected QID exists in the provided candidates.
    """
    try:
        analysis_match = re.search(r"Analysis:(.*?)(?=\nDecision:)", result_text, re.DOTALL)
        decision_match = re.search(r"Decision:(.*)", result_text, re.DOTALL)

        if not analysis_match or not decision_match:
            print(f"Parsing Error: Could not find 'Analysis' or 'Decision' in the output.")
            return None

        analysis = analysis_match.group(1).strip()
        decision = decision_match.group(1).strip().upper()
        
        # Case 1: NO_MATCH
        if decision == "NO_MATCH":
            return {
                "analysis": analysis,
                "entity": None
            }
            
        # Case 2: QID
        match = re.match(r'^(Q\d+)$', decision)
        if match:
            selected_qid = match.group(1)
            final_entity_details = next((cand for cand in candidates if cand['qid'] == selected_qid), None)
            
            if final_entity_details is None:
                print(f"Parsing Warning: LLM returned a QID '{selected_qid}' that was not in the candidate list. Will attempt retry.")
                return None
            
            return {
                "analysis": analysis,
                "entity": final_entity_details
            }

        # Case 3: Malformed decision
        print(f"Could not parse QID or NO_MATCH from decision: {decision}")
        return None

    except Exception as e:
        print(f"An exception occurred during parsing analysis result: {e}\nInput text was: {result_text}")
        return None


# --- 第 2 步: 更新主消歧函数 ---
def run_entity_disambiguation(question, analysis, query, mention, candidates, args, prompt_version="advanced"):
    """
    Uses an LLM via generate_process to disambiguate between multiple candidate entities.
    Now uses a parser that validates the QID against the candidate list.
    Can select different prompt versions.

    Args:
        question (str): The original user question.
        analysis (str): The analysis of the question provided by the LLM.
        query (str): The current sub-query being processed.
        mention (str): The entity mention to disambiguate.
        candidates (list): The list of candidate entities.
        args (argparse.Namespace): Command-line arguments.
        prompt_version (str): "simple", "advanced", or "analysis".

    Returns:
        tuple: A tuple containing:
            - str: The status ("SUCCESS", "NO_MATCH", "FAILED").
            - object: The result (a entity detail dict, or a list of candidates, or a reason string).
                      If using "analysis", the dict will also contain the 'analysis' from the LLM.
    """
    if not candidates:
        print(f"Entity recall failed: The mention '{mention}' did not return any candidate entities from the Knowledge Graph.")
        return ("FAILED", f"No candidate entities were recalled from the KG for the mention '{mention}'.")

    # # Heuristic to skip LLM call if one candidate is a clear winner
    # if len(candidates) == 1 or (len(candidates) > 1 and (candidates[0]['final_score'] - candidates[1]['final_score']) > 0.15):
    #     print(f"High-confidence candidate found ({candidates[0]['label']}). Skipping LLM disambiguation.")
    #     return ("SUCCESS", candidates[0])

    print(f"--- Disambiguating '{mention}' using LLM (version: {prompt_version}) with top {len(candidates)} candidates. ---")
    formatted_candidates = format_candidate_entities(candidates)
    # print(formatted_candidates)

    template_inputs = {
        'question': question,
        'analysis': analysis,
        'query': query,
        'entity': mention,
        'formatted_candidates': formatted_candidates
    }

    if prompt_version == "simple":
        prompt_template = prompt_entity_link
        parsing_function = functools.partial(parse_and_find_entity, candidates=candidates)
    elif prompt_version == "advanced":
        prompt_template = prompt_entity_link_advanced
        parsing_function = functools.partial(parse_and_find_entity, candidates=candidates)
    elif prompt_version == "analysis":
        prompt_template = prompt_entity_link_analysis
        parsing_function = functools.partial(parse_entity_link_analysis_result, candidates=candidates)
    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")
    
    result_object = generate_process(
        step_name=f"Disambiguate Entity '{mention}' (v: {prompt_version})",
        prompt_template=prompt_template,
        template_inputs=template_inputs,
        parsing_function=parsing_function,
        args=args,
        module='kg',
        max_retries=3,
        max_tokens=args.max_length_entity_link # Adjust this value for reasoning model
    )

    # --- Process the result from generate_process ---
    if prompt_version in ["simple", "advanced"]:
        if isinstance(result_object, dict): # Success case
            print(f"LLM selected entity: {result_object.get('label')} ({result_object.get('qid')}): {result_object.get('description')}\n")
            return ("SUCCESS", result_object)
        
        elif result_object == "NO_MATCH":
            print(f"--- LLM determined NO_MATCH for mention '{mention}'. ---")
            return ("NO_MATCH", {"analysis": None, "candidates": candidates})

    elif prompt_version == "analysis":
        if isinstance(result_object, dict):
            entity = result_object.get("entity")
            analysis_text = result_object.get("analysis", "N/A")
            print(f"LLM Analysis: {analysis_text}")

            if entity:
                # Add analysis to the returned entity object
                entity['analysis'] = analysis_text
                print(f"LLM selected entity: {entity.get('label')} ({entity.get('qid')}): {entity.get('description')}\n")
                return ("SUCCESS", entity)
            else:
                print(f"--- LLM determined NO_MATCH for mention '{mention}'. ---")
                return ("NO_MATCH", {"analysis": analysis_text, "candidates": candidates})
        
    # Fallback for failure
    reason = f"LLM-based disambiguation failed for mention '{mention}' after multiple retries."
    print(f"--- {reason} ---")
    return ("FAILED", reason)