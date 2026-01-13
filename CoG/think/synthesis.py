import re
from utils import generate_process, indent_text
from string import Template


prompt_synthesis_and_judgment = Template('''\
### ROLE
You are a master AI strategist leading a multi-hop question-answering mission. Your task is to synthesize retrieved information from various sources, evaluate progress against the overall plan, and decide the most logical next step.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Sub-Queries:** ${queries}
- **Core Entities in Sub-Queries:** ${entities}

### EVIDENCE
This section contains the information retrieved from different sources for all sub-queries executed in this turn.
${evidence_blocks}
### YOUR TASK
Carefully review all evidence, and in conjunction with the **Original Question** and your **Overall Plan**, complete the following three steps:
1.  **Synthesize and Extract:**  Consolidate all useful information from the "Evidence". Your goal is to capture every piece of potentially useful information. **Do not summarize or shorten the information.** Your extraction should include two types of information:
    - **Direct Facts:** Core facts that directly contribute to answering the original question.
    - **Promising Leads:** New entities or critical factual clues that do not answer the question directly but are essential for guiding the next step of the investigation.
    - **[CRITICAL RULE]** This section is for **extraction only**. Do NOT include your own reasoning, hypotheses, assumptions, or inferences here. All analytical thinking belongs in the "Think Step-by-Step" section.
2.  **Think Step-by-Step:** Document your thought process. 
    - Explain how the direct facts help answer the question.
    - Discuss the potential value of the promising leads and how they might be explored.
    - Crucially, identify what key information or critical links are still missing to form a complete answer.
    - If you conclude the evidence is useless, explain *why* the current sub-queries failed.
3.  **Make a Judgment:** Based on your analysis, make a clear judgment on the current progress. You must choose one of the following three options:
    - `SUFFICIENT`: The information is adequate to generate a final, complete answer.
    - `INSUFFICIENT_USEFUL`: Progress has been made and valuable clues have been found, but more information is needed. The investigation should continue based on the current findings.
    - `INSUFFICIENT_USELESS`: The information gathered in this round is irrelevant or has led to a dead end. A new strategy is needed, such as exploring different entities or query directions.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Thought Process: Your step-by-step thinking process. Explain how the extracted information helps answer the question and what is still missing.
Extracted Content: A structured collection of key facts and promising new leads from the evidence. Structure the output with 'Direct Facts:' and 'Promising Leads:' if applicable. If no useful information was found, state "None".
Judgment: SUFFICIENT, INSUFFICIENT_USEFUL, or INSUFFICIENT_USELESS.

### YOUR RESPONSE:
''')

prompt_extract_and_judgment = Template('''\
### ROLE
You are a master AI strategist leading a multi-hop question-answering mission. Your task is to synthesize retrieved information from various sources, evaluate progress against the overall plan, and decide the most logical next step.

### CONTEXT
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Notebook (Summary of Known Facts):**
${notebook}
- **Current Sub-Queries:** ${queries}
- **Core Entities in Sub-Queries:** ${entities}

### EVIDENCE
This section contains the information retrieved from different sources for all sub-queries executed in this turn.
${evidence_blocks}
### YOUR TASK
Carefully review all evidence, and in conjunction with the **Original Question** and your **Overall Plan**, complete the following three steps:
1.  **Verbatim Information Extraction:** Your task is to meticulously extract all useful information from the "Evidence" section. Your primary goal is to faithfully transfer all potentially relevant information from the 'Evidence' section to your 'Extracted Content'. Your extraction should include two types of information:
    - **Direct Facts:** Core facts that directly contribute to answering the original question.
    - **Promising Leads:** New entities or critical factual clues that do not answer the question directly but are essential for guiding the next step of the investigation.
    - **[CRITICAL]** Do NOT summarize, rephrase, or shorten the information. Your extracted output will be used as direct input for future automated queries, so precision is paramount.
    - **[CRITICAL]** This section is for **extraction only**. Do NOT include your own reasoning, hypotheses, assumptions, or inferences here. All analytical thinking belongs in the "Think Step-by-Step" section.
2.  **Think Step-by-Step:** Document your thought process. 
    - Explain how the direct facts help answer the question.
    - Discuss the potential value of the promising leads and how they might be explored.
    - Crucially, identify what key information or critical links are still missing to form a complete answer.
    - If you conclude the evidence is useless, explain *why* the current sub-queries failed.
3.  **Make a Judgment:** Based on your analysis, make a clear judgment on the current progress. You must choose one of the following three options:
    - `SUFFICIENT`: The information is adequate to generate a final, complete answer.
    - `INSUFFICIENT_USEFUL`: Progress has been made and valuable clues have been found, but more information is needed. The investigation should continue based on the current findings.
    - `INSUFFICIENT_USELESS`: The information gathered in this round is irrelevant, repetitive or has led to a dead end. A new strategy is needed, such as exploring different entities or query directions.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Thought Process: Your step-by-step thinking process. Explain how the extracted information helps answer the question and what is still missing.
Extracted Content: A structured collection of key facts and promising new leads from the evidence. Structure the output with 'Direct Facts:' and 'Promising Leads:' if applicable. If no useful information was found, state "None".
Judgment: SUFFICIENT, INSUFFICIENT_USEFUL, or INSUFFICIENT_USELESS.

### YOUR RESPONSE:
''')

'''
    - `INSUFFICIENT_USELESS`: The information gathered in this round meets **any** of the following conditions:
        - **【+】(a) Irrelevant:** The information is irrelevant to the question.
        - **【+】(b) Redundant:** No new, meaningful information was acquired compared to what was already known.
 or **fails to provide any new insights beyond what is already known**.

 - **【+】Critically Compare with Previous Turns:** Did you acquire any **new, non-redundant** information in this turn? If you are stuck getting the same information repeatedly, explicitly state this.
- **【+】Attribute Failure:** If you conclude the evidence is insufficient or useless, explain *why* the current sub-queries failed. Was it the query phrasing? The entity choice? Or a **fundamental limitation of the tool** (e.g., inability to parse tables, access a certain website)? This analysis is crucial for your judgment.
 '''

def format_evidence_block(query, entity, kg_result, wikipedia_result):
    """Formats a single block of evidence for the synthesis prompt."""
    return f"""
---
#### Evidence for Sub-Query: "{query}" (Entity: "{entity}")

##### Source 1: Knowledge Graph Exploration
{kg_result}

##### Source 2: Wikipedia Search
{wikipedia_result}
---
"""

def parse_synthesis_and_judgment_result(result_text: str):
    """
    Parses the LLM's output to extract the synthesis, thought process, and judgment.
    """
    try:
        extracted_content_match = re.search(r"Extracted Content:(.*?)Judgment:", result_text, re.DOTALL)
        thought_process_match = re.search(r"Thought Process:(.*?)Extracted Content:", result_text, re.DOTALL)
        # Use a more flexible regex to capture the entire line for the judgment.
        judgment_match = re.search(r"Judgment:(.*)", result_text)

        if not (extracted_content_match and thought_process_match and judgment_match):
            print("Parsing Error: Could not find one or more required fields in the output.")
            print(f"Full Response:\n{result_text}")
            return None

        extracted_content = extracted_content_match.group(1).strip()
        thought_process = thought_process_match.group(1).strip()
        
        # Robustly parse the judgment value.
        judgment_raw = judgment_match.group(1).strip()
        # Normalize by removing quotes/punctuation, converting to uppercase, and replacing spaces.
        judgment_normalized = judgment_raw.strip('\'"., ').upper().replace(" ", "_")
        if judgment_normalized != judgment_raw:
            print(f"Judgment value replaced: '{judgment_raw}' -> '{judgment_normalized}'")

        if judgment_normalized not in ['SUFFICIENT', 'INSUFFICIENT_USEFUL', 'INSUFFICIENT_USELESS']:
            print(f"Parsing Warning: Invalid judgment value '{judgment_raw}'. Normalized to '{judgment_normalized}', which is not valid. Will attempt retry.")
            return None
            
        return {
            "extracted_content": extracted_content,
            "thought_process": thought_process,
            "judgment": judgment_normalized
        }

    except Exception as e:
        print(f"An exception occurred during parsing: {e}\nInput text was: {result_text}")
        return None


def run_synthesis_and_judgment(question, analysis, notebook, current_queries, current_entities, turn_results, args, max_retries=3):
    """
    Runs the information synthesis and judgment step based on a full turn of evidence.

    Args:
        question (str): The original question.
        analysis (str): The current analysis of the problem.
        notebook (str): The current content of the notebook.
        current_queries (list): The list of sub-queries for the current turn.
        current_entities (list): The list of entities for the current turn.
        turn_results (list): A list of dictionaries for each sub-query.
        args: Arguments containing model configurations.
        max_retries (int): The maximum number of retries for the LLM call.

    Returns:
        tuple: A tuple containing (status, result_dict).
    """
    print("--- Running Synthesis and Judgment ---")

    evidence_blocks = ""
    for result in turn_results:
        evidence_blocks += format_evidence_block(
            query=result['query'],
            entity=result['entity'],
            kg_result=result['kg_result'],
            wikipedia_result=result['wiki_result']
        )
    print('retrieved evidence:', evidence_blocks)
    
    # Indent the notebook content for proper prompt formatting.
    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."
    
    template_inputs = {
        'question': question,
        'analysis': analysis,
        'notebook': indented_notebook,
        'queries': str(current_queries),
        'entities': str(current_entities),
        'evidence_blocks': evidence_blocks
    }
    synthesis_result = generate_process(
        step_name="Synthesize and Judge Evidence",
        prompt_template=prompt_extract_and_judgment,
        template_inputs=template_inputs,
        parsing_function=parse_synthesis_and_judgment_result,
        args=args,
        module='main',
        max_retries=max_retries,
        # max_tokens=2000
    )

    if not synthesis_result:
        reason = "LLM failed to synthesize and judge the evidence after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    synthesis_result['evidence_blocks'] = evidence_blocks
    print(f"LLM Judgment: {synthesis_result['judgment']}")
    print(f"LLM Thought Process: {synthesis_result['thought_process']}")
    print(f"LLM Extracted Content: {synthesis_result['extracted_content']}")

    return "SUCCESS", synthesis_result