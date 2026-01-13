import re
from string import Template
from utils import generate_process, indent_text

# =================================================================
# 1. Prompt Template for Failure Conclusion
# =================================================================

prompt_failure_conclusion = Template('''\
### ROLE
You are a master AI strategist and system analyst. Your mission is to provide a comprehensive analysis of a failed multi-hop question-answering workflow, summarize any partial findings, and provide a best-effort answer.

### CONTEXT
- **Original Question:** ${question}
- **Notebook (all accumulated facts before failure):**
${notebook}
- **Full Interaction History:**
${interaction_history}

### FAILURE DETAILS
The automated investigation was forcefully terminated due to a critical internal error.
- **Failed Module:** ${failed_module}
- **Reason for Failure:** ${reason}

### YOUR TASK
As the final analyst, you must summarize the state of the investigation, provide the most informed answer possible based on the incomplete evidence, and diagnose the root cause of the failure.

1.  **Construct Final Thought Process:**
    - Briefly acknowledge the reason for termination (failed module and reason).
    - Summarize the key facts that were successfully gathered and stored in the **"Notebook"**.
    - Clearly state what critical information is still missing to fully answer the **"Original Question"**.
    - Explain how the gathered facts relate to the original question, even if they don't fully answer it.

2.  **Generate Best-Effort Final Answer:**
    - Based on your thought process, provide the most complete answer possible with the available information.
    - If no relevant information was found at all, state that you cannot answer the question.

3.  **Analyze Root Cause of Failure:**
    - This is the most critical part. Based on the **"Full Interaction History"** and the specific **"FAILURE DETAILS"**, analyze and explain why the investigation failed.
    - Consider multiple possibilities: Was it a flawed strategic plan from the start? A poorly chosen entity? A limitation of the knowledge sources? Or a persistent issue with the LLM's ability to follow structured output formats, leading to parsing errors?
    - Your analysis should be insightful and provide actionable feedback for improving the system.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Final Thought Process: Your complete, step-by-step reasoning for the partial answer.
Final Answer: The best-effort, potentially partial, answer.
Root Cause Analysis: Your detailed analysis of why the process failed.

### YOUR RESPONSE:
''')

# =================================================================
# 2. Parsing Function for the Failure Conclusion
# =================================================================

def parse_failure_conclusion_result(result_text: str):
    """
    Parses the LLM's output for the failure conclusion.
    """
    try:
        thought_process_match = re.search(r"Final Thought Process:(.*?)Final Answer:", result_text, re.DOTALL)
        final_answer_match = re.search(r"Final Answer:(.*?)Root Cause Analysis:", result_text, re.DOTALL)
        analysis_match = re.search(r"Root Cause Analysis:(.*)", result_text, re.DOTALL)

        if not (thought_process_match and final_answer_match and analysis_match):
            print("Parsing Error: Could not find one or more required fields in the failure conclusion output.")
            print(f"Full Response:\\n{result_text}")
            return None

        final_thought_process = thought_process_match.group(1).strip()
        final_answer = final_answer_match.group(1).strip()
        root_cause_analysis = analysis_match.group(1).strip()

        return {
            "final_thought_process": final_thought_process,
            "final_answer": final_answer,
            "root_cause_analysis": root_cause_analysis,
        }
    except Exception as e:
        print(f"An unexpected exception occurred during failure conclusion parsing: {e}\\nInput text was: {result_text}")
        return None

# =================================================================
# 3. Main Workflow Function to Run Failure Conclusion
# =================================================================

def run_failure_conclusion(question, notebook, interaction_history, failure_details, args, max_retries=3):
    """
    Runs the final conclusion stage when the workflow has failed.
    """
    print("--- Running Failure Conclusion Generation ---")

    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."

    template_inputs = {
        'question': question,
        'notebook': indented_notebook,
        'interaction_history': str(interaction_history),
        'failed_module': failure_details.get('module', 'N/A'),
        'reason': failure_details.get('reason', 'N/A'),
        # 'llm_output': failure_details.get('llm_output', 'N/A'), # TODO: add this
        # 'error_log': failure_details.get('error_log', 'N/A'), # TODO: add this
    }

    conclusion_result = generate_process(
        step_name="Generate Failure Conclusion",
        prompt_template=prompt_failure_conclusion,
        template_inputs=template_inputs,
        parsing_function=parse_failure_conclusion_result,
        args=args,
        module='main',
        max_retries=max_retries,
    )
    
    if not conclusion_result:
        reason = "LLM failed to generate a valid failure conclusion after multiple retries."
        print(f"!!! [WORKFLOW CRITICAL] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    print("\n--- Failure Conclusion Generated ---")
    print(f"Final Thought Process: {conclusion_result['final_thought_process']}")
    print(f"Final Answer: {conclusion_result['final_answer']}")
    print(f"Root Cause Analysis: {conclusion_result['root_cause_analysis']}")

    return "SUCCESS", conclusion_result