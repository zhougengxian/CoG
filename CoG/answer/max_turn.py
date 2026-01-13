import re
from string import Template
from utils import generate_process, indent_text

# =================================================================
# 1. Prompt Template for MAX_TURNS Conclusion
# =================================================================


prompt_answer_generation_max_turns = Template('''\
### ROLE
You are a master AI strategist and synthesizer. Your mission is to provide a "best-effort" summary and partial answer to a complex question because has reached its predefined exploration limit ${max_turns}.

### CONTEXT
- **Original Question:** ${question}
- **Notebook (all accumulated facts):**
${notebook}
- **Full Interaction History:**
${interaction_history}

### YOUR TASK
The automated investigation has ended without reaching a definitive conclusion. Your task is to act as a final analyst, summarizing the state of the investigation and providing the most informed answer possible based on the incomplete evidence.

1.  **Construct Final Thought Process:**
    - Briefly acknowledge the reason for termination.
    - Summarize the key facts that were successfully gathered and stored in the **"Notebook"**.
    - Clearly state what critical information is still missing to fully answer the **"Original Question"**.
    - Explain how the gathered facts relate to the original question, even if they don't fully answer it.

2.  **Generate Best-Effort Final Answer:**
    - Based on your thought process, provide the most complete answer possible with the available information.
    - If no relevant information was found at all, state that you cannot answer the question.

3.  **Analyze Incompletion:** Based on the "Full Interaction History", analyze and explain why the investigation failed to reach a definitive answer within the time limit. Consider the query paths taken, the complexity of the question, or potential dead ends that were encountered.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Final Thought Process: Your complete, step-by-step reasoning for the partial answer.
Final Answer: The best-effort, potentially partial, answer.
Analysis of Incompletion: Your analysis of why the process did not complete.

### YOUR RESPONSE:
''')


# =================================================================
# 2. Parsing Function for the MAX_TURNS Conclusion
# =================================================================

def parse_max_turns_conclusion_result(result_text: str):
    """
    Parses the LLM's output for the MAX_TURNS conclusion.
    """
    try:
        thought_process_match = re.search(r"Final Thought Process:(.*?)Final Answer:", result_text, re.DOTALL)
        final_answer_match = re.search(r"Final Answer:(.*?)Analysis of Incompletion:", result_text, re.DOTALL)
        analysis_match = re.search(r"Analysis of Incompletion:(.*)", result_text, re.DOTALL)

        if not (thought_process_match and final_answer_match and analysis_match):
            print("Parsing Error: Could not find one or more required fields in the MAX_TURNS output.")
            print(f"Full Response:\\n{result_text}")
            return None

        final_thought_process = thought_process_match.group(1).strip()
        final_answer = final_answer_match.group(1).strip()
        analysis_of_incompletion = analysis_match.group(1).strip()

        return {
            "final_thought_process": final_thought_process,
            "final_answer": final_answer,
            "analysis_of_incompletion": analysis_of_incompletion,
        }
    except Exception as e:
        print(f"An unexpected exception occurred during MAX_TURNS conclusion parsing: {e}\\nInput text was: {result_text}")
        return None

# =================================================================
# 3. Main Workflow Function to Run MAX_TURNS Conclusion
# =================================================================

def run_max_turns_conclusion(question, notebook, interaction_history, max_turns, args, max_retries=3):
    """
    Runs the final conclusion stage when the maximum number of turns is reached.
    """
    print("--- Running Max Turns Conclusion Generation ---")

    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."

    template_inputs = {
        'question': question,
        'notebook': indented_notebook,
        'interaction_history': str(interaction_history),
        'max_turns': max_turns,
    }

    conclusion_result = generate_process(
        step_name="Generate MAX_TURNS Conclusion",
        prompt_template=prompt_answer_generation_max_turns,
        template_inputs=template_inputs,
        parsing_function=parse_max_turns_conclusion_result,
        args=args,
        module='main',
        max_retries=max_retries,
    )
    
    if not conclusion_result:
        reason = "LLM failed to generate a valid conclusion after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    print("\n--- MAX_TURNS Conclusion Generated ---")
    print(f"Final Thought Process: {conclusion_result['final_thought_process']}")
    print(f"Final Answer: {conclusion_result['final_answer']}")
    print(f"Analysis of Incompletion: {conclusion_result['analysis_of_incompletion']}")

    return "SUCCESS", conclusion_result