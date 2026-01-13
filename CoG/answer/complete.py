import re
from string import Template
from utils import generate_process, indent_text


# =================================================================
# 1. Prompt Template & Parsing for Final Answer Generation
# =================================================================

prompt_answer_generation = Template('''\
### ROLE
You are a master AI strategist and synthesizer. Your mission is to provide a final, comprehensive answer to a complex multi-hop question, based on all the evidence gathered.

### CONTEXT
- **Original Question:** ${question}
- **Notebook (accumulated known facts):**
${notebook}

### Key Evidence from Final Turn
- **Overall Plan (Analysis):** ${analysis}
- **Sub-Queries:** ${queries}
- **Entities in Sub-Queries:** ${entities}
- **Extracted Content:**
${extracted_content}
- **Concluding Thought Process:**
${thought_process}

### YOUR TASK
Based on all the provided context and evidence, generate a final, definitive answer to the original question. Your answer must be self-contained and directly address the user's query.

1.  **Construct Final Thought Process:** Create a clear, step-by-step reasoning explaining how you arrived at the final answer. You must synthesize information from both the **"Notebook"** and the **"Key Evidence from Final Turn"**. Show how these facts connect to directly address every part of the **"Original Question"**. This is your final reasoning narrative.
2.  **Generate Final Answer:** After your reasoning, provide a concise and direct answer to the question.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Final Thought Process: Your complete, step-by-step reasoning.
Final Answer: The final, direct answer to the question.

### YOUR RESPONSE:
''')

'''2.  **Extract Final Answer:** Based on your reasoning, provide the definitive answer entity or value.
    - **Crucial Rule:** Do **NOT** write a full sentence.
    - **Content:** Provide **only** the name(s) or value(s) that directly answer the question.
    - **Multiple Answers:** If there are several valid names, aliases, or answers (like in the webqsp dataset), list them and separate them with a comma.
2.  **Generate Final Answer:** After your reasoning, provide a concise and direct answer. **Crucially, the answer must contain only the core information (e.g., a name, location, date, number, or short phrase) without forming a complete sentence or repeating the question.**
2.  **Generate Final Answer:** After your reasoning, provide a concise and direct answer. **The answer should be the core entity, name, or value, without any surrounding sentences or rephrasing of the question.**
### EXAMPLE OF A VALID RESPONSE
Final Thought Process: 1. The original question asks for the CEO of the company that developed the Windows operating system. 2. The notebook confirms that Microsoft is the developer of Windows. 3. The evidence from the final turn states that Satya Nadella is the CEO of Microsoft. 4. By linking these two pieces of information, I can definitively conclude that Satya Nadella is the correct answer.
Final Answer: Satya Nadella
'''

def parse_answer_generation_result(result_text: str):
    """
    Parses the LLM's output for the final answer generation.
    """
    try:
        thought_process_match = re.search(r"Final Thought Process:(.*?)Final Answer:", result_text, re.DOTALL)
        final_answer_match = re.search(r"Final Answer:(.*)", result_text, re.DOTALL)

        if not (thought_process_match and final_answer_match):
            print("Parsing Error: Could not find 'Final Thought Process' or 'Final Answer' in the output.")
            print(f"Full Response:\\n{result_text}")
            return None

        final_thought_process = thought_process_match.group(1).strip()
        final_answer = final_answer_match.group(1).strip()

        return {
            "final_thought_process": final_thought_process,
            "final_answer": final_answer,
        }
    except Exception as e:
        print(f"An unexpected exception occurred during answer parsing: {e}\\nInput text was: {result_text}")
        return None

# =================================================================
# 2. Main Workflow Function to Run the Answer Generation Step
# =================================================================

def run_answer_generation(question, analysis, notebook, queries, entities, extracted_content, thought_process, args, max_retries=3):
    """
    Runs the final answer generation stage when information is sufficient.
    """
    print("--- Running Final Answer Generation ---")
    
    indented_notebook = indent_text(notebook, "  ", use_indent=args.use_indent) if notebook else "The notebook is currently empty."
    
    template_inputs = {
        'question': question,
        'analysis': analysis,
        'notebook': indented_notebook,
        'queries': str(queries),
        'entities': str(entities),
        'extracted_content': extracted_content,
        'thought_process': thought_process,
    }

    answer_result = generate_process(
        step_name="Generate Final Answer",
        prompt_template=prompt_answer_generation,
        template_inputs=template_inputs,
        parsing_function=parse_answer_generation_result,
        args=args,
        module='main',
        max_retries=max_retries,
    )
    if not answer_result:
        reason = "LLM failed to generate a valid final answer after multiple retries."
        print(f"!!! [WORKFLOW FAILED] {reason} !!!")
        return "FAILED", {"reason": reason}
    
    print("\n--- Final Answer Generated ---")
    print(f"Final Thought Process: {answer_result['final_thought_process']}")
    print(f"Final Answer: {answer_result['final_answer']}")

    return "SUCCESS", answer_result