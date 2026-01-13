from string import Template
from utils import generate_process

# =================================================================
# 1. Prompt Template for Direct CoT Answer (Final Fallback)
# =================================================================

prompt_direct_cot = Template('''\
### ROLE
You are an intelligent assistant. Your task is to answer the following question based on your internal knowledge.

### TASK
Please provide a detailed, step-by-step reasoning to answer the question. After the reasoning, state the final answer clearly.

### EXAMPLES
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

### QUESTION
${question}

### YOUR RESPONSE:
''')


# =================================================================
# 2. Main Workflow Function to Run Direct CoT Answer
# =================================================================

def run_direct_cot_answer(question, args, max_retries=3):
    """
    Runs the final fallback, generating an answer using a simple CoT prompt
    without requiring a specific format.
    """
    print("--- Running Direct CoT Answer (Final Fallback) ---")

    template_inputs = {'question': question}

    # We use generate_process but without a parsing function,
    # so it will return the raw LLM output.
    llm_output = generate_process(
        step_name="Final Fallback Generation",
        prompt_template=prompt_direct_cot,
        template_inputs=template_inputs,
        parsing_function=lambda x: x,  # Pass-through function
        args=args,
        module='main',
        max_retries=max_retries,
    )
    
    if not llm_output:
        # This is the ultimate failure point.
        return "Failed to generate any response from the language model."

    return llm_output