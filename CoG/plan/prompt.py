from string import Template


init_plan_prompt = Template('''\
### ROLE
You are a highly strategic retrieval planner. Your function is to analyze a complex question and identify the immediate, parallelizable search actions required to proceed.

### TASK
Given a user's question, do NOT provide the final answer. Instead, determine the next logical search queries. A single question might require multiple parallel queries to gather the necessary initial information. Your plan must include your reasoning, a list of search queries, and a corresponding list of entities.

### CONSTRAINTS
- The number of queries in the `Query` list must not exceed 5.
- **[CRITICAL]** The `Query` and `Entities` lists must have the exact same number of items. For each query, the corresponding item in the `Entities` list must be the **single, most specific, and central named entity** being investigated in that query.
- Only output the query/queries for the immediate next step. Do not plan, describe, or allude to any subsequent steps.

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary:

Analysis: Your thought process explaining why these next steps are necessary and what information is being sought.
Query: [A Python-style list of concise and effective search query strings.]
Entities: [A Python-style list of the core entities in ths Query, corresponding one-to-one with the Query list.]

---
### EXAMPLES

**Example 1:**
**Question:** What is the nationality of one of the men for whom the Schumann–Runge bands are named?
**Output:**
Analysis: The question requires the nationality of a person. Before I can find their nationality, I must first identify the individuals the "Schumann–Runge bands" are named after.
Query: ["who are the Schumann-Runge bands named after"]
Entities: ["Schumann–Runge bands"]

**Example 2:**
**Question:** Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
**Output:**
Analysis: To compare the nationalities of the two directors, I first need to identify who each director is. These are two independent pieces of information that can be retrieved in parallel as the first step.
Query: ["director of Move (1970 Film)", "director of Méditerranée (1963 Film)"]
Entities: ["Move (1970 Film)", "Méditerranée (1963 Film)"]

**Example 3:**
**Question:** How many Primetime Emmy Awards did the actor, who starred with Dan Aykroyd and Donna Dixon in Spies Like Us, earn?
**Output:**
Analysis: To find out how many Primetime Emmy Awards the actor earned, I need to first identify which actor starred with Dan Aykroyd and Donna Dixon in "Spies Like Us."
Query: ["actor who starred with Dan Aykroyd and Donna Dixon in Spies Like Us"]
Entities: ["Spies Like Us"]

**Example 4:**
**Question:** Where did the leader of the largest European country after the collapse of the country that denied anything more than an advisory role in the Korean war die?
**Output:**
Analysis: The question is complex and requires multiple pieces of information. First, I need to identify which country denied anything more than an advisory role in the Korean War. Then, I need to determine the largest European country after the collapse of that country. Once I have the leader of that country, I can determine where they died. I will begin by identifying the country that denied a combat role in the Korean War.  
Query: ["country that denied anything more than an advisory role in Korean War"]
Entities: ["Korean War"]
---

### YOUR TASK
**Question:** ${question}

**Output:**
''')

'A query is NOT the first step if it depends on information that is not yet known. These dependent queries often contain placeholders (e.g., `[figure skater]`, `[country]`) that need to be resolved by a prior query.'

'''
**Example 2: Refinement Needed (Complex Dependency Chain)**
**Input Plan:**
- **Question:** "Where did the leader of the largest European country after the collapse of the country that denied anything more than an advisory role in the Korean war die?"
- **Analysis:** "This is a multi-step query. First, identify the country in the Korean war. Then find the largest European country after its collapse. Then find that country's leader. Finally, find where the leader died."
- **Query:** ["country that denied role in Korean war", "largest European country after collapse of [country]", "leader of [largest country]", "death location of [leader]"]
- **Entities:** ["Korean War", "[country]", "[largest country]", "[leader]"]

**Refined Output:**
Analysis: The original plan listed a full sequential path. However, all steps after the first are dependent on prior results. The only query that can be executed immediately is the one to resolve the most deeply nested clause. The plan is refined to contain only this single, foundational query.
Query: ["country that denied anything more than an advisory role in the Korean war"]
Entities: ["Korean War"]
'''



filter_sequential_prompt = Template('''\
### ROLE
You are a meticulous Retrieval Plan Refiner. Your function is to analyze a proposed search plan and distill it down to *only* the immediate next actionable first step(s).

### TASK
You will be given an original question and a search plan (Analysis, Query, Entities) that might contain multiple sequential steps. Your goal is to identify and isolate the very first query or queries that can be executed *without* any prior knowledge. You must discard any subsequent steps that depend on the results of the first step (these are often recognizable by placeholders like `[entity_name]`).

### CONSTRAINTS
- The number of queries in the `Query` list must not exceed 5.
- **[CRITICAL]** The `Query` and `Entities` lists must have the exact same number of items. For each query, the corresponding item in the `Entities` list must be the **single, most specific, and central named entity** being investigated in that query.

### OUTPUT FORMAT
Your output must be the refined plan, strictly following the original format:
Refinement Rationale: Your analysis explaining the refinement or confirmation of the original plan.
Analysis: The final, refined analysis for the refined plan.
Query: [A Python-style list containing only the first, actionable query/queries.]
Entities: [A Python-style list of the core entities in each Query, corresponding one-to-one with the Query list. The Nth entity must be the central subject of the Nth query.]

Your output must be a "Refined Plan" that contains only the immediate, non-dependent next step(s), following the original format exactly. 

---
### EXAMPLES

**Example 1: Refinement Needed**
**Input Plan:**
- **Question:** "What is the nationality of one of the men for whom the Schumann–Runge bands are named?"
- **Analysis:** "To find the nationality, I first need to find who the bands are named after, and then look up their nationality."
- **Query:** ["who are the Schumann-Runge bands named after", "nationality of [person's name]"]
- **Entities:** ["Schumann–Runge bands", "[person's name]"]
 
**Refined Output:**
Refinement Rationale: The original plan incorrectly included a future step. The query 'nationality of [person's name]' is dependent on the result of the first query and cannot be executed yet. The plan should be  refined to contain only the immediate next, independent query.
Analysis: To eventually answer the question about nationality, the foundational first step is to identify the individual(s) the Schumann–Runge bands are named after.
Query: ["who are the Schumann-Runge bands named after"]
Entities: ["Schumann–Runge bands"]

**Example 2:**
**Input Plan:**
- **Question:** Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
- **Analysis:** To compare the nationalities of the two directors, I first need to identify who each director is. These are two independent pieces of information that can be retrieved in parallel as the first step.
- **Query:** ["director of Move (1970 Film)", "director of Méditerranée (1963 Film)"]
- **Entities:** ["Move (1970 Film)", "Méditerranée (1963 Film)"]

**Refined Output:**
Refinement Rationale: The provided plan is correct. It identifies two parallel, independent queries that represent the immediate next steps. No refinement is needed as there are no dependencies between the queries.
Analysis: To compare the nationalities of the two directors, I first need to identify who each director is. These are two independent pieces of information that can be retrieved in parallel as the first step.
Query: ["director of Move (1970 Film)", "director of Méditerranée (1963 Film)"]
Entities: ["Move (1970 Film)", "Méditerranée (1963 Film)"]
---  

### YOUR TASK
**Input Plan:**
- **Question:** ${question}
- **Analysis:** ${analysis}
- **Query:** ${query_list}
- **Entities:** ${entity_list}

**Refined Output:**
''')