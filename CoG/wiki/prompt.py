from string import Template


prompt_refine_search = Template('''\
### ROLE
You are a research agent trying to find the right information on Wikipedia after a search query failed.

### TASK
Your search for "${entity}" did not find a matching Wikipedia page. Use the contextual information and the suggestions provided to decide on the best query for your next attempt.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}
**Failed Search Query:** "${entity}"

### SUGGESTIONS
Below are suggestions from the Wikipedia search API to help you refine your query.

- **"Did you mean?":** This is the API's top recommendation, often correcting a typo or suggesting a more standard page title.
  - Suggestion: ${suggestion}

- **Similar Pages Found:** These are pages with titles that are textually similar to your failed query. One of them might be the correct page under a slightly different name.
  - Similar Pages: ${search_results_list}

### YOUR DECISION
Based on the context and the suggestions, choose the best query for your next search attempt. This could be the "Did you mean?" suggestion, one of the similar pages, or a completely new query you formulate based on the feedback from this failed search. Your output must be a single line containing only the new search query. If none of the suggestions seem useful for answering the original question, output the string "NO_MATCH".

### YOUR RESPONSE:
''')

prompt_disambiguate = Template('''\
### ROLE
You are a research agent resolving an ambiguous search query on Wikipedia.

### TASK
Your search for "${entity}" led to a disambiguation page with multiple possible meanings. You must carefully analyze the full context provided below to choose the single most relevant link from the options.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}
**Ambiguous Search Query:** "${entity}"

### OPTIONS
${options_list}

### YOUR DECISION
Analyze the options based on the context to select the best fit. Your output must be a single line containing only the exact title of the chosen page from the list. If none of the options seem relevant for answering the original question, output the string "NO_MATCH".

### YOUR RESPONSE:
''')

prompt_refine_search_with_reasoning = Template('''\
### ROLE
You are a research agent trying to find the right information on Wikipedia after a search query failed.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}
**Failed Search Query:** "${entity}"

### SUGGESTIONS
Below are suggestions from the Wikipedia search API to help you refine your query.

- **"Did you mean?":** This is the API's top recommendation, often correcting a typo or suggesting a more standard page title.
  - Suggestion: ${suggestion}

- **Similar Pages Found:** These are pages with titles that are textually similar to your failed query. One of them might be the correct page under a slightly different name.
  - Similar Pages: ${search_results_list}

### YOUR TASK
Your search for "${entity}" did not find a matching Wikipedia page. Use the contextual information and the suggestions provided to decide on the best query for your next attempt.

1.  **Reasoning:** First, write down your thought process. Analyze the original failed query, the context of the main question, and the provided suggestions ("Did you mean?" and similar pages). Explain which option is the most promising and why, or if none are suitable.
2.  **Decision:** Based on your reasoning, provide the single best query for the next search attempt. This could be the "Did you mean?" suggestion, one of the similar pages, or a completely new query you formulate based on the feedback from this failed search. If none of the suggestions seem useful for answering the original question, the new query should be the string "NO_MATCH".

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Rationale: Your step-by-step reasoning on why you are choosing or formulating the next search query.
New Query: The single best query for the next search attempt or NO_MATCH.

### YOUR RESPONSE:
''')

prompt_disambiguate_with_reasoning = Template('''\
### ROLE
You are a research agent resolving an ambiguous search query on Wikipedia.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}
**Ambiguous Search Query:** "${entity}"

### OPTIONS
${options_list}

### YOUR TASK
Your search for "${entity}" led to a disambiguation page with multiple possible meanings. You must carefully analyze the full context provided above to choose the single most relevant page from the options.

1.  **Reasoning:** First, write down your thought process. Based on the original question, overall plan, and current sub-task, explain why you are selecting one specific page as the most relevant, or why none of them are suitable.
2.  **Decision:** Based on your reasoning, provide the single most relevant page title from the options list. If none of the options seem relevant for answering the original question, the chosen page should be the string "NO_MATCH".

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Rationale: Your step-by-step reasoning for choosing an option or concluding "NO_MATCH".
Chosen Page: The exact title of the chosen page from the option list or NO_MATCH.

### YOUR RESPONSE:
''')

prompt_refine_search_history = Template('''\
### ROLE
You are a research agent trying to find the right information on Wikipedia after a search query failed.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}

### SEARCH HISTORY
This is the history of your previous search attempts, which have failed. Use it to inform your next decision and avoid repeating mistakes.
${search_history}

### CURRENT FAILED SEARCH
**Failed Search Query:** "${entity}"

### SUGGESTIONS
Below are suggestions from the Wikipedia search API to help you refine your query.

- **"Did you mean?":** This is the API's top recommendation, often correcting a typo or suggesting a more standard page title.
  - Suggestion: ${suggestion}

- **Similar Pages Found:** These are pages with titles that are textually similar to your failed query. One of them might be the correct page under a slightly different name.
  - Similar Pages: ${search_results_list}

### YOUR TASK
Your search for "${entity}" did not find a matching Wikipedia page. Use the contextual information, your search history, and the suggestions provided to decide on the best query for your next attempt.

1.  **Reasoning:** First, write down your thought process. Analyze the original failed query, the search history, the context of the main question, and the provided suggestions ("Did you mean?" and similar pages). Explain which option is the most promising and why, or if none are suitable, ensuring you do not repeat a past failed attempt.
2.  **Decision:** Based on your reasoning, provide the single best query for the next search attempt. This could be the "Did you mean?" suggestion, one of the similar pages, or a completely new query you formulate based on your analysis of all the information provided. If none of the suggestions seem useful for answering the original question, the new query should be the string "NO_MATCH".

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Rationale: Your step-by-step reasoning on why you are choosing or formulating the next search query.
New Query: The single best query for the next search attempt or NO_MATCH.

### YOUR RESPONSE:
''')

prompt_disambiguate_history = Template('''\
### ROLE
You are a research agent resolving an ambiguous search query on Wikipedia.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Sub-Task (Query):** ${query}

### SEARCH HISTORY
You have already tried the following search queries and they did not lead to the correct page. Do not choose them again if they appear in the options below.
${search_history}

### CURRENT AMBIGUOUS SEARCH
**Ambiguous Search Query:** "${entity}"

### OPTIONS
Below are the options for the ambiguous query.
${options_list}

### YOUR TASK
Your search for "${entity}" led to a disambiguation page with multiple possible meanings. You must carefully analyze the full context provided above and your search history to choose the single most relevant page from the options.

1.  **Reasoning:** First, write down your thought process. Based on the original question, overall plan, current sub-task, and your search history, explain why you are selecting one specific page as the most relevant, or why none of them are suitable. Ensure your choice is not one of the previously failed attempts.
2.  **Decision:** Based on your reasoning, provide the single most relevant page title from the options list. If none of the options seem relevant for answering the original question, the chosen page should be the string "NO_MATCH".

### OUTPUT FORMAT
Your output must follow this exact structure, with no additional commentary. Each key must be on a new line.

Rationale: Your step-by-step reasoning for choosing an option or concluding "NO_MATCH".
Chosen Page: The exact title of the chosen page from the option list or NO_MATCH.

### YOUR RESPONSE:
''')


'''If none of the options seem relevant, output the string "NEW_SEARCH".
You must carefully analyze the full context provided below to choose the single most relevant link from the options.
You must choose the most relevant link from the options below to answer the original question.
'''

prompt_present_info3 = Template('''\
### ROLE
You are a research agent tasked with answering a complex question by navigating Wikipedia.

### TASK
You have been provided with the summary and section list of a Wikipedia page relevant to your current query. Your goal is to either extract the key information needed for the current step directly from the summary, decide which section(s) of the article to read next for more details, or determine that this page is irrelevant.

### YOUR DECISION
Based on the information above, what your the next best action? Choose one of the following three options.

1.  **If the summary contains enough information to satisfy the current search query:**
    - Output the decision and the concise piece of information. This is not necessarily the final answer to the original question, but the information needed for this specific step.
    - Format:
      Decision: EXTRACT_INFO
      Extracted Info: [The key information found in the summary]
                               
2.  **If the summary is insufficient but some sections look promising:**
    - Output the decision and a list of the exact titles of the most relevant sections to explore.
    - Format:
      Decision: EXPLORE_SECTIONS
      Sections: [A Python-style list of exact section titles, e.g., ["History", "Development"]]

3.  **If this page ### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Your Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"

### AVAILABLE INFORMATION
**Page Summary:**
${page_summary}

**Page Sections (Top-Level):**
${page_sections}is not relevant to the original question:**
    - Output the decision, a brief rationale, and a better query to try next.
    - Format:
      Decision: NEW_SEARCH
      Rationale: Briefly explain why the page is irrelevant
      New Query: "Your new search query"

### YOUR RESPONSE:
''')

prompt_present_info = Template('''\
### ROLE
You are a research agent tasked with answering a complex question by navigating Wikipedia.

### TASK
You have been provided with the summary and section list of a Wikipedia page relevant to your current query. Your goal is to extract all useful information from the summary, identify promising sections for deeper investigation, or determine that the page is irrelevant.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Your Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"

### AVAILABLE INFORMATION
**Page Summary:**
${page_summary}

**Page Sections (Top-Level):**
${page_sections}

### YOUR DECISION
Based on the information above, what is your next best action? Choose one of the following two options.

1.  **If the page seems relevant to the query:**
    - Provide a brief rationale explaining your choices for extraction and exploration.
    - Extract two types of information from the **Page Summary**:
      - **Direct Information:** Any fact that directly answers or helps answer the **Current Search Query**.
      - **Promising Clues:** Identify and extract key facts or entities that are strongly related to the query, or are valuable for solving the **Original Question**. These are the critical starting points for the next phase of the investigation.
      - **[CRITICAL RULE]** The `Extracted Info` field is for **EXTRACTION ONLY**. Do NOT add any reasoning, hypotheses, or information not explicitly present in the summary. All reasoning belongs in the `Rationale`.
    - Identify any sections that seem promising for finding more details.
    - **Format:**
      Decision: EXTRACT_AND_EXPLORE
      Rationale: Briefly explain why this page is relevant, what key information the summary provides (or lacks) regarding the query, and why you are choosing specific sections to explore further.
      Extracted Info: The key information found in the summary. Do not include any explanations or reasoning here. If the summary has no relevant information, state None.
      Relevant Sections: [A Python-style list of exact section titles to explore further, e.g., ["History", "Development"]. If no further exploration is needed or no sections seem relevant, provide an empty list [].]

2.  **If this page is clearly not relevant to the query:**
    - Explain why the page is irrelevant and suggest a better search query.
    - **Format:**
      Decision: IRRELEVANT
      Rationale: Briefly explain why the page is irrelevant.
      New Query: "Your new search query"

### YOUR RESPONSE:
''')

prompt_present_info_with_table = Template('''\
### ROLE
You are a research agent tasked with answering a complex question by navigating Wikipedia.

### TASK
You have been provided with the summary, an infobox table, and the section list of a Wikipedia page relevant to your current query. Your goal is to extract all useful information from the summary and infobox, identify promising sections for deeper investigation, or determine that the page is irrelevant.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Your Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"

### AVAILABLE INFORMATION
**Page Summary:**
${page_summary}

**Summary Infobox:**
${infobox_table}

**Page Sections (Top-Level):**
${page_sections}

### YOUR DECISION
Based on the information above, what is your next best action? Choose one of the following two options.

1.  **If the page seems relevant to the query:**
    - Provide a brief rationale explaining your choices for extraction and exploration, considering both the text summary and the infobox.
    - Extract two types of information from the **Page Summary** and **Summary Infobox**:
      - **Direct Information:** Any fact that directly answers or helps answer the **Current Search Query**.
      - **Promising Clues:** Identify and extract key facts or entities that are strongly related to the query, or are valuable for solving the **Original Question**. These are the critical starting points for the next phase of the investigation.
      - **[CRITICAL RULE]** The `Extracted Info` field is for **EXTRACTION ONLY**. Do NOT add any reasoning, hypotheses, or information not explicitly present in the summary or infobox. All reasoning or explanation belongs in the `Rationale`.
    - Identify any sections that seem promising for finding more details.
    - **Format:**
      Decision: EXTRACT_AND_EXPLORE
      Rationale: Briefly explain why this page is relevant, what key information the summary and infobox provide (or lack) regarding the query, and why you are choosing specific sections to explore further.
      Extracted Info: The key information found in the summary and infobox. Do not include any explanations or reasoning here. If no relevant information is found, state None.
      Relevant Sections: [A Python-style list of exact section titles to explore further, e.g., ["History", "Development"]. If no further exploration is needed or no sections seem relevant, provide an empty list [].]

2.  **If this page is clearly not relevant to the query:**
    - Explain why the page is irrelevant and suggest a better search query.
    - **Format:**
      Decision: IRRELEVANT
      Rationale: Briefly explain why the page is irrelevant.
      New Query: "Your new search query"

### YOUR RESPONSE:
''')

'''
      Rationale: Your reasoning for why you extracted the info (or not) and why you chose the sections for further exploration.
**VERBATIM EXTRACTION ONLY**
      Rationale: Briefly explain why this page is relevant, what key information the summary provides (or lacks) regarding the query, and why you are choosing specific sections to explore further.'''
'''Extracted Info: [Directly relevant facts from the summary. Must be "None" if no direct information is found.]'''
'''Relevant Sections: [A Python-style list of exact section titles to explore next, e.g., ["History", "Personal life"]. Provide an empty list [] if no further exploration is needed or if no sections look promising.]'''


prompt_extract_from_chunks = Template('''\
### ROLE
You are a meticulous research agent. Your current task is to extract specific information from text chunks retrieved from a Wikipedia page section.

### TASK
You have been given several retrieved text chunks from a specific section of a Wikipedia page. Your goal is to carefully read these chunks and extract any information that helps answer your **current search query**.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"
**Section Being Investigated:** "${section_title}"
**Section Exploration Rationale (during summary reading):** ${exploration_rationale}

### RETRIEVED CHUNKS
The following are the top ${k} most relevant text chunks from the section "${section_title}":
${chunks}

### OUTPUT FORMAT
Based on the information above, provide your response in the following format.
Rationale: Briefly explain whether the chunks contain relevant information and why. If they are irrelevant or unhelpful, state that clearly.
Extracted Info: The key information found in the section. Do not include any explanations or reasoning here. If the section has no relevant information, state None.

### YOUR RESPONSE:
''')

prompt_extract_from_chunks_full = Template('''\
### ROLE
You are a meticulous research agent. Your current task is to extract specific information from text chunks retrieved from a Wikipedia page section.

### TASK
You have been given several retrieved text chunks from a specific section of a Wikipedia page. Your goal is to carefully read these chunks and extract any information that helps answer your **current search query**.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"
**Section Being Investigated:** "${section_title}"
**Section Exploration Rationale (during summary reading):** ${exploration_rationale}
**Information Already Extracted (from Summary):** ${summary_info}

### RETRIEVED CHUNKS
The following are the top ${k} most relevant text chunks from the section "${section_title}":
${chunks}

### OUTPUT FORMAT
Based on the information above, provide your response in the following format.
Rationale: Briefly explain whether the chunks contain relevant information and why. If they are irrelevant or only repeat existing information, state that clearly.
Extracted Info: Synthesize the new and relevant information into a concise, factual summary. Adhere strictly to the information present in the chunks. Do not include any explanations or reasoning here. If no new, relevant information is found in the chunks, state None.

### YOUR RESPONSE:
''')
'''A bulleted list of new, relevant facts extracted from the chunks. Each bullet point should be a concise and self-contained piece of information. If no new information is found, state 'None'.
Rationale: [Your brief explanation here. State whether new information was found and why it's relevant, or explain why no useful information was present.]
Extracted Info: [A synthesized paragraph or bulleted list of NEW facts. If no new information was found, you MUST write the single word "None".]
'''

prompt_analyze_section_with_tables = Template('''\
### ROLE
You are a meticulous research agent. Your task is to analyze a section of a Wikipedia page that contains both text and tables to extract relevant information for a given query.

### TASK
You have been given several retrieved text chunks and a preview of all tables found within a specific Wikipedia section. Your goal is to:
1.  Extract two types of information from BOTH the **Retrieved Text Chunks** and the **Table Previews**:
    - **Direct Information:** Any fact that directly answers or helps answer the **Current Search Query**.
    - **Promising Clues:** Identify and extract key facts that are strongly related to the query, or are valuable for solving the **Original Question**.
2.  Analyze the **table previews**. If a table seems to contain relevant information but the preview is insufficient to extract it completely, add its name to the `Selected Tables` list for a full read later.
3.  Provide a single, unified rationale that explains both your information extraction findings and your table selections.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"
**Section Being Investigated:** "${section_title}"
**Section Exploration Rationale (during summary reading):** ${exploration_rationale}

### AVAILABLE INFORMATION
**1. Retrieved Text Chunks:**
The following are the top ${k} most relevant text chunks from the section "${section_title}":
${context_chunks}

**2. Table Previews:**
This section contains the following tables: ${table_names_in_section}

Below are previews of these tables.
${tables_preview}

### OUTPUT FORMAT
Based on all the information provided, generate a response in the following strict format. Do not add any extra explanations outside of the designated fields.

Rationale: A consolidated explanation covering both the text and tables. Explain what information you found (or didn't find) in the text and table previews. For each table you select for a full read, briefly justify why its preview is insufficient but it seems promising.
Extracted Info: The key information (both Direct Information and Promising Clues) found and extracted from **both the text chunks and table previews**. If no relevant information can be extracted from the provided context, state None.
Selected Tables: A Python-style list of the exact names of tables that seem relevant but require a full read because their previews are insufficient. For example: ["Awards and Nominations", "Filmography 2"]. If no tables need a full read, provide an empty list [].

### YOUR RESPONSE:
''')

prompt_extract_from_table_verbose = Template('''\
### ROLE
You are a meticulous research agent. Your task is to perform a detailed analysis of a single, complete Wikipedia table to extract information that supplements findings from the section's text.

### TASK
You have already extracted some information from the text of a Wikipedia section. Now, you are analyzing the full data of a table that you previously identified as promising. Your goal is to extract new information that addresses the **Current Search Query** or provides crucial context for the overall **Original Question**. The information you extract should **supplement, refine, or correct** the information already found in the section's text. Avoid extracting information that is redundant or already covered.

### CONTEXT
#### 1. Overall Objective
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Search Query:** "${query}"

#### 2. Page-level Investigation (Previous Step)
- **Wikipedia Page Explored:** "${page_title}"
- **Information Extracted from page summary:** ${summary_content}
- **Rationale for Analyzing this Page:** ${summary_reading_rationale}

#### 3. Section-level Investigation (Previous Step)
- **Current Section Under Analysis:** "${section_title}"
- **Information Extracted from Section Text:** ${extracted_text_info}
- **Rationale for Analyzing this Section:** ${section_reading_rationale}

### TABLE FOR ANALYSIS
- **Table Name:** "${table_name}"
- **Table Data:**
${full_table_data}

### OUTPUT FORMAT
Based on the information above, provide your response in the following format.

Rationale: Explain what new, relevant information you found in the table and how it supplements, refines, or corrects the text-based findings. If the table is not useful or only contains redundant information, state that clearly.
Extracted Table Info: The new information or facts extracted from the table. Do not include any explanations here. If the table provides no new, relevant information, state None.

### YOUR RESPONSE:
''')


prompt_extract_from_table = Template('''\
### ROLE
You are a meticulous research agent. Your task is to perform a detailed analysis of a single, complete Wikipedia table to extract information that supplements findings from the section's text.

### TASK
You have already extracted some information from the text of a Wikipedia section. Now, you are analyzing the full data of a table that you previously identified as promising. Your goal is to extract new information that addresses the **Current Search Query** or provides crucial context for the overall **Original Question**. The information you extract should **supplement, refine, or correct** the information already found in 'Previously Extracted Information' section. Avoid extracting information that is redundant or already covered.

### CONTEXT
#### 1. Overall Objective
- **Original Question:** ${question}
- **Overall Plan (Analysis):** ${analysis}
- **Current Search Query:** "${query}"

#### 2. Previously Extracted Information
- **Wikipedia Page Explored:** "${page_title}"
- **Information Extracted from page summary:** ${summary_content}
- **Current Section Under Analysis:** "${section_title}"
- **Information Extracted from Section Text:** ${extracted_text_info}

### TABLE FOR ANALYSIS
- **Table Name:** "${table_name}"
- **Table Data:**
${full_table_data}

### OUTPUT FORMAT
Based on the information above, provide your response in the following format.

Rationale: Explain what new, relevant information you found in the table and how it supplements, refines, or corrects the text-based findings. If the table is not useful or only contains redundant information, state that clearly.
Extracted Table Info: The new information or facts extracted from the table. Do not include any explanations here. If the table provides no new, relevant information, state None.

### YOUR RESPONSE:
''')

'''1.  Extract any information from BOTH the **text chunks** and the **table previews** that helps answer the **Current Search Query**.
Provide a step-by-step rationale for your extraction. 
'''

prompt_analyze_section_holistically = Template('''\
### ROLE
You are a meticulous research agent. Your task is to perform a comprehensive analysis of an entire Wikipedia section, including its full text and all associated tables, to extract information relevant to a specific query.

### TASK
You have been provided with the complete text and all tables from a single Wikipedia section that was previously identified as promising. Your goal is to treat this unified content as a single knowledge source and extract all information that helps answer the **Current Search Query** or provides valuable context for the **Original Question**. Your analysis should be holistic, synthesizing information from both the text and the tables to form a complete picture.

### CONTEXT
**Original Question:** ${question}
**Overall Plan (Analysis):** ${analysis}
**Current Search Query:** "${query}"
**Retrieved Wikipedia Page Title:** "${page_title}"
**Section Being Investigated:** "${section_title}"
**Rationale for Exploring this Section (during summary reading):** ${exploration_rationale}

### SECTION CONTENT FOR ANALYSIS
Below is the complete content for the section "${section_title}". You must read and analyze everything provided here.

#### Full Section Text
${section_text}

#### All Tables in Section
${all_tables_string}

### OUTPUT FORMAT
Based on all the information provided, generate a response in the following format.

Rationale: A consolidated explanation of your findings. Explain what relevant information you found (or didn't find) by synthesizing from both the full text and the tables. Describe how the information collectively addresses the query. If the section is ultimately not helpful, explain why.
Extracted Info: The key information extracted from the entire section (text and tables). If no relevant information is found, state None.

### YOUR RESPONSE:
''')
