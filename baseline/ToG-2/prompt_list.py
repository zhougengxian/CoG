"""Prompts retained for the CoG ToG-2 baseline.

Adapted from IDEA-FinAI/ToG-2.
"""

generate_directly = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: Washington, D.C.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: Bharoto Bhagyo Bidhata

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: Jason Allen Alexander

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: Peter Paul Rubens

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: Georgia

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: Heroin."""

find_entity_candidates_prompt_wiki = """Please identify the two Entites Most relevant to the question in the following Entites, to fill the Triplet (Entity-Relationship-Entity). Please refer to Related References for entities. You must choose two entities in the Entites list, and enclose each in curly brackets like {{xxx}},{{xxx}} in the answer.
Question: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?
Triplet:({{Staten Island Summer}}--{{cast member}}--?)
Entites:
{{Ashley Greene}},{{Bobby Moynihan}},{{Camille Saviola}},{{Cecily Strong}},{{Mary Birdsong}}
Related References: 
{{Entity: Ashley Greene}}:{{Reference: is an American actress and model, best known for her role as Alice Cullen in the "Twilight" film series.}}; 
{{Entity: Bobby Moynihan}}:{{Reference: is an American actor and comedian who was a cast member on "Saturday Night Live" from 2008 to 2017. While I couldn't find a direct connection between Bobby Moynihan and the film "Staten Island Summer" }}; 
{{Entity: Camille Saviola}}:{{Reference: is an American actress known for her work in both film and theater. However, there doesn't appear to be a direct connection between Camille Saviola and the question about "Staten Island Summer" and "Saturday Night Live."}}; 
{{Entity: Cecily Strong}}:{{Reference: is an American actress and comedian who is best known as a cast member on "Saturday Night Live." She has been a part of the show since 2012 and has gained recognition for her comedic performances. }}; 
{{Entity: Kate Walsh}}:{{Reference: is an American actress best known for her role as Dr. Addison Montgomery in the television series "Grey's Anatomy" and its spin-off "Private Practice." While I couldn't find a direct connection between Kate Walsh and the film "Staten Island Summer" or "Saturday Night Live,"}};

To fill the Triplet ({{Staten Island Summer}}-{{cast member}}-?) for the question, we need to determine which entities are most relevant to the question and have a higher likelihood of being the correct answer.You must provide two entities in the answer. please note that the analyzed answer entity should be enclosed in curly brackets {{xxxxxx}}.
In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."
The two most relevant entities that meet these criteria are:
{{Cecily Strong}},{{Kate Walsh}}

"""

find_entity_candidates_prompt_wiki2 = """
Question: {}
Triplet:{}
Entites:{}
Related References: 
"""

score_entity_candidates_prompt_wiki = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?
Relation: cast member
Entites and Related References: 
{{Entity: Ashley Greene}}:{{Reference: is an American actress and model, best known for her role as Alice Cullen in the "Twilight" film series.}}; 
{{Entity: Bobby Moynihan}}:{{Reference: is an American actor and comedian who was a cast member on "Saturday Night Live" from 2008 to 2017. While I couldn't find a direct connection between Bobby Moynihan and the film "Staten Island Summer" }}; 
{{Entity: Camille Saviola}}:{{Reference: is an American actress known for her work in both film and theater. However, there doesn't appear to be a direct connection between Camille Saviola and the question about "Staten Island Summer" and "Saturday Night Live."}}; 
{{Entity: Cecily Strong}}:{{Reference: is an American actress and comedian who is best known as a cast member on "Saturday Night Live." She has been a part of the show since 2012 and has gained recognition for her comedic performances. }}; 
{{Entity: Mary Birdsong}}:{{Reference: is an American actress and comedian. She has appeared in various television shows and films throughout her career. While I couldn't find a direct connection between Mary Birdsong and the film "Staten Island Summer" or "Saturday Night Live"}}
Score: 0.0, 0.4, 0.2, 0.4, 0.0
To score the entities\' contribution to the question, we need to determine which entities are relevant to the question and have a higher likelihood of being the correct answer.
In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."
The relevant entities that meet these criteria are:\n- Ashley Greene\n- Bobby Moynihan\n- Camille Saviola\n- Cecily Strong\n- Mary Birdsong\n\nTo distribute the scores, we can assign a higher score to entities that are more likely to be the correct answer. In this case, the most likely answer would be an actress who was a cast member of "Saturday Night Live" around the time the movie was released.
Based on this reasoning, the scores could be assigned as follows:\n- Ashley Greene: 0\n- Bobby Moynihan: 0.4\n- Camille Saviola: 0.2\n- Cecily Strong: 0.4\n- Mary Birdsong: 0.0

Q: {}
Relation: {}
Entites and Related References: 
"""

vanilla_prompt_reasoning_qa_2shot= """
# Task: Given a question, the associated retrieved knowledge, you are asked to answer whether it's sufficient for you to answer the question ({{Yes}} or {{No}}) at the beginning, with these Information and your knowledge. If {{Yes}}, please note that the analyzed answer entity should be enclosed in curly brackets {{xxxxxx}}
# Note that your answer must begin with {{Yes}} or {{No}}. Here are some examples:
# Example1:
Question: 
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army, which belonged to the Empire of Japan. The Empire of Japan was a sovereign state that existed from 1868 to 1947, ruled by the Emperor of Japan.
The Imperial Japanese Army, in which Viscount Yamaji Motoharu served as a general, was the land-based military force of the Empire of Japan during its early years.
During the time when Viscount Yamaji Motoharu held the rank of general, the Empire of Japan was expanding its influence throughout East Asia and engaging in various military campaigns.
As a general in the early Imperial Japanese Army, Viscount Yamaji Motoharu played a role in shaping the military strategies and operations of the Empire of Japan during that period.
The Empire of Japan, under which Viscount Yamaji Motoharu served as a general, had aspirations of regional dominance and pursued an aggressive expansionist policy, leading to conflicts with neighboring countries such as China and Russia.
Answer: 
{Yes}. Based on the given knowledge and my own knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

# Example2:
Question: 
Who is the coach of the team owned by Steve Bisciotti?
Retrieved References:
The team owned by Steve Bisciotti is the Baltimore Ravens.
Answer: 
{No}. Based on the given knowledge and my own knowledge, the team owned by Steve Bisciotti is the Baltimore Ravens. Additional knowledge about the specific coach of the Baltimore Ravens is required to answer the question. 

#Now, please carefully consider the following case:
"""

prompt_reasoning_qa_2shot= """
# Task: Given a question, the associated retrieved knowledge triplets and retrieved sentences, you are asked to answer whether it's sufficient for you to answer the question ({{Yes}} or {{No}}) at the beginning, with these triplets and Related Information and your knowledge. If {{Yes}}, please note that the analyzed answer entity should be enclosed in curly brackets {{xxxxxx}}
# Note that your answer must begin with {{Yes}} or {{No}}. Here are some examples:
# Example1:
Question: 
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge triplets: 
Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
Retrieved References:
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army, which belonged to the Empire of Japan. The Empire of Japan was a sovereign state that existed from 1868 to 1947, ruled by the Emperor of Japan.
The Imperial Japanese Army, in which Viscount Yamaji Motoharu served as a general, was the land-based military force of the Empire of Japan during its early years.
During the time when Viscount Yamaji Motoharu held the rank of general, the Empire of Japan was expanding its influence throughout East Asia and engaging in various military campaigns.
As a general in the early Imperial Japanese Army, Viscount Yamaji Motoharu played a role in shaping the military strategies and operations of the Empire of Japan during that period.
The Empire of Japan, under which Viscount Yamaji Motoharu served as a general, had aspirations of regional dominance and pursued an aggressive expansionist policy, leading to conflicts with neighboring countries such as China and Russia.
Answer: 
{Yes}. Based on the given knowledge triplets, retrieved sentences and my own knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

# Example2:
Question: 
Who is the coach of the team owned by Steve Bisciotti?
Knowledge triplets: 
psilocybin, described by source, Opium Law
Opium Law, part of, norcodeine
Gymnopilus spectabilis, parent taxon, Gymnopilus
Retrieved References:
The team owned by Steve Bisciotti is the Baltimore Ravens.
Answer: 
{No}. Based on the given knowledge triplets, retrieved sentences and my own knowledge, the team owned by Steve Bisciotti is the Baltimore Ravens. Additional knowledge about the specific coach of the Baltimore Ravens is required to answer the question. 

#Now, please carefully consider the following case:
"""

prompt_reasoning_qa_query_change_2shot= """
Given a question, the associated retrieved knowledge triplets and retrieved references, you are asked to evaluate if these resources, combined with your pre-existing knowledge, are sufficient to formulate a answer ({{Yes}} or {{No}}).
Your answer must begin with {{Yes}} or {{No}}.
If {{Yes}}, please note that the analyzed answer entity must be enclosed in curly brackets {{xxxxxx}}
If {{No}}, which means the resources are useless or provide clues are helpful but insufficient to conclusively answer the question, identify the missing aspects and refine the search query to specifically target the information needed to complete the answer. The targeted search query also must be enclosed in curly brackets {{xxxxxx}}

Here are some examples:
### Example 1:
Question: 
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Clues: None
Knowledge triplets: 
Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
Retrieved References:
Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army, which belonged to the Empire of Japan. The Empire of Japan was a sovereign state that existed from 1868 to 1947, ruled by the Emperor of Japan.
The Imperial Japanese Army, in which Viscount Yamaji Motoharu served as a general, was the land-based military force of the Empire of Japan during its early years.
During the time when Viscount Yamaji Motoharu held the rank of general, the Empire of Japan was expanding its influence throughout East Asia and engaging in various military campaigns.
As a general in the early Imperial Japanese Army, Viscount Yamaji Motoharu played a role in shaping the military strategies and operations of the Empire of Japan during that period.
The Empire of Japan, under which Viscount Yamaji Motoharu served as a general, had aspirations of regional dominance and pursued an aggressive expansionist policy, leading to conflicts with neighboring countries such as China and Russia.
### Answer: 
{Yes}. Based on the given knowledge triplets, retrieved sentences and my own knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

### Example 2:
Question: 
Who is the coach of the team owned by Steve Bisciotti?
Clues: None
Knowledge triplets: 
psilocybin, described by source, Opium Law
Opium Law, part of, norcodeine
Gymnopilus spectabilis, parent taxon, Gymnopilus
Retrieved References:
The team owned by Steve Bisciotti is the Baltimore Ravens.
Steve Bisciotti made a 3 years contract with his major coach.
### Answer: 
{No}. Based on the given knowledge triplets, retrieved sentences and my own knowledge, the team owned by Steve Bisciotti is the Baltimore Ravens. Additional knowledge about the specific coach of the Baltimore Ravens is required to answer the question. Therefore, the clue is {the team owned by Steve Bisciotti is the Baltimore Ravens}

Now, please carefully consider the following case:
"""

extract_all_relation_prompt_wiki = """
# Task: 
1. Carefully review the question provided.
2. From the list of available relations for their corresponding entity, select the %s that you believe are most likely to link to the entities that can provide the most relevant information to help answer the provided question.
3. For each selected relation, provide a score between 0 to 10 reflecting its usefulness in answering the question, with 10 being most useful.
4. Provide a brief explanation for your choices, highlighting how each selected relation potentially contributes to answering the question.

# The input follows below format:
Question:[The question text]
Entity 1:[The name of the entity 1]
Available Relations:[A relation list of entity 1 to be chosen.]
Entity 2:[The name of the entity 2]:
Available Relations:[A relation list of entity 2 to be chosen.]
...(Continue in the same manner for additional entities)

# Below is two examples:
# Example1:
Question: Mesih Pasha's uncle became emperor in what year?
Topic Entity: Mesih Pasha
Relations:
1. wiki.relation.child
2. wiki.relation.country_of_citizenship
3. wiki.relation.date_of_birth
4. wiki.relation.family
5. wiki.relation.father
6. wiki.relation.languages_spoken, written_or_signed
7. wiki.relation.military_rank
Answer: 
1. {wiki.relation.family (Score: 1.0)}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.
2. {wiki.relation.father (Score: 0.4)}: Uncle is father's brother, so father might provide some information as well.
3. ......

# Example2:
Question: what the attitude of Joe Biden towards China?
Entity 1: china
Relations:
1. wiki.relation.alliance
2. wiki.relation.international_relation
3. wiki.relation.political_system
4. wiki.relation.population
Entity 2: joe biden
Relations:
1. wiki.relation.political_position
2. wiki.relation.presidency
3. wiki.relation.family
4. wiki.relation.early_life

Answer:
Entity 1: 
1. {wiki.relation.alliance (Score: 8)}: This relation is highly relevant as it can provide information about the relationship between china and other parties.
2. {wiki.relation.political_system (Score: 7)}:  This relation is relevant as it can provide information about the policies that might reflect the relationship between china and other parties.
3. ......
Entity 2: 
1. {wiki.relation.political_position (Score: 10)}: This relation is highly relevant as it can provide information about joe biden's political position to other parties or countries.
2. {wiki.relation.presidency (Score: 2)}: This relation is slightly relevant as it can provide information about joe biden's position, which might provide clues to answer the question.
3. ......

# For questions involving multiple entities, you are required to analyze the relation list for each entity separately. Then select the %s most relevant relations for each entity, based on the analysis as done in the given example. 
# It's essential to maintain strict adherence to the line breaks and format as seen in the provided example for clarity and consistency:

Answer: 
Entity 1: The Name of Entity 1
1. {Relation1 (Score: X)}: Explanation.
2. {Relation2 (Score: Y)}: Explanation.

Entity 2: The Name of Entity 2
1. {Relation1 (Score: X)}: Explanation
2. {Relation2 (Score: Y)}: Explanation

...(Continue in the same manner for additional entities)

# Now, I will provide you a new question with %s entities and relations. Please analyse it following all the guidance above carefully. 

Question: """

extract_relation_prompt_wiki = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1.
Question: Mesih Pasha's uncle became emperor in what year?
Topic Entity: Mesih Pasha
Relations:
1. child
2. country_of_citizenship
3. date_of_birth
4. family
5. father
6. languages_spoken
7. military_rank
8. occupation
9. place_of_death
10. position_held

Answer: 
1. {family (Score: 1.0)}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.
2. {father (Score: 0.4)}: Uncle is father's brother, so father might provide some information as well.
3. {position held (Score: 0.1)}: This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to becoming an emperor.
4. ......

Question: Van Andel Institute was founded in part by what American businessman, who was best known as co-founder of the Amway Corporation?
Topic Entity: Van Andel Institute
Relations:
1. wiki.relation.affiliation
2. wiki.relation.country
3. wiki.relation.donations
4. wiki.relation.educated_at
5. wiki.relation.employer

Answer: 
1. {wiki.relation.affiliation (Score: 0.8)}: This relation is relevant because it can provide information about the individuals or organizations associated with the Van Andel Institute, including the American businessman who co-founded the Amway Corporation.
2. {wiki.relation.donations (Score: 0.3)}: This relation is relevant because it can provide information about the financial contributions made to the Van Andel Institute, which may include donations from the American businessman in question.
3. {wiki.relation.educated_at (Score: 0.3)}: This relation is relevant because it can provide information about the educational background of the American businessman, which may have influenced his involvement in founding the Van Andel Institute.
4. ......

Question: """

cot_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
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
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""

topic_prune_demos = '''
Given a question and a group of related topic entities derived from the Wikipedia knowledge graph, the task is to select which of these entities are suitable as starting points for reasoning on the wiki knowledge graph to find information and clues that are useful for answering the question. 
Note that your output should be strictly Json formatted: {id: entity}. 
Here are examples showing how to analyse and output a Json formatted answer:

Example 1:
question: What major city is the Faith Lutheran Middle School and High School located by?
topic entities: {
    "Q111": "Faith Lutheran Middle School",
    "Q39722": "Faith Lutheran High School"
}
Analysis: 
All entities are directly related to the core of the question—finding the possibale information about Faith Lutheran and its location.
Output: 
{"Q111": "Faith Lutheran Middle School"，"Q39722": "Faith Lutheran High School"}

Example 2:
question: How many Turkish verbs ending with “uş” with their lemma.
topic entities: {
    "Q24905": "verb"
}
Analysis: 
Q24905 "verb" focuses on verbs, which are action words in a language. The task involves not just identifying verbs but specifically Turkish verbs that end with the suffix "uş,". Therefore, the entity "verb" is too broad and does not point to a specific Turkish verbs. Thus there is no suitable topic entity as a starting point for reasoning. 
Output: 
{}

Example 3:
question: On which island is the Indonesian capital located?
topic entities: {
    "Q252": "Indonesia",
    "Q23442": "island"
}
Analysis: 
Q252 "Indonesia" represents the country of Indonesia. Considering the question is about the capital of Indonesia and which island it is located on, the entity "Indonesia" is directly related to the core of the question—finding the capital of Indonesia and then identifying the island it is situated on. Therefore, this entity is highly suitable as a starting point for reasoning. 
Q23442 "island" represents the concept of an island. While the question does indeed relate to an island, the concept of "island" itself is too broad and does not point to a specific geographical location or country. From the perspective of reasoning on a knowledge graph, trying to find the capital of a specific country based solely on the concept of an island is less relevant and efficient compared to starting from that country. 
Hence, Q252 "Indonesia" is the more suitable topic entity as a starting point for reasoning on the knowledge graph. It directly connects to the key information of the question and can effectively guide the search for entities related to the answer within the knowledge graph.
Output: 
{"Q252": "Indonesia"}
'''

hotpotqa_s1_prompt_demonstration = """
Please strictly follow the format of the examples below and think step-by-step to answer the question. You need to first provide a step-by-step reasoning that clearly explains the logic for reaching the answer, and then conclude with 'The answer is' to give the final answer.
Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in what year
A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second, Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991.

Q: What band did Antony King work with that formed in 1985 in Manchester?
A: First, Antony King worked as house engineer for Simply Red. Second, Simply Red formed in 1985 in Manchester. The answer is Simply Red.

Q: How many inhabitants were in the city close to where Alberta Ferretti’s studios was located?
A: First, Alberta Ferretti’s studio is near Rimini. Second, Rimini is a city of 146,606 inhabitants. The answer is 146,606.

"""
