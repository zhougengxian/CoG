"""Knowledge-graph reasoning functions adapted from IDEA-FinAI/ToG-2."""

import heapq
import json
import math
import random
import time
from utils import *
from search import *

def transform_relation(wiki_relation):
    relation_without_prefix = wiki_relation.replace("wiki.relation.", "").replace("_", " ")
    return relation_without_prefix

def clean_relations(string, entity_id, head_relations,args):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        wiki_relation = match.group("relation").strip()
        wiki_relation = transform_relation(wiki_relation)
        if ';' in wiki_relation:
            continue
        score = match.group("score")
        if not wiki_relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if wiki_relation in head_relations:
            relations.append({"entity_id": entity_id, "relation": wiki_relation, "score": score, "head": True})
        else:
            relations.append({"entity_id": entity_id, "relation": wiki_relation, "score": score, "head": False})

    if not relations:
        return False, "No relations found"
    filtered_relations = [x for x in relations if x['score'] >= 0.2]
    if not filtered_relations:
        return False, "No relations found"
    sorted_data = sorted(filtered_relations, key=lambda x: x['score'], reverse=True)[0:args.width]
    return True, sorted_data

def clean_relation_all_e(results, all_entity_relations):
    entities_info = []
    entity_sections = re.split(r"Entity\s?\d+:", results)[1:]
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    for section in entity_sections:
        section = section.strip()
        try:
            entity_name = re.match(r"(.+?)\n(.+)", section, re.DOTALL).group(1)
            entity_name = entity_name.strip()
        except AttributeError:
            print("=======================No entity name found", section)
            continue
        if entity_name not in all_entity_relations.keys():
            # print("Matching error for entity name", entity_name, all_entity_relations.keys())
            continue
        for match in re.finditer(pattern, section):
            match_relation = match.group("relation").strip().lower()
            match_relation = transform_relation(match_relation)
            if ';' in match_relation:
                continue
            score = match.group("score")
            temp = {}
            temp["relation"] = match_relation
            temp["score"] = float(score)
            relation = [d for d in all_entity_relations[entity_name] if d["relation"] == temp["relation"]]
            if not relation:
                # print("Matching error for relation", match_relation, all_entity_relations[entity_name])
                continue

            temp["entity_id"] = relation[0]["entity_id"]
            temp["entity_name"] = entity_name
            temp['head'] = relation[0]["head"]
            entities_info.append(temp)

    if not entities_info:
        return False, "No relations found"

    seen = set()
    temp_list = []
    for item in entities_info:
        entity_id = item['entity_id']
        relation = item['relation']
        if (entity_id, relation) in seen:
            continue
        seen.add((entity_id, relation))
        temp_list.append(item)
    return True, temp_list

def construct_all_relation_prune_prompt(question, all_entity_relations, args):
    temp_prompt = extract_all_relation_prompt_wiki % (args.width, args.width, args.width) + question
    for i, entity_relations in enumerate(all_entity_relations.values(), start=1):
        if len(entity_relations) > 0:
            temp_prompt += ('\nEntity %s: ' % i + entity_relations[0]['entity_name'] + '\nAvailable Relations:\n' +
                            '\n'.join([f"{i}. {item['relation']}" for i, item in enumerate(entity_relations, start=1)]))
    return temp_prompt + '\nAnswer:'

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_wiki % (args.width)+question+'\nTopic Entity: '+entity_name+ '\nRelations:\n'+'\n'.join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])+'Answer:\n'


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject', 'country of citizenship', "commons category", "commons gallery", "country of origin", "country", "nationality"]
    if check_end_word(relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        return True
    return False


def construct_entity_score_prompt0(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def construct_entity_find_prompt(question,topic_entity_name, relation, entity_candidates):
    
    Triplet='('+'{'+topic_entity_name+'}'+'--'+'{'+relation+'}'+'--'+'?'+')'

    entities=[]
    entity_pack=[]
    for entity in entity_candidates:

        text_list = [d['text'] for d in entity['related_sentences']]

        doc=''.join(text_list)

        entity_pack.append('{Entity: '+entity['name']+'}: {Reference: '+doc[0:300]+'}')

        entities.append('{'+entity['name']+'}')
        
    Entites=",".join(entities)

    check_prompt = find_entity_candidates_prompt_wiki2.format(question, Triplet,Entites) + "; \n".join(entity_pack) + '\nA: '
    final_prompt = find_entity_candidates_prompt_wiki + check_prompt

    return final_prompt,check_prompt

def relation_prune_all(all_entity_relations, question, args):
    prompt = construct_all_relation_prune_prompt(question, all_entity_relations, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.api_key, args.relation_prune_model, base_url=args.base_url)
    flag, retrieve_relations_with_scores = clean_relation_all_e(result, all_entity_relations)
    if flag:
        return retrieve_relations_with_scores
    else:
        return []

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    pre_relations = [pre_relations]
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_relations = [rel['label'] for rel in relations['head']]
    tail_relations = [rel['label'] for rel in relations['tail']]
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head:
        tail_relations = set(tail_relations) - set(pre_relations)
        head_relations = set(head_relations)
    else:
        head_relations = set(head_relations) - set(pre_relations)

    total_relations = list(head_relations|tail_relations)
    total_relations.sort()

    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.api_key, args.model, base_url=args.base_url)
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations,args)
    if flag:
        return retrieve_relations_with_scores
    else:
        return []


def relation_search(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    # 针对一个实体，找出它作为主语或宾语的所有关系，并进行一些清理和格式化
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_relations = [rel['label'].lower() for rel in relations['head']]
    tail_relations = [rel['label'].lower() for rel in relations['tail']]
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head: # 当前实体是之前关系中的头实体，从当前获取的尾关系中移除已有的关系
        tail_relations = set(tail_relations) - set(pre_relations)
    else:
        head_relations = set(head_relations) - set(pre_relations)

    head_relations = list(set(head_relations))
    h = [{"relation": s, 'head': True, 'entity_name': entity_name, 'entity_id': entity_id} for s in head_relations]
    tail_relations = list(set(tail_relations))
    t = [{"relation": s, 'head': False, 'entity_name': entity_name, 'entity_id': entity_id} for s in tail_relations]
    total_relations = h + t
    # e.g. { "relation": "子公司",  "head": True, "entity_name": "阿里巴巴集团", "entity_id": "Q123456"}
    return total_relations 

def run_llm_json(prompt, temperature, max_tokens, api_key, args, engine):
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "EMPTY", base_url=args.base_url, timeout=3600)
    res_format = {"type": "json_object"}

    sys_prompt = "You are a helpful assistant designed to output JSON."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    result = ""
    f = 3
    while (f > 0):
        try:
            if 'qwen3' in engine.lower():
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                temperature = 0.7
                top_p = 0.8
                response = client.chat.completions.create(
                    model=engine,
                    response_format=res_format,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    extra_body=extra_body)
            else:
                response = client.chat.completions.create(
                    model=engine,
                    response_format=res_format,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response.choices[0].message.content
            f = -1
        except Exception as e:
            print(e)
            f -= 1
            time.sleep(5)
    if f == 0:
        print('Json Generation Error')
    return result

def topic_e_prune(question, entities, args):
    def extract_output(text):

        match = re.search(r'Output:\s*(\{.*?\})', text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return ""
    def construct_topic_prune_prompt(question, entities):
        entities_json_string = json.dumps(entities)
        prompt = 'question: ' + question + '\ntopic entities:\n' + entities_json_string + '\nOutput:' # No Analysis
        return prompt


    prompt = construct_topic_prune_prompt(question, entities)
    prompt = topic_prune_demos + '\n' + prompt
    results = run_llm_json(prompt, args.temperature_exploration, args.max_length, args.api_key, args, args.model)
    if 'llama' in args.model.lower() or 'qwen' in args.model.lower():
        # json_str = extract_output(results).strip() # may not required
        try:
            results = json.loads(results)
        except ValueError:
            print("Entity prune failed, output original entities")
            return entities
    else:
        try:
            results = json.loads(results)
        except Exception as e:
            print("Entity prune failed, output original entities. Error result:{}. Error Info{}".format(results, e))
            return entities

    return results


def all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def entity_search(entity_id, relation, wiki_client, head):
    rid = wiki_client.query_all("label2pid", relation)
    if not rid or rid == "Not Found!":
        return []

    rid_str = rid.pop()

    entities = wiki_client.query_all("get_tail_entities_given_head_and_relation", entity_id, rid_str)

    if head:
        entities_set = entities['tail']
    else:
        entities_set = entities['head']

    if not entities_set:
        values = wiki_client.query_all("get_tail_values_given_head_and_relation", entity_id, rid_str)
        if not isinstance(values, list):
            values = []
        candidate_list = [{'name': name, 'id': '[FINISH_ID]'} for name in list(values)]
    else:
        candidate_list = []
        for item in entities_set:
            if item['label'] != "N/A":
                find_entity_name = item['label']
                find_entity_id = item['qid']
                candidate_list.append({'name': find_entity_name, 'id': find_entity_id})

    if len(candidate_list) >= 50:
        candidate_list = random.sample(candidate_list, 50)

    return candidate_list


def entity_find(question, entity_candidates, topic_entity_name, relation, args):
    if len(entity_candidates) < 3:
        print('entity_candidates<3')
        return entity_candidates
    if len(entity_candidates) == 0:
        return []

    prompt , check_prompt= construct_entity_find_prompt(question, topic_entity_name, relation, entity_candidates)
    
    print('-----------entity_find check_prompt-----------')
    print(check_prompt)
    print('-----------entity_find end check_prompt-----------')

    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.api_key, args.model, base_url=args.base_url)
    
    print('-----------entity_find result-----------')
    print(result)

    pattern = r'\{([^}]*)\}'
    find_re = re.findall(pattern, result)

    top2_entities=[]
    for string in find_re:
        if ',' in string:
            result = string.split(',')
            top2_entities.extend(result)
        else:
            top2_entities.append(string)

    print('-----------entity_find-----------')
    print(top2_entities)

    entity_candidates_find = match_top2_entities(top2_entities,entity_candidates)

    return entity_candidates_find

def contains_yes_regex(text):
    words = re.findall(r'\b\w+\b', text.lower())

    first_100_words = words[:100]
    if 'yes' in first_100_words:
        return True
    else:
        return False
def match_top2_entities(top2_entities,entity_candidates):
    entity_candidates_find=[]

    for entity in top2_entities:
        flag=False
        for entity_candidate in entity_candidates:
            if entity==entity_candidate['name']:
                flag=True
                entity_candidates_find.append(entity_candidate)
        if not flag:
            print('top2_entities not match entity_candidate[name]')
            print('entity_candidate[name]: '+','.join([d['name'] for d in entity_candidates]))
            
    print('\n---------------check top2_entities match back entity_candidates_name ---------------')

    print(','.join([d['name'] for d in entity_candidates_find]))

    return entity_candidates_find

      
def update_history_find_entity(entity_candidates_find,relation, total_candidates):

    for entity_candidate in entity_candidates_find:
        candidate={}
        candidate['relation'] = relation['relation']
        candidate['topic_entities'] = relation['entity_name']
        if 'head' in relation:
            candidate['head'] = relation['head']
        candidate['id'] = entity_candidate['id']
        candidate['name'] = entity_candidate['name']
        candidate['related_paragraphs'] = entity_candidate['related_paragraphs']
        if 'pre_path' in entity_candidate:
            candidate["pre_path"] = entity_candidate['pre_path']

        total_candidates.append(candidate)

    return total_candidates


def para_rank_topk(question, Indepth_total_candidates, args, emb_model,k = 10):
    # 根据候选实体相关段落与给定问题的相关性，对这些候选实体进行排序和筛选，并提取出最有价值的实体及其信息
    paras = []
    for candidate in Indepth_total_candidates:
        paras.extend(candidate['related_paragraphs'])
    scores = s2p_relevance_scores(paras, question, args, emb_model)
    top_paragraphs_heap = []
    cnt = 0
    for candidate in Indepth_total_candidates:
        for paragraph in candidate['related_paragraphs']:
            heapq.heappush(top_paragraphs_heap, (scores[cnt], paragraph, candidate['name'], candidate['id'], candidate['relation'], candidate['topic_entities'], candidate['head']))
            cnt += 1
            # 仅保留k个最相关的段落
            if len(top_paragraphs_heap) > k:
                heapq.heappop(top_paragraphs_heap)
    top_k_paragraphs = []
    while top_paragraphs_heap:
        score, text, e_name, e_id, rel, topic_e, h, = heapq.heappop(top_paragraphs_heap)
        top_k_paragraphs.append({'entity_name': e_name, 'paragraph': text, 'score': score, 'entity_id': e_id, 'relation': rel, 'topic_entitie': topic_e, 'head': h})

    top_k_paragraphs.reverse()
    # 计算每个实体的得分
    entities_with_score = {}
    alpha = 0.8
    for rank, paragraph in enumerate(top_k_paragraphs, start=1):

        score = float(paragraph['score'])
        weight = math.exp(-alpha * rank)
        name_idx = paragraph['entity_name']
        if name_idx in entities_with_score:
            entities_with_score[name_idx] += score * weight
        else:
            entities_with_score[name_idx] = score * weight

    sorted_entities = sorted(entities_with_score.items(), key=lambda x: x[1], reverse=True)
    sorted_entity_list = []

    # 根据得分对实体进行排序，并提取出最有价值的实体及其信息
    for i in range(min(args.width, len(sorted_entities))):
        ename = sorted_entities[i][0]
        score = sorted_entities[i][1]
        flg = 0
        sents = []
        # 从最相关的k个段落中，提取该实体相关的段落，并且分为句子
        for j, paragraph in enumerate(top_k_paragraphs):
            if ename == paragraph['entity_name']:
                if flg == 0:
                    eid = paragraph['entity_id']
                    tpc_e = paragraph['topic_entitie']
                    rel = paragraph['relation']
                    h = paragraph['head']
                    flg = 1
                splited_sentences = split_sentences_windows(paragraph['paragraph'], *args.sliding_window)
                sents.extend(splited_sentences)
        sents_dict = [{'text': s, 'score': 0} for s in sents]

        # 实体结构化信息，以及最相关的段落中切割后的句子
        entity_info = {
            'id': eid,
            'name': ename,
            'entity_score': score,
            'topic_entities': tpc_e,
            'relation': rel,
            'head': h
        }
        Indepth_total_candidates.append(entity_info)
        entity_info['sentences'] = sents_dict
        sorted_entity_list.append(entity_info)


    entities_id = [d['id'] for d in sorted_entity_list]
    relations = [d['relation'] for d in sorted_entity_list]
    entities_name = [d['name'] for d in sorted_entity_list]
    topics = [d['topic_entities'] for d in sorted_entity_list]
    heads = [d['head'] for d in sorted_entity_list]

    if len(entities_id) ==0:
        return False, [], [], [], [],[]

    cluster_chain_of_entities = [(topics[i], relations[i], entities_name[i]) for i in range(len(entities_name))]

    return True, cluster_chain_of_entities, entities_id, relations, heads, sorted_entity_list, Indepth_total_candidates


def question_clearify(question, args, clue = ''):
    system_prompt = prompt_requery_clue + question + "\nClues:"+ clue +'\nOutput:'
    result = run_llm(system_prompt, 0, 512, args.api_key, args.model, base_url=args.base_url)
    result = extract_answer(result)
    if result:
        return result
    else:
        return question

def reasoning(question, Indepth_total_candidates, Total_Related_Senteces, cluster_chain_of_entities, args, clue):
    # def num_tokens_from_string(string: str) -> int:
    #     """Returns the number of tokens in a text string."""
    #     import tiktoken
    #     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    #     num_tokens = len(encoding.encode(string))
    #     return num_tokens
    # 将拓展的路径历史展平为文本
    chain_prompt = '('+')\n('.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])+')'

    check_prompt = '### Question:' + question
    if args.clue_query:
        chain_prompt += '\n' + 'clue:' + clue + '\n'
        system_prompt = prompt_reasoning_qa_query_change_2shot
    else:
        system_prompt = prompt_reasoning_qa_2shot

    # 把最相关的句子中和其原始实体关联
    sorted_sentences = Total_Related_Senteces[0:args.num_sents_for_reasoning]
    texts=[]
    for sentence in sorted_sentences:
        have_name=False
        for candidate in Indepth_total_candidates: # 实体的格式不是related_sentences，而是sentences
            if 'related_sentences' in candidate.keys(): 
                for related_sentence in candidate['related_sentences']:
                    if sentence['text'] == related_sentence['text']:
                        have_name=True
                        texts.append('Entity: '+candidate['name']+' Referrence: '+sentence['text'])
        if not have_name:
            texts.append(sentence['text'])

    # num_tokens = num_tokens_from_string(system_prompt + "\nKnowledge Triplets:\n" + chain_prompt + '\nRetrieved sentences:\n' + '\nAnswer:')
    related_sentences_prompt = ''
    if len(texts) > 0:
        for i in range(min(args.num_sents_for_reasoning, len(texts))):
            # num_tokens += num_tokens_from_string(texts[i]) + 1
            # if num_tokens < 4060:
            related_sentences_prompt += '\n' + texts[i]
            # else:
            #     break
    check_prompt += "\n### Knowledge Triplets:\n" + chain_prompt +'\n### Retrieved References:\n' + related_sentences_prompt + '\n### Answer:'

    final_prompt = system_prompt + check_prompt
    
    print('\n[REASONING] LLM Context:')
    if chain_prompt.strip():
        print('>>> Knowledge Triplets:\n' + chain_prompt.rstrip('clue:\n') + '\n')
    if related_sentences_prompt.strip():
        print('>>> Retrieved References:\n' + related_sentences_prompt.strip())

    response = run_llm(final_prompt, args.temperature_reasoning, args.max_length, args.api_key, args.model, base_url=args.base_url)

    print('\n========== [REASONING RESULT] ==========')
    print(response)
    print('========================================')

    result = extract_answer(response)
    if if_true(result) or contains_yes_regex(response):
        return True, response,final_prompt
    else:
        return False, response,final_prompt


def get_wikipedia_page(wiki_client,entity_dict, dataset):
    del dataset
    for _ in range(3):
        related_passage = wiki_client.get_wikipedia_page(entity_dict)
        
        if related_passage == "Not Found!":
            break
        elif related_passage != "Fetch Error!":
            break
        time.sleep(1)

    if related_passage == "Fetch Error!":
        related_passage = "Not Found!"

    return str(related_passage)
