import json
import re


def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        for item in datas:
            if 'qid_topic_entity' in item and item['qid_topic_entity']:
                 item['entities'] = item['qid_topic_entity']
            elif 'topic_entity' in item:
                 item['entities'] = item['topic_entity']
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        for item in datas:
             if 'qid_topic_entity' in item:
                 item['entities'] = item['qid_topic_entity']
    elif dataset_name == 'hotpot_e':
        with open('../data/hotpotadv_dev.json', encoding='utf-8') as file:
            datas = json.load(file)
        question_string = 'question'
        try:
            with open('../data/hotpotadv_entities_azure.json', encoding='utf-8') as f:
                ent_data = json.load(f)
            ent_map = {x.get('question'): x.get('entities', {}) for x in ent_data}
            for item in datas:
                q = item.get(question_string)
                if q in ent_map:
                    item['entities'] = ent_map[q]
        except FileNotFoundError:
            pass
    elif dataset_name == 'webqsp':
        with open('../data/webqsp_test.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        for item in datas:
             if 'qid_topic_entity' in item:
                 item['entities'] = item['qid_topic_entity']
    elif dataset_name == '2wiki':
        with open('../data/2wikimultihopqa.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        try:
            with open('../data/2wikimultihopqa_entities_azure.json', encoding='utf-8') as f:
                ent_data = json.load(f)
            ent_map = {x.get('question'): x.get('entities', {}) for x in ent_data}
            for item in datas:
                q = item.get(question_string)
                if q in ent_map:
                    item['entities'] = ent_map[q]
        except FileNotFoundError:
            pass
    elif dataset_name == 'KGQAGen':
        with open('../data/KGQAGen-10k.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        try:
            with open('../data/KGQAGen-10k_entities_azure.json', encoding='utf-8') as f:
                ent_data = json.load(f)
            ent_map = {x.get('question'): x.get('entities', {}) for x in ent_data}
            for item in datas:
                q = item.get(question_string)
                if q in ent_map:
                    item['entities'] = ent_map[q]
        except FileNotFoundError:
            pass
    elif dataset_name == 'musique':
        with open('../data/musique.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        try:
            with open('../data/musique_entities_azure.json', encoding='utf-8') as f:
                ent_data = json.load(f)
            ent_map = {x.get('question'): x.get('entities', {}) for x in ent_data}
            for item in datas:
                q = item.get(question_string)
                if q in ent_map:
                    item['entities'] = ent_map[q]
        except FileNotFoundError:
            pass
    else:
        print("dataset not found")
        exit(-1)
    
    with open(output_file, encoding='utf-8') as f:
        try:
            output_datas = json.load(f)
        except json.JSONDecodeError:
            # Try reading as JSONL
            f.seek(0)
            output_datas = [json.loads(line) for line in f]
            
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, origin_data=None, data_question_string='question'):
    answer_list= []
    if origin_data is None:
        origin_data = [j for j in ground_truth_datas if j[question_string] == data[data_question_string]][0]
    else:
        assert origin_data[question_string] == data[data_question_string], f'not matching origin_data <-> data:\n{origin_data}\n<----->\n{data}'
    if dataset_name == 'cwq':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == '2wiki' or dataset_name == 'KGQAGen' or dataset_name == 'musique':
        answer_list = origin_data["answer"]
    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
    elif dataset_name == 'hotpot_e':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'webqsp':
        answer_list = origin_data["answers"]



    return list(set(answer_list))
    
def check_string(string):
    return "{" in string

def clean_results(string):
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"
    

def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    if not clean_result:  # An empty response should not be considered a match.
        return False
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False

def save_result2json(dataset_name, num_right, num_error, total_nums, method):
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    with open('ToG_{}_results.json'.format(dataset_name), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
                     
def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'
