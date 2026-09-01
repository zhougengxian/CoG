"""CoG release entry point for the ToG-2 baseline.

Adapted from IDEA-FinAI/ToG-2. Modified to use CoG datasets and services, stream JSONL results.
"""

import argparse
import os
from pathlib import Path

from client import MultiServerWikidataQueryClient
from dataset import DATASET_FILES, default_data_dir, load_dataset
from search import (
    append_result,
    get_output_filename,
    load_completed_questions,
    pages_embedding_search,
    pages_embedding_search_only_para,
    s2p_relevance_scores,
    scores_rank,
)
from utils import generate_only_with_gpt, if_finish_list, run_llm, self_consistency
from wiki_func import *


def main_wiki_new(original_question, topic_entity, ground_truth, data_point, args, emb_model, wiki_client):
    clue = ''
    question = original_question
    print('\n')
    print('Question:   ' + question)
    if not topic_entity:
        print('topic_entity: None')
    elif isinstance(topic_entity, dict):
        print('topic_entity: ' + str(', '.join([f"{k} ({v})" for k, v in topic_entity.items()])))
    else:
        print('topic_entity: ' + str(','.join(topic_entity)))
    print('')
    cluster_chain_of_entities = []
    search_entity_list = []
    Total_Related_Senteces = []
    
    if args.self_consistency: # 如果使用自一致性
        if data_point["cot_sc_score"] >= args.self_consistency_threshold:
            return data_point["cot_sc_response"], search_entity_list, [], [], 'gpt self-consistency', ''

    if len(topic_entity) == 0 or args.gpt_only: # 如果主题实体为空或只使用GPT生成答案
        answer = generate_only_with_gpt(question, args)
        if args.gpt_only:
            endmode = 'generate_only_with_gpt'
        else:
            endmode = 'generate_without_explored_paths'
        remark = 'no_topic_entity'
        print(remark)
        print('Answer (CoT): ', answer)
        return answer, search_entity_list, [], [], endmode, remark

    if args.topic_prune and len(topic_entity) > 2: # 如果需要进行主题实体修剪，并且主题实体数量大于2

        topic_entity = topic_e_prune(question, topic_entity, args)
        print('>>> Topic entities retained after pruning: ', topic_entity)

        if len(topic_entity) == 0:
            answer = generate_only_with_gpt(question, args)
            endmode = 'generate_without_explored_paths.'
            remark = 'no_topic_entity_tp'
            print(remark)
            return answer, search_entity_list, [], [], endmode, remark
    else:
        # print("No topic prune.")
        pass

    for entity_id in topic_entity:
        if entity_id != "[FINISH_ID]":
            entity_name = topic_entity[entity_id]
            related_passage = get_wikipedia_page(wiki_client, {'name': entity_name, 'id': entity_id}, args.dataset)
            paragraph, sorted_sentences = pages_embedding_search(question, related_passage, args,
                                                                 emb_model, top_k=3)
            Total_Related_Senteces.extend(sorted_sentences)


    if args.depth == 0: # 如果深度为0，则根据搜索的内容直接生成答案
        references = ''
        if len(Total_Related_Senteces) > 0:
            references += "# References \n"
            for idx, s in enumerate(Total_Related_Senteces[:args.num_sents_for_reasoning]):
                references += s["text"].strip() + '\n'
        check_prompt = '### Question:' + question
        system_prompt = vanilla_prompt_reasoning_qa_2shot
        final_prompt = system_prompt + '\n' + check_prompt + '\n' + references + '\n'
        answer = run_llm(final_prompt, 0, 512, args.api_key, args.model, base_url=args.base_url)
        return answer, [], [], [], '', ''

    pre_relations = [''] * len(topic_entity)
    pre_heads = [-1] * len(topic_entity)

    for depth in range(1, args.depth + 1):
        print(f'\n[DEPTH {depth}] Starting multi-hop exploration...')
        current_entity_relations_list = []
        all_entity_relations = {}
        # 遍历每个主题实体，获取其关系列表
        for index, entity_id in enumerate(topic_entity):
            if entity_id != "[FINISH_ID]":
                print(f'    [Entity] {topic_entity.get(entity_id, "")} ({entity_id})')
                if args.relation_prune:
                    if args.relation_prune_combination: # 集合所有关系后进行修剪
                        retrieve_relations = relation_search(entity_id, topic_entity[entity_id], pre_relations[index],
                                                             pre_heads[index], question, args, wiki_client)
                        all_entity_relations[topic_entity[entity_id]] = retrieve_relations
                        print('        - Initial number of relations retrieved: ', len(retrieve_relations))
                    else: # 逐个关系进行修剪
                        retrieve_relations_with_scores = relation_search_prune(entity_id, topic_entity[entity_id],
                                                                               pre_relations[index], pre_heads[index],
                                                                               question, args,
                                                                               wiki_client)
                        for relation in retrieve_relations_with_scores:
                            relation['entity_id'] = entity_id
                            relation['entity_name'] = topic_entity[entity_id]
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                else:
                    retrieve_relations = relation_search(entity_id, topic_entity[entity_id], pre_relations[index],
                                                         pre_heads[index], question, args, wiki_client)

                    print('        - 初始获取到的关系数量: ', len(retrieve_relations))
                    current_entity_relations_list.extend(retrieve_relations)

        if args.relation_prune_combination and args.relation_prune: # 集合所有关系后进行修剪
            current_entity_relations_list.extend(
                relation_prune_all(all_entity_relations, question, args))

        print(f'\n    >>> For entities {",".join(topic_entity.values())}, {len(current_entity_relations_list)} relation paths retained after pruning:')
        for i, rel in enumerate(current_entity_relations_list):
            head_str = "-->" if rel.get('head', True) else "<--"
            print(f"        {i+1}. [{rel.get('score', 'N/A')}分] {rel.get('entity_name')} {head_str} [{rel.get('relation')}]")

        if depth == 1 and len(current_entity_relations_list) == 0:
            answer = generate_only_with_gpt(question, args)
            remark = 'WiKi Error: cant find relation of first topic_entity. Depth 1 '
            print(remark, ": ", question)
            end_mode = 'generate_only_with_gpt'

            return answer, search_entity_list, Total_Related_Senteces, [], end_mode, remark

        Indepth_total_candidates = []
        each_relation_right_entityList = []
        # 搜索每个关系对应的实体
        for relation in current_entity_relations_list:
            dir_str = "-->" if relation.get('head', True) else "<--"
            print(f"\n    >>> Exploring along path: {relation.get('entity_name')} {dir_str} [{relation.get('relation')}]")
            # 查找下一跳实体
            if relation['head']: # 根据头实体和关系，获取尾实体
                entity_candidates = entity_search(relation['entity_id'], relation['relation'], wiki_client, True)
            else:
                entity_candidates = entity_search(relation['entity_id'], relation['relation'], wiki_client, False)
            if len(entity_candidates) == 0:
                continue

            cand_strs = []
            for cand in entity_candidates[:15]:
                if cand['id'] == '[FINISH_ID]':
                    cand_strs.append(cand['name'])
                else:
                    cand_strs.append(f"{cand['name']} ({cand['id']})")
            if len(entity_candidates) > 15:
                print(f"        * Found {len(entity_candidates)} candidate entities. (Top 15): {', '.join(cand_strs)}")
            else:
                print(f"        * Found {len(entity_candidates)} candidate entities: {', '.join(cand_strs)}")

            entity_candidates = [candidate for candidate in entity_candidates if len(candidate['name']) > 2]
            candidates_before_doc_filter = len(entity_candidates)
            print(f'        * Collecting Wikipedia documents for candidate entities...')
            # 获取实体的wiki页面
            for candidate in entity_candidates:
                if candidate['id'] != '[FINISH_ID]':
                    related_passage = get_wikipedia_page(wiki_client, candidate, args.dataset)
                    paragraphs = pages_embedding_search_only_para(related_passage) # 分割wiki页面为段落列表
                    candidate['related_paragraphs'] = paragraphs
                else:
                    candidate['related_paragraphs'] = []

            entity_candidates = [candidate for candidate in entity_candidates if
                                 bool(candidate.get('related_paragraphs'))] # 只保留有相关段落的实体
            print(f"        * Candidates with usable documents: {len(entity_candidates)} / {candidates_before_doc_filter}")
            # 更新历史查找的实体的结构化表示
            Indepth_total_candidates = update_history_find_entity(entity_candidates, relation, Indepth_total_candidates)
            # 记录当前关系和对应的实体列表
            each_relation_right_entityList.append({'current_relation': relation, 'right_entity': entity_candidates})

        search_entity_list.append({'depth': depth, 'current_entity_relations_list': current_entity_relations_list,
                                   'each_relation_right_entityList': each_relation_right_entityList})

        if len(Indepth_total_candidates) == 0:
            if depth:
                answer = generate_only_with_gpt(question, args)
                remark = 'no entity find in depth{}'.format(depth)
                end_mode = 'generate_only_with_gpt'
                print(remark)
                print('Answer (CoT): ', answer)
                return answer, search_entity_list, Total_Related_Senteces, [], end_mode, remark

        # 根据实体段落，对实体进行排序，保留宽度数量个实体，并以topk相关段落中的句子作为实体描述
        # 过滤后未利用实体的属性
        flag, chain_of_entities, entities_id, pre_relations, pre_heads, sorted_entity_list, Indepth_total_candidates = para_rank_topk(
            question, Indepth_total_candidates, args, emb_model)
        print('\n[EVALUATION] Entity scoring and filtering at current depth (Entities after prune):')
        for i, e in enumerate(sorted_entity_list):
            print(f"    {i+1}. [{e['entity_score']:.2f}] {e['name']} ({e['id']})")
            if e.get('sentences') and len(e['sentences']) > 0:
                evidence = e['sentences'][0]['text'].replace('\n', ' ').strip()
                if len(evidence) > 120:
                    evidence = evidence[:120] + '...'
                print(f"       * Evidence preview: {evidence}")

        cluster_chain_of_entities.append(chain_of_entities)
        if flag: # 排序后，有实体可以推理和拓展
            # 将实体的topk段落中句子和前面的找到的相关句子进行合并
            for entity in sorted_entity_list:
                s = entity['sentences']
                Total_Related_Senteces.extend(s)
            Total_Related_Senteces = list({sentence['text']: sentence for sentence in Total_Related_Senteces}.values())
            sents = [s['text'] for s in Total_Related_Senteces]
            scores = s2p_relevance_scores(sents, question, args, emb_model)
            # 根据问题和汇聚的相关句子，进行打分排序
            Total_Related_Senteces = scores_rank(scores, sents)
            # 结合当前推理路径、最相关的sents个句子进行推理
            stop, answer, kg_prompt = reasoning(original_question, Indepth_total_candidates, Total_Related_Senteces,
                                                cluster_chain_of_entities, args, clue)
            if stop:
                print(f"\n[STOP] Answer found, reasoning stopped at Depth {depth}.")
                end_mode = 'reasoning stop'
                remark = "Find answer. ToG stoped at depth %d." % depth

                return answer, search_entity_list, Total_Related_Senteces, cluster_chain_of_entities, end_mode, remark
            else:
                print(f"\n[CONTINUE] Evidence from Depth {depth} is insufficient, continuing deeper exploration.")
                flag_finish, entities_id = if_finish_list(entities_id)

                if flag_finish: # 没有实体可继续拓展，本轮搜到的全是[FINISH_ID]（可能是数值实体带来的）
                    answer = generate_only_with_gpt(question, args)
                    remark = "After entity_find_prune, all entities_id == [FINISH_ID]. No new knowledge added during search depth %d, stop searching." % depth

                    end_mode = 'generate_only_with_gpt'
                    print(remark)
                    print('Answer (CoT): ', answer)
                    return answer, search_entity_list, Total_Related_Senteces, [], end_mode, remark
                else: # 有实体可继续拓展，更新主题实体为当前搜索到的实体
                    next_entity_ids = set(entities_id)
                    topic_entity = {
                        candidate['id']: candidate['name']
                        for candidate in sorted_entity_list
                        if candidate['id'] in next_entity_ids
                    }
                    continue
        else:
            remark = 'Last situation topic entity rank list in empty in depth {}, generate_only_with llm.'.format(depth)
            end_mode = 'generate_only_with_gpt'
            print(remark)
            answer = generate_only_with_gpt(question, args)
            return answer, search_entity_list, Total_Related_Senteces, [], end_mode, remark

    answer = generate_only_with_gpt(question, args)
    remark = 'Last situation.Not into depth. whether it trigger'
    end_mode = 'generate_only_with_gpt'
    print(remark)
    print('Answer (CoT): ', answer)
    return answer, search_entity_list, Total_Related_Senteces, [], end_mode, remark

class DenseEmbeddingAdapter:
    """Expose one dense-embedding interface for bge-bi and bge-m3."""

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        if model_name == "bge-bi":
            from FlagEmbedding import FlagModel

            self.model = FlagModel(
                "BAAI/bge-large-en-v1.5",
                use_fp16=False,
                devices=device,
            )
        else:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=True,
                devices=device,
            )

    def encode_queries(self, texts):
        if self.model_name == "bge-bi":
            return self.model.encode_queries(texts)
        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"]

    def encode_passages(self, texts):
        if self.model_name == "bge-bi":
            return self.model.encode(texts)
        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"]


def sliding_window_type(value):
    try:
        window_size, step_size = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sliding window must be WINDOW,STEP, for example 3,2"
        ) from exc
    if window_size < 1 or step_size < 1:
        raise argparse.ArgumentTypeError("WINDOW and STEP must be positive")
    return window_size, step_size


def build_parser():
    baseline_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the CoG ToG-2 baseline")
    parser.add_argument("--dataset", choices=list(DATASET_FILES), default="hotpot_e")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--temperature-exploration", type=float, default=0)
    parser.add_argument("--temperature-reasoning", type=float, default=0)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--remove-unnecessary-rel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument(
        "--relation-prune-model",
        default=None,
        help="Defaults to --model when omitted.",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--base-url", default="http://127.0.0.1:9040/v1")
    parser.add_argument(
        "--server-urls",
        type=Path,
        default=baseline_dir / "server_urls.txt",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["bge-bi", "bge-m3"],
        default="bge-bi",
    )
    parser.add_argument(
        "--relation-prune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relation-prune-combination",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-sents-for-reasoning", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--topic-prune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--gpt-only", choices=["cot", "io"], default=None)
    run_group.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--self-consistency-threshold", type=float, default=-1)
    parser.add_argument(
        "--clue-query",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sliding-window",
        type=sliding_window_type,
        default=(1, 1),
        metavar="WINDOW,STEP",
    )
    parser.add_argument("--output-dir", type=Path, default=baseline_dir / "results")
    parser.add_argument("--file-suffix", default="")
    return parser


def load_server_urls(path):
    with Path(path).open(encoding="utf-8") as file:
        urls = [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not urls:
        raise ValueError(f"No Wikidata server URL found in {path}")
    return urls


def run(args):
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.samples is not None and args.samples < 0:
        raise ValueError("--samples must be non-negative")

    args.relation_prune_model = args.relation_prune_model or args.model
    data, question_key = load_dataset(args.dataset, args.data_dir)
    end = len(data) if args.samples is None else min(len(data), args.start + args.samples)

    output_file = args.output_dir / get_output_filename(args)
    completed_questions = load_completed_questions(output_file)

    embedding_model = None
    wiki_client = None
    if not args.gpt_only:
        print(f"Loading embedding model: {args.embedding_model}")
        embedding_model = DenseEmbeddingAdapter(args.embedding_model, args.device)
        wiki_client = MultiServerWikidataQueryClient(load_server_urls(args.server_urls))
        wiki_client.test_connections()

    print(f"Dataset: {args.dataset}; selected rows: [{args.start}, {end})")
    print(f"Output: {output_file}")

    for index in range(args.start, end):
        item = data[index]
        question = item[question_key]
        if question in completed_questions:
            print(f"[{index}] Skip completed question")
            continue

        if args.self_consistency:
            data_point = self_consistency(question, item, index, args)
        else:
            data_point = {}

        ground_truth = item.get("answer", item.get("answers", ""))
        answer, *_ = main_wiki_new(
            question,
            item["topic_entity"],
            ground_truth,
            data_point,
            args,
            embedding_model,
            wiki_client,
        )
        append_result(output_file, question, answer)
        completed_questions.add(question)


if __name__ == "__main__":
    run(build_parser().parse_args())
