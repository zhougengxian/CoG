import numpy as np
from string import Template
from kg.client import MultiServerWikidataQueryClient
from langchain_huggingface import HuggingFaceEmbeddings

IMPORTANT_PROPERTIES = {
    # === 核心身份与分类 (Core Identity & Classification) ===
    # 用于回答“这是什么？”
    "P31": "instance of",          # 实例属于 (最核心的关系, 例如: 实体是 人、城市、大学)
    "P279": "subclass of",         # 子类属于 (用于概念层级, 例如: 城市 是 聚居地 的子类)
    "P106": "occupation",          # 职业 (用于个人, 例如: 政治家, 科学家)
    "P101": "field of work",       # 工作领域 (用于个人或概念, 例如: 机器学习, 物理学)
    "P136": "genre",               # 类型/流派 (用于作品, 例如: 摇滚乐, 科幻小说)
    "P39": "position held",        # 担任职位 (用于个人, 例如: 美国总统)
    "P921": "main subject",        # 主要主题 (描述一个作品或事件的核心内容)

    # === 个人履历与属性 (Personal Profile & Attributes) ===
    # 用于描述一个人的关键信息
    "P21": "sex or gender",        # 性别
    "P27": "country of citizenship", # 国籍
    "P569": "date of birth",       # 出生日期
    "P19": "place of birth",       # 出生地点
    "P570": "date of death",        # 逝世日期
    "P20": "place of death",       # 逝世地点
    "P69": "educated at",          # 教育背景 (例如: 剑桥大学)
    "P166": "award received",      # 所获奖项 (例如: 诺贝尔物理学奖)
    "P1412": "languages spoken, written or signed", # 使用语言

    # === 地理与行政信息 (Geographical & Administrative Info) ===
    # 用于描述地点和区域
    "P17": "country",              # 所属国家 (用于组织、地点等)
    "P131": "located in the administrative territorial entity", # 所在行政区 (例如: 巴黎 位于 法兰西岛)
    "P276": "location",            # 位置 (更通用的位置关系)
    "P150": "contains the administrative territorial entity", # 包含行政区 (P131的反向关系)
    "P1082": "population",         # 人口数量
    "P2044": "elevation above sea level", # 海拔
    "P1435": "heritage designation",# 遗产地状态 (如：世界遗产)

    # === 组织与公司信息 (Organizational & Corporate Info) ===
    # 用于描述组织、公司或机构
    "P495": "country of origin",    # 来源国 (适用于产品或公司)
    "P571": "inception",           # 成立/创建日期
    "P127": "owned by",            # 所有者
    "P108": "employer",            # 雇主 (用于个人)
    # "P856": "official website",    # 官方网站 (重要的外部链接)

    # === 作品与创作 (Works & Creation) ===
    # 用于描述书籍、电影、音乐、艺术品等
    "P50": "author",               # 作者 (用于文字作品)
    "P170": "creator",             # 创作者 (更通用的创建者)
    "P577": "publication date",    # 出版日期
    "P1476": "title",              # 标题
    "P407": "language of work or name", # 作品或名称的语言
    "P179": "part of the series",  # 系列作品
    "P161": "cast member",         # 演员
    "P1433": "published in",       # 发表/刊登于 (例如: 论文发表在《自然》)

    # === 成员与从属关系 (Membership & Affiliation) ===
    # 用于描述实体间的归属关系
    "P463": "member of",           # ...的成员 (通用)
    "P102": "member of political party", # 所属政党
    "P641": "sport",               # 从事运动 (例如: 篮球)
    "P54": "member of sports team", # 所属运动队 (例如: 洛杉矶湖人)
    
    # === 结构与事件关系 (Structural & Event Relations) ===
    # 用于描述部分-整体，以及时序和事件关系
    "P361": "part of",             # ...的一部分 (非常通用, 例如: 引擎是汽车的一部分)
    "P527": "has part or parts",   # 包含...部分 (P361的反向关系)
    "P155": "follows",             # 前接 (时序关系, 例如: 奥巴马 前接 布什)
    "P156": "followed by",         # 后继 (时序关系, P155的反向)
    "P793": "significant event",   # 重大事件 (为实体提供关键历史背景)
    "P585": "point in time",       # 时间点 (用于描述事件发生的精确时间)

    # === 家庭关系 (Family Relations) ===
    # 用于描述个人间的家庭联系
    "P22": "father",               # 父亲
    "P25": "mother",               # 母亲
    "P26": "spouse",               # 配偶
    "P40": "child",                # 子女

    # === 物理属性 (可选, Physical Attributes) ===
    # 描述实体的物理特征，对特定领域的消歧有用
    "P2048": "height",             # 高度
    "P2049": "width",              # 宽度
    "P2067": "mass",               # 质量
    "P186": "made from material",   # 构成材料
    "P274": "chemical formula",    # 化学式 (适用于化合物)
}

CONTEXT_INSTRUCTION_FOR_EMBEDDING = "Represent this context for retrieving the most relevant entity from a knowledge graph."


def get_context_triplets_properties(client, qid: str, max_relations: int = 6, num_priority_relations: int = 3, max_tails_per_relation: int = 3) -> tuple:
    """
    Fetches a sample of important triplets connected to a given entity to provide context,
    prioritizing a predefined list of important properties with fine-grained control.
    Additionally, it calculates coverage metrics to assess the quality of the sampled triplets.

    :param client: The Wikidata query client.
    :param qid: The QID of the entity to explore.
    :param max_relations: The total maximum number of relations to fetch.
    :param num_priority_relations: The desired number of relations from the important list.
    :param max_tails_per_relation: The maximum number of connected entities/values to show per relation.
    :return: A tuple containing:
             - triplets_dict (dict): The sampled triplets.
             - relation_coverage (float): The ratio of sampled relations to total relations.
             - entity_coverage_proxy (float): An averaged score of how well the sampled values represent the total values for each sampled relation.
    """
    all_relations = client.query_all("get_all_relations_of_an_entity", qid)
    if not isinstance(all_relations, dict):
        return {}, 0.0, 0.0

    head_relations = all_relations.get("head", [])
    tail_relations = all_relations.get("tail", [])
    
    relations_with_direction = []
    if head_relations:
        for rel in head_relations:
            relations_with_direction.append({"direction": "head", **rel})
    if tail_relations:
        for rel in tail_relations:
            relations_with_direction.append({"direction": "tail", **rel})

    total_relation_count = len(relations_with_direction)
    if total_relation_count == 0:
        return {}, 1.0, 1.0

    # --- Dynamic relation selection with bidirectional compensation ---
    important_pids = set(IMPORTANT_PROPERTIES.keys())
    
    priority_relations_all = [r for r in relations_with_direction if r['pid'] in important_pids]
    other_relations_all = [r for r in relations_with_direction if r['pid'] not in important_pids]
    
    priority_relations_all.sort(key=lambda x: x.get('counter', 0), reverse=True)
    other_relations_all.sort(key=lambda x: x.get('counter', 0), reverse=True)
    
    # We aim for `num_priority_relations` from priority and the rest from other,
    # but will compensate from either side if one is lacking.
    
    # 1. Get the initial list of priority relations
    selected_priority = priority_relations_all[:num_priority_relations]
    
    # 2. See how many 'other' relations we need and can get
    num_other_needed = max_relations - len(selected_priority)
    selected_other = other_relations_all[:num_other_needed]
    
    # 3. Compensate: if we still need more relations (because 'other' was insufficient),
    #    try to get more from 'priority'.
    num_still_needed = max_relations - (len(selected_priority) + len(selected_other))
    if num_still_needed > 0:
        # The starting point for extra priority relations is after the ones we've already selected.
        start_index = len(selected_priority)
        end_index = start_index + num_still_needed
        extra_priority = priority_relations_all[start_index:end_index]
        selected_priority.extend(extra_priority)
        
    sampled_relations = selected_priority + selected_other

    relation_coverage = len(sampled_relations) / total_relation_count if total_relation_count > 0 else 1.0

    triplets_dict = {"head": {}, "tail": {}}
    entity_coverage_scores = []

    for relation in sampled_relations:
        pid = relation["pid"]
        rel_label = relation["label"]
        direction = relation["direction"]

        all_tails_for_relation = []
        
        # Fetch entities
        related_entities_result = client.query_all("get_tail_entities_given_head_and_relation", qid, pid)
        if isinstance(related_entities_result, dict):
            target_key = "tail" if direction == "head" else "head"
            entities = related_entities_result.get(target_key, [])
            all_tails_for_relation.extend([entity.get('label', 'N/A') for entity in entities])

        # Fetch values (only for head direction)
        if direction == "head":
            related_values_result = client.query_all("get_tail_values_given_head_and_relation", qid, pid)
            if isinstance(related_values_result, list):
                all_tails_for_relation.extend(related_values_result)

        # Calculate entity coverage for this relation, ensuring uniqueness of tails
        if all_tails_for_relation:
            unique_tails_ordered = list(dict.fromkeys(all_tails_for_relation))
            unique_tails_count = len(unique_tails_ordered)
            
            kept_tails = unique_tails_ordered[:max_tails_per_relation]
            coverage = len(kept_tails) / unique_tails_count
            entity_coverage_scores.append(coverage)
            
            # Populate triplets dict
            if rel_label not in triplets_dict[direction]:
                triplets_dict[direction][rel_label] = []
            triplets_dict[direction][rel_label].extend(kept_tails)

    entity_coverage_proxy = sum(entity_coverage_scores) / len(entity_coverage_scores) if entity_coverage_scores else 1.0
    
    return triplets_dict, relation_coverage, entity_coverage_proxy


def get_context_triplets(client, qid: str, max_relations: int = 6, max_tails_per_relation: int = 3) -> dict:
    """
    Fetches a sample of important triplets connected to a given entity to provide context.

    :param client: The Wikidata query client.
    :param qid: The QID of the entity to explore.
    :param max_relations: The number of most popular relations to explore.
    :param max_tails_per_relation: The maximum number of connected entities/values to show per relation.
    :return: A dictionary representing the triplets, separated by direction (head/tail).
    """
    all_relations = client.query_all("get_all_relations_of_an_entity", qid)
    if not isinstance(all_relations, dict):
        return {"head": {}, "tail": {}}

    head_relations = all_relations.get("head", [])
    tail_relations = all_relations.get("tail", [])
    
    relations_with_direction = []
    if head_relations:
        for rel in head_relations:
            relations_with_direction.append({"direction": "head", **rel})
    if tail_relations:
        for rel in tail_relations:
            relations_with_direction.append({"direction": "tail", **rel})

    # Sort by popularity (counter)
    sorted_relations = sorted(relations_with_direction, key=lambda x: x.get('counter', 0), reverse=True)
    
    num_relations_to_sample = min(max_relations, len(sorted_relations))
    sampled_relations = sorted_relations[:num_relations_to_sample]

    triplets_dict = {"head": {}, "tail": {}}
    for relation in sampled_relations:
        pid = relation["pid"]
        rel_label = relation["label"]
        direction = relation["direction"]

        if direction == "head":
            if rel_label not in triplets_dict["head"]:
                triplets_dict["head"][rel_label] = []
            
            # Fetch entities
            related_entities_result = client.query_all("get_tail_entities_given_head_and_relation", qid, pid)
            if isinstance(related_entities_result, dict):
                target_entities = related_entities_result.get("tail", [])
                for entity in target_entities[:max_tails_per_relation]:
                    triplets_dict["head"][rel_label].append(entity.get('label', 'N/A'))
            
            # Fetch values
            related_values_result = client.query_all("get_tail_values_given_head_and_relation", qid, pid)
            if isinstance(related_values_result, list):
                for value in related_values_result[:max_tails_per_relation]:
                     triplets_dict["head"][rel_label].append(value)

        else: # direction == "tail"
            if rel_label not in triplets_dict["tail"]:
                triplets_dict["tail"][rel_label] = []

            related_entities_result = client.query_all("get_tail_entities_given_head_and_relation", qid, pid)
            if isinstance(related_entities_result, dict):
                target_entities = related_entities_result.get("head", [])
                for entity in target_entities[:max_tails_per_relation]:
                    triplets_dict["tail"][rel_label].append(entity.get('label', 'N/A'))
    
    return triplets_dict


def _textualize_triplets(triplets: dict) -> str:
    """
    Converts a triplets dictionary into a semi-structured, embedding-friendly string.

    Args:
        triplets (dict): A dictionary with "head" and "tail" keys, containing relations.

    Returns:
        str: A formatted string representing the entity's knowledge graph neighborhood.
    """
    fact_parts = []
    
    # Process head relations (outgoing)
    head_relations = triplets.get("head", {})
    for relation, values in head_relations.items():
        if values:
            fact_parts.append(f"{relation}: {', '.join(map(str, values))}")
            
    # Process tail relations (incoming)
    tail_relations = triplets.get("tail", {})
    for relation, values in tail_relations.items():
        if values:
            # Add a prefix to indicate an inverse relationship
            fact_parts.append(f"inverse_{relation}: {', '.join(map(str, values))}")
            
    return "; ".join(fact_parts)


def recall_entity_from_KG(
    mention: str, 
    question: str, 
    query: str, 
    client: MultiServerWikidataQueryClient, 
    embeddings: HuggingFaceEmbeddings, 
    context_method: str,
    top_k: int = 5
):
    """
    将文本指称链接到知识图谱中的实体。

    Args:
        mention (str): 要链接的名称或短语 (例如, "华盛顿").
        question (str): 完整的原始问题，用于提供实体消歧的宏观上下文。
        query (str): 当前的子查询，用于指导关系采样。
        client (MultiServerWikidataQueryClient): 用于查询知识图谱的客户端。
        embeddings (HuggingFaceEmbeddings): 用于语义相似度计算的嵌入模型。
        context_method (str): 用于构建实体消歧上下文的方法。
        top_k (int): 返回的顶部候选实体的数量。

    Returns:
        list: 按置信度排序的 top-k 候选实体列表。
              列表中的每一项都是一个包含链接实体详细信息的字典。
              如果没有找到合适的实体，则返回空列表。
    """
    # === 1. 候选生成 ===
    print(f"--- Stage 0: Entity Linking for '{mention}' ---")
    candidates = {}

    # 精确匹配
    exact_qids = client.query_all('label2qid', mention)
    if isinstance(exact_qids, list):
        for qid in exact_qids:
            candidates[qid] = {'source': 'exact', 'name_sim': 1}

    # 别名匹配
    alias_qids = client.query_all('alias2qids', mention)
    if isinstance(alias_qids, list):
        for qid in alias_qids:
            if qid not in candidates:
                candidates[qid] = {'source': 'alias', 'name_sim': 1}

    # 当精确/别名匹配无效时，进行模糊匹配
    if not candidates:
        similar_entities = client.query_all('find_similar_entities', mention, 10)
        if isinstance(similar_entities, list):
            for entity in similar_entities:
                for qid in entity['qids']:
                    if qid not in candidates:
                        # 直接使用返回的相似度分数
                        similarity = entity['score']
                        candidates[qid] = {'source': 'fuzzy', 'name_sim': similarity}
    
    if not candidates:
        print("--- No candidates found. ---")
        return []

    print(f"--- Generated {len(candidates)} candidates. Scoring... ---")

    # === 2. 候选实体重新排序 ===
    scored_candidates = []
    
    if context_method == "question_only":
        context = question
    elif context_method == "query_only":
        context = query
    elif context_method == "question_query":
        context = f"Question: {question}\nQuery: {query}"
    elif context_method == "instruct":
        if "e5" in embeddings.model_name.lower() and "instruct" in embeddings.model_name.lower():
            # E5-instruct 格式: Instruct: task\nQuery: actual_query
            text_to_embed = f"Question: {question}; Sub-query: {query}"
            context = f"Instruct: {CONTEXT_INSTRUCTION_FOR_EMBEDDING}\nQuery: {text_to_embed}"
        else:
            text_to_embed = f"Question: {question}\nQuery: {query}"
            context = f"{CONTEXT_INSTRUCTION_FOR_EMBEDDING}\n{text_to_embed}"
    else: # Default fallback, same as our recommended best practice
        context = f"Question: {question}\nQuery: {query}"
        
    context_vec = embeddings.embed_query(context)

    for qid, data in candidates.items():
        # --- 获取候选实体信息 ---
        label_list = client.query_all('qid2label', qid)
        alias_list = client.query_all('qid2aliases', qid)
        desc_list = client.query_all('qid2description', qid)
        degrees = client.query_all('get_label_degree', qid)

        # 确保我们有标签
        if not isinstance(label_list, list) or not label_list:
            continue
        label = label_list[0]
        
        # 提取描述（如果存在）
        description = desc_list[0] if isinstance(desc_list, list) and desc_list else None
        
        # 获取实体相关的三元组及采样质量
        triplets, relation_coverage, entity_coverage_proxy = get_context_triplets_properties(
            client, qid, max_relations=8, num_priority_relations=5, max_tails_per_relation=3)

        # --- 计算特征 ---
        # 1. 名称相似度 (已在候选生成阶段获得)
        name_sim = data['name_sim']

        # 2. 流行度分数
        popularity = 0
        if isinstance(degrees, dict):
            popularity = sum(degrees.values())
        pop_score = np.log1p(popularity) # log(1+x) 用于处理0值并平滑尺度

        # 3. 上下文描述相似度分数
        desc_context_sim = 0.0
        if description:
            desc_vec = embeddings.embed_documents([description])[0]
            # 假设嵌入是归一化的
            desc_context_sim = np.dot(context_vec, desc_vec)

        # 4. 上下文三元组相似度分数
        triplet_context_sim = 0.0
        if triplets and (triplets.get('head') or triplets.get('tail')):
            triplet_text = _textualize_triplets(triplets)
            if triplet_text:
                triplet_vec = embeddings.embed_documents([triplet_text])[0]
                triplet_context_sim = np.dot(context_vec, triplet_vec)

        # --- 动态权重计算 ---
        w_name = 0.3
        w_pop = 0.1
        W_CONTEXT_TOTAL = 0.6
        
        # 计算综合覆盖率
        combined_coverage = (0.7 * relation_coverage) + (0.3 * entity_coverage_proxy)
        
        # 根据覆盖率在 [0.2, 0.45] 区间内动态分配三元组权重
        w_triplet_dynamic = 0.2 + (0.45 - 0.2) * combined_coverage
        w_desc_dynamic = W_CONTEXT_TOTAL - w_triplet_dynamic

        # --- 最终分数 ---
        score = (w_name * name_sim) + \
                (w_pop * pop_score) + \
                (w_desc_dynamic * desc_context_sim) + \
                (w_triplet_dynamic * triplet_context_sim)

        scored_candidates.append({
            'qid': qid,
            'label': label,
            'description': description if description else "Not Found!",
            'degree': degrees,
            'alias': alias_list,
            'triplets': triplets,
            'final_score': score,
            'score_details': {
                'name_sim': name_sim,
                'pop_score': pop_score,
                'desc_context_sim': desc_context_sim,
                'triplet_context_sim': triplet_context_sim,
                'relation_coverage': relation_coverage,
                'entity_coverage_proxy': entity_coverage_proxy,
                'combined_coverage': combined_coverage,
                'w_triplet_dynamic': w_triplet_dynamic,
                'w_desc_dynamic': w_desc_dynamic
            }
        })
    
    # === 3. 最终选择 ===
    if not scored_candidates:
        print("--- All candidate entity scoring failed. ---")
        return []

    # 按最终分数降序排序
    sorted_candidates = sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)

    # # 应用一个分数阈值
    # score_threshold = 0.3 # 这是一个经验阈值，可能需要调整
    # top_candidate = sorted_candidates[0]
    # if top_candidate['final_score'] < score_threshold:
    #     print(f"--- 最佳候选实体的分数 ({top_candidate['final_score']:.4f}) 低于阈值 ({score_threshold})。返回 NIL。 ---")
    #     return []
        
    return sorted_candidates[:top_k]