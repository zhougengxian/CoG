"""Plan-on-Graph reasoning adapted to the CoG Wikidata services."""

import ast
import json
import random
import re
import time

from sentence_transformers import util

from prompts import (
    add_ent_prompt,
    answer_depth_prompt,
    answer_prompt,
    cot_prompt,
    extract_relation_prompt,
    judge_reverse,
    prune_entity_prompt,
    subobjective_prompt,
    update_mem_prompt,
)


FINISH_ID = "[FINISH_ID]"


def call_model(client, args, prompt, temperature, *, retries=5, print_output=False):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {"role": "user", "content": prompt},
    ]
    request = {"model": args.model, "messages": messages}
    if temperature is not None:
        request["temperature"] = temperature
    if args.top_p is not None:
        request["top_p"] = args.top_p
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.model == "Qwen3-32B":
        request["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        }

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(**request)
            result = completion.choices[0].message.content or ""
            if print_output:
                print(result)
            return result
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = 2 * (attempt + 1)
            print(
                f"Model request failed ({exc}); retrying in {delay}s "
                f"({attempt + 1}/{retries})"
            )
            time.sleep(delay)
    raise RuntimeError("Model request failed")


def parse_string_list(text):
    """Parse the last bracketed Python-style list containing only strings."""
    left_positions = [index for index, char in enumerate(text) if char == "["]
    right_positions = [index for index, char in enumerate(text) if char == "]"]
    for left in reversed(left_positions):
        for right in reversed(right_positions):
            if right <= left:
                continue
            try:
                value = ast.literal_eval(text[left : right + 1])
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                return value
    return None


def _extract_json_object(text):
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right <= left:
        return None
    try:
        value = json.loads(text[left : right + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _answer_to_text(answer):
    if answer is None:
        return "Null"
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (list, dict)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def extract_answer_fields(text):
    """Return answer, sufficiency, and reason from the PoG JSON response."""
    value = _extract_json_object(text)
    if value is not None:
        answer_block = value.get("A", value)
        if isinstance(answer_block, dict):
            answer = _answer_to_text(answer_block.get("Answer"))
            sufficient = str(
                answer_block.get("Sufficient", answer_block.get("Known", ""))
            )
            reason = str(value.get("R", ""))
            return answer, sufficient, reason

    answer_matches = re.findall(
        r'["\']Answer["\']\s*:\s*["\'](.*?)["\']', text, re.DOTALL
    )
    answer = answer_matches[-1] if answer_matches else None
    if answer is None:
        list_match = re.search(
            r'["\']Answer["\']\s*:\s*(\[[^\]]*\])', text, re.DOTALL
        )
        if list_match:
            answer = list_match.group(1)
    sufficient_match = re.search(
        r'["\'](?:Sufficient|Known)["\']\s*:\s*["\'](.*?)["\']',
        text,
        re.DOTALL,
    )
    reason_match = re.search(
        r'["\']R["\']\s*:\s*["\'](.*?)["\']', text, re.DOTALL
    )
    return (
        _answer_to_text(answer),
        sufficient_match.group(1) if sufficient_match else "",
        reason_match.group(1) if reason_match else "",
    )


def extract_final_answer(text):
    answer, _, _ = extract_answer_fields(text)
    if answer != "Null":
        return answer
    if _extract_json_object(text) is not None or re.search(
        r'["\']Answer["\']\s*:', text
    ):
        return "Null"
    return text.strip() or "Null"


def extract_memory(text):
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right > left:
        return text[left : right + 1]
    return text.strip()


def extract_add_and_reason(text):
    value = _extract_json_object(text)
    if value is not None:
        flag = str(value.get("Add", ""))
        reason = str(value.get("Reason", ""))
    else:
        flag_match = re.search(
            r'["\']Add["\']\s*:\s*["\'](.*?)["\']', text, re.DOTALL
        )
        reason_match = re.search(
            r'["\']Reason["\']\s*:\s*["\'](.*?)["\']',
            text,
            re.DOTALL,
        )
        flag = flag_match.group(1) if flag_match else ""
        reason = reason_match.group(1) if reason_match else ""
    print("Add:", flag)
    print("Reason:", reason)
    return "yes" in flag.lower(), reason


def retrieve_top_docs(query, docs, model, width):
    query_embedding = model.encode(query)
    document_embeddings = model.encode(docs)
    scores = util.dot_score(query_embedding, document_embeddings)[0].cpu().tolist()
    pairs = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [document for document, _ in pairs[:width]]


def print_graph_structure(graph, entity_names, truncate_limit=300):
    for topic_id, directions in graph.items():
        print(f"Entity: {entity_names.get(topic_id, topic_id)}")
        for direction, relation_map in directions.items():
            for relation, entity_ids in relation_map.items():
                names = [entity_names.get(entity_id, entity_id) for entity_id in entity_ids]
                count = len(names)
                suffix = ""
                if count > truncate_limit:
                    names = names[:truncate_limit]
                    suffix = f", ... (and {count - truncate_limit} more)"
                rendered = "[" + ", ".join(f'"{name}"' for name in names) + suffix + "]"
                print(f"  [{direction}] {relation}: {rendered}")
    print()


def _format_chains(chains):
    return "\n".join(
        " -> ".join(str(value) for value in triple)
        for depth_chain in chains
        for triple in depth_chain
    )


def break_question(question, args, llm_client):
    response = call_model(
        llm_client,
        args,
        subobjective_prompt + question,
        args.temperature_reasoning,
    )
    parsed = parse_string_list(response)
    return parsed if parsed is not None else response.strip()


def abandon_relation(relation):
    end_words = (
        " ID",
        " code",
        " number",
        "instance of",
        "website",
        "URL",
        "inception",
        "image",
        " rate",
        " count",
    )
    useless = {
        "category's main topic",
        "topic's main category",
        "stack exchange site",
        "main subject",
        "country of citizenship",
        "commons category",
        "commons gallery",
        "country of origin",
        "country",
        "nationality",
    }
    lowered = relation.lower()
    return (
        relation.endswith(end_words)
        or "wikidata" in lowered
        or "wikimedia" in lowered
        or lowered in useless
    )


def select_relations(text, entity_id, head_map, tail_map):
    relation_labels = parse_string_list(text)
    if relation_labels is None:
        return []
    selected = []
    for label in relation_labels:
        if label in head_map:
            selected.append(
                {
                    "entity": entity_id,
                    "relation": label,
                    "relation_pid": head_map[label],
                    "head": True,
                }
            )
        elif label in tail_map:
            selected.append(
                {
                    "entity": entity_id,
                    "relation": label,
                    "relation_pid": tail_map[label],
                    "head": False,
                }
            )
    return selected


def relation_search_prune(
    entity_id,
    sub_questions,
    entity_name,
    previous_relations,
    previous_head,
    question,
    args,
    wiki_client,
    llm_client,
):
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_items = list(relations.get("head", []))
    tail_items = list(relations.get("tail", []))
    if args.remove_unnecessary_rel:
        head_items = [item for item in head_items if not abandon_relation(item["label"])]
        tail_items = [item for item in tail_items if not abandon_relation(item["label"])]

    if previous_head != -1:
        if previous_head:
            tail_items = [
                item for item in tail_items if item["label"] not in previous_relations
            ]
        else:
            head_items = [
                item for item in head_items if item["label"] not in previous_relations
            ]

    head_map = {item["label"]: item["pid"] for item in head_items}
    tail_map = {item["label"]: item["pid"] for item in tail_items}
    relation_labels = sorted(set(head_map) | set(tail_map))
    if not relation_labels:
        return []

    prompt = (
        extract_relation_prompt
        + question
        + "\nSubobjectives: "
        + str(sub_questions)
        + "\nTopic Entity: "
        + entity_name
        + "\nRelations: "
        + "; ".join(relation_labels)
    )
    response = call_model(
        llm_client, args, prompt, args.temperature_exploration
    )
    return select_relations(response, entity_id, head_map, tail_map)


def entity_search(entity_id, relation_pid, head, wiki_client):
    result = wiki_client.query_all(
        "get_tail_entities_given_head_and_relation", entity_id, relation_pid
    )
    candidates = list(result.get("tail" if head else "head", []))
    if head:
        values = wiki_client.query_all(
            "get_tail_values_given_head_and_relation", entity_id, relation_pid
        )
        candidates.extend({"qid": value, "label": value} for value in values)
    return candidates


def _candidate_names_and_ids(candidates):
    pairs = sorted((item["label"], item["qid"]) for item in candidates)
    return [name for name, _ in pairs], [entity_id for _, entity_id in pairs]


def entity_condition_prune(
    question,
    graph,
    entity_names,
    name_to_id,
    args,
    embedding_model,
    llm_client,
):
    filtered_graph = {}
    selected_records = []

    for topic_id, directions in sorted(graph.items()):
        for direction, relation_map in sorted(directions.items()):
            for relation, original_ids in sorted(relation_map.items()):
                entity_ids = list(original_ids)
                all_literals = all(not str(item).startswith("Q") for item in entity_ids)
                if all_literals or len(entity_ids) <= 1:
                    selected_names = [entity_names[entity_id] for entity_id in sorted(entity_ids)]
                else:
                    names = [entity_names.get(entity_id, entity_id) for entity_id in entity_ids]
                    garbage = all(
                        (name.startswith("Q") and name[1:].isdigit()) or name == "N/A"
                        for name in names
                    )
                    if garbage and len(entity_ids) > args.garbage_sample_limit:
                        entity_ids = random.sample(entity_ids, args.garbage_sample_limit)
                    if len(entity_ids) > args.candidate_pool_limit:
                        entity_ids = random.sample(entity_ids, args.candidate_pool_limit)
                    if len(entity_ids) > args.semantic_prune_limit:
                        ranked_names = [entity_names[entity_id] for entity_id in entity_ids]
                        top_names = retrieve_top_docs(
                            question,
                            ranked_names,
                            embedding_model,
                            args.semantic_prune_limit,
                        )
                        entity_ids = [name_to_id[name] for name in top_names if name in name_to_id]

                    candidate_names = [entity_names[entity_id] for entity_id in sorted(entity_ids)]
                    prompt = (
                        prune_entity_prompt
                        + question
                        + "\nTriples: "
                        + entity_names[topic_id]
                        + " "
                        + relation
                        + " "
                        + str(candidate_names)
                    )
                    response = call_model(
                        llm_client, args, prompt, args.temperature_reasoning
                    )
                    parsed = parse_string_list(response)
                    if parsed is None:
                        print("Unable to parse entity-pruning result:", response)
                        parsed = []
                    selected_names = sorted(
                        name for name in parsed if name in candidate_names
                    )

                for name in selected_names:
                    entity_id = name_to_id.get(name)
                    if entity_id is None:
                        continue
                    filtered_graph.setdefault(topic_id, {}).setdefault(direction, {}).setdefault(
                        relation, []
                    ).append(entity_id)
                    selected_records.append(
                        {
                            "topic_id": topic_id,
                            "topic_name": entity_names[topic_id],
                            "relation": relation,
                            "entity_id": entity_id,
                            "entity_name": name,
                            "head": direction == "head",
                        }
                    )

    chain = [
        (record["topic_name"], record["relation"], record["entity_name"])
        for record in selected_records
    ]
    return selected_records, chain, filtered_graph


def update_memory(
    question,
    sub_questions,
    memory,
    graph,
    entity_names,
    args,
    llm_client,
):
    lines = []
    for topic_id, directions in sorted(graph.items()):
        for _, relation_map in sorted(directions.items()):
            for relation, entity_ids in sorted(relation_map.items()):
                names = [entity_names[entity_id] for entity_id in sorted(entity_ids)]
                lines.append(f"{entity_names[topic_id]} -> {relation} -> {names}")
    prompt = (
        update_mem_prompt
        + question
        + "\nSubobjectives: "
        + str(sub_questions)
        + "\nMemory: "
        + memory
        + "\nKnowledge Triplets:\n"
        + "\n".join(lines)
    )
    response = call_model(llm_client, args, prompt, args.temperature_reasoning)
    new_memory = extract_memory(response)
    print("Updated memory:", new_memory)
    return new_memory


def reason_on_graph(question, memory, graph, entity_names, args, llm_client):
    lines = []
    for topic_id, directions in sorted(graph.items()):
        for _, relation_map in sorted(directions.items()):
            for relation, entity_ids in sorted(relation_map.items()):
                names = [entity_names[entity_id] for entity_id in sorted(entity_ids)]
                lines.append(f"{entity_names[topic_id]} -> {relation} -> {names}")
    prompt = (
        answer_depth_prompt
        + question
        + "\nMemory: "
        + memory
        + "\nKnowledge Triplets:\n"
        + "\n".join(lines)
    )
    response = call_model(llm_client, args, prompt, args.temperature_reasoning)
    answer, sufficient, reason = extract_answer_fields(response)
    print("Sufficient:", sufficient)
    print("Reason:", reason)
    print("Answer:", answer)
    return response, answer, sufficient


def generate_answer(question, chains, args, llm_client):
    prompt = (
        answer_prompt
        + question
        + "\nKnowledge Triplets: "
        + _format_chains(chains)
    )
    return call_model(
        llm_client, args, prompt, args.temperature_reasoning, print_output=True
    )


def generate_direct_answer(question, args, llm_client):
    return call_model(
        llm_client,
        args,
        cot_prompt + question,
        args.temperature_reasoning,
        print_output=True,
    )


def choose_backtracking_entities(
    question,
    frontier_ids,
    graphs_by_depth,
    entity_names,
    name_to_id,
    memory,
    chains,
    args,
    embedding_model,
    llm_client,
):
    current_ids = [
        entity_id
        for entity_id in frontier_ids
        if entity_id != FINISH_ID and str(entity_id).startswith("Q")
    ]
    all_entity_ids = set()
    for graph in graphs_by_depth.values():
        for topic_id, directions in graph.items():
            all_entity_ids.add(topic_id)
            for relation_map in directions.values():
                for entity_ids in relation_map.values():
                    candidates = list(entity_ids)
                    names = [str(entity_names.get(item, item)) for item in candidates]
                    garbage = all(
                        (name.startswith("Q") and name[1:].isdigit()) or name == "N/A"
                        for name in names
                    )
                    if garbage and len(candidates) > args.garbage_sample_limit:
                        candidates = random.sample(candidates, args.garbage_sample_limit)
                    if len(candidates) > args.candidate_pool_limit:
                        candidates = random.sample(candidates, args.candidate_pool_limit)
                    if len(candidates) > args.semantic_prune_limit:
                        candidate_names = [entity_names[item] for item in candidates]
                        top_names = retrieve_top_docs(
                            question,
                            candidate_names,
                            embedding_model,
                            args.semantic_prune_limit,
                        )
                        candidates = [name_to_id[name] for name in top_names if name in name_to_id]
                    all_entity_ids.update(
                        item for item in candidates if str(item).startswith("Q")
                    )

    current_names = sorted({entity_names[item] for item in current_ids})
    prompt = (
        judge_reverse
        + question
        + "\nEntities set to be retrieved: "
        + str(current_names)
        + "\nMemory: "
        + memory
        + "\nKnowledge Triplets:"
        + _format_chains(chains)
    )
    response = call_model(llm_client, args, prompt, args.temperature_reasoning)
    print("\nJudge Backtracking:")
    should_add, reason = extract_add_and_reason(response)
    if not should_add:
        return current_ids, []

    other_ids = sorted(all_entity_ids - set(current_ids))
    other_names = [entity_names[item] for item in other_ids]
    prompt = (
        add_ent_prompt
        + question
        + "\nReason: "
        + reason
        + "\nCandidate Entities: "
        + str(sorted(other_names))
        + "\nMemory: "
        + memory
    )
    response = call_model(llm_client, args, prompt, args.temperature_reasoning)
    selected_names = parse_string_list(response)
    if selected_names is None:
        print("Unable to parse backtracking entities:", response)
        return current_ids, []
    selected_ids = sorted(
        name_to_id[name] for name in selected_names if name in other_names
    )
    return current_ids, selected_ids


def add_previous_info(added_ids, graphs_by_depth, filtered_graph):
    relations = []
    heads = []
    for entity_id in sorted(added_ids):
        found = False
        for graph in graphs_by_depth.values():
            for topic_id, directions in graph.items():
                for direction, relation_map in directions.items():
                    for relation, entity_ids in relation_map.items():
                        if entity_id not in entity_ids:
                            continue
                        filtered_graph.setdefault(topic_id, {}).setdefault(
                            direction, {}
                        ).setdefault(relation, [])
                        if entity_id not in filtered_graph[topic_id][direction][relation]:
                            filtered_graph[topic_id][direction][relation].append(entity_id)
                        if not found:
                            relations.append(relation)
                            heads.append(direction == "head")
                            found = True
        if not found:
            relations.append("")
            heads.append(-1)
            filtered_graph.setdefault(entity_id, {})
    return sorted(added_ids), relations, heads, filtered_graph


def answer_question(record, args, wiki_client, llm_client, embedding_model):
    question = record["question"]
    topic_entities = dict(record.get("topic_entity", {}))
    sub_questions = break_question(question, args, llm_client)
    if not topic_entities:
        print("No topic entity; answering without explored paths.")
        return extract_final_answer(generate_direct_answer(question, args, llm_client))

    entity_names = dict(topic_entities)
    name_to_id = {name: entity_id for entity_id, name in topic_entities.items()}
    chains = []
    graphs_by_depth = {}
    memory = ""
    previous_relations = []
    previous_heads = [-1] * len(topic_entities)
    backtracked_ids = set()
    backtrack_count = 0

    print("Exploration Plan:", sub_questions)
    for depth in range(1, args.depth + 1):
        print(f"\nDepth {depth}:")
        print("Topic entities:", topic_entities)
        selected_relations = []
        for index, (entity_id, entity_name) in enumerate(topic_entities.items()):
            if entity_id == FINISH_ID:
                continue
            previous_head = previous_heads[index] if index < len(previous_heads) else -1
            selected_relations.extend(
                relation_search_prune(
                    entity_id,
                    sub_questions,
                    entity_name,
                    previous_relations,
                    previous_head,
                    question,
                    args,
                    wiki_client,
                    llm_client,
                )
            )

        graph = {}
        for relation in selected_relations:
            candidates = entity_search(
                relation["entity"],
                relation["relation_pid"],
                relation["head"],
                wiki_client,
            )
            if not candidates:
                continue
            candidate_names, candidate_ids = _candidate_names_and_ids(candidates)
            entity_names.update(zip(candidate_ids, candidate_names))
            name_to_id.update(zip(candidate_names, candidate_ids))
            direction = "head" if relation["head"] else "tail"
            target = graph.setdefault(relation["entity"], {}).setdefault(
                direction, {}
            ).setdefault(relation["relation"], [])
            existing_ids = set(target)
            for candidate_id in candidate_ids:
                if candidate_id not in existing_ids:
                    target.append(candidate_id)
                    existing_ids.add(candidate_id)

        graphs_by_depth[depth] = graph
        print("Retrieved Entities:")
        print_graph_structure(graph, entity_names)
        if not graph:
            response = generate_answer(question, chains, args, llm_client)
            return extract_final_answer(response)

        selected, chain, filtered_graph = entity_condition_prune(
            question,
            graph,
            entity_names,
            name_to_id,
            args,
            embedding_model,
            llm_client,
        )
        if not selected:
            response = generate_answer(question, chains, args, llm_client)
            return extract_final_answer(response)

        chains.append(chain)
        previous_relations = [item["relation"] for item in selected]
        previous_heads = [item["head"] for item in selected]
        print("Filtered Entities:")
        print_graph_structure(filtered_graph, entity_names)
        memory = update_memory(
            question,
            sub_questions,
            memory,
            filtered_graph,
            entity_names,
            args,
            llm_client,
        )
        response, answer, sufficient = reason_on_graph(
            question, memory, filtered_graph, entity_names, args, llm_client
        )
        solved = (
            answer.lower() not in {"null", "none"}
            and not answer.startswith("Q")
            and "yes" in sufficient.lower()
        )
        if solved:
            print(f"PoG stopped at depth {depth}.")
            return answer

        frontier_ids = [item["entity_id"] for item in selected]
        if backtrack_count < args.max_backtracks:
            frontier_ids, added_ids = choose_backtracking_entities(
                question,
                frontier_ids,
                graphs_by_depth,
                entity_names,
                name_to_id,
                memory,
                chains,
                args,
                embedding_model,
                llm_client,
            )
            added_ids = [item for item in added_ids if item not in backtracked_ids]
            if added_ids:
                backtrack_count += 1
                backtracked_ids.update(added_ids)
                added_ids, added_relations, added_heads, filtered_graph = add_previous_info(
                    added_ids, graphs_by_depth, filtered_graph
                )
                previous_relations.extend(added_relations)
                previous_heads.extend(added_heads)
                frontier_ids.extend(added_ids)

        next_topics = {
            entity_id: entity_names.get(entity_id, entity_id)
            for entity_id in frontier_ids
            if str(entity_id).startswith("Q")
        }
        if not next_topics or depth >= args.depth:
            response = generate_answer(question, chains, args, llm_client)
            return extract_final_answer(response)
        if len(next_topics) > args.max_topic_entities:
            next_topics = dict(
                random.sample(list(next_topics.items()), args.max_topic_entities)
            )
        topic_entities = next_topics

    response = generate_answer(question, chains, args, llm_client)
    return extract_final_answer(response)
