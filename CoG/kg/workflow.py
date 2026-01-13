from kg.entity_link import run_entity_disambiguation, format_candidates_compact
from kg.relation_discovery import run_relation_discovery
from kg.fact_prune import run_fact_retrieval_and_pruning
from kg.recall_entity import recall_entity_from_KG
from kg.client import MultiServerWikidataQueryClient
from langchain_huggingface import HuggingFaceEmbeddings
from utils import indent_text


# =========================================================================================
# End-to-End KG Exploration Workflow
# =========================================================================================

def run_full_kg_exploration(
    question: str, 
    analysis: str, 
    query: str, 
    entity_mention: str, 
    client: MultiServerWikidataQueryClient, 
    embeddings: HuggingFaceEmbeddings,
    args
) -> str:
    """
    Orchestrates the entire KG exploration pipeline for a single query-entity pair,
    from entity linking to final report generation.
    """
    print("="*40)
    print(f"Running Full KG Exploration for Query: \"{query}\" and Entity Mention: \"{entity_mention}\"")
    print("="*40)

    # --- Step 1: Entity Linking & Disambiguation ---
    linked_candidates = recall_entity_from_KG(
        mention=entity_mention, question=question, query=query, client=client, 
        embeddings=embeddings, context_method=args.entity_link_context, top_k=args.kg_top_k
    )
    formatted_candidates = format_candidates_compact(linked_candidates)
    print(f"--- Recalled candidates for '{entity_mention}' ---\n{formatted_candidates}\n" + "-"*40)
    
    link_status, link_result = run_entity_disambiguation(
        question=question, analysis=analysis,query=query, 
        mention=entity_mention, candidates=linked_candidates, args=args,
        prompt_version=args.entity_link_method
    )

    # Case 1: The LLM explicitly determined no candidate is a good match.
    if link_status == "NO_MATCH":
        llm_analysis = link_result.get("analysis")
        analysis_detail = f"\n- **LLM Analysis:**{indent_text(llm_analysis, '  ')}" if llm_analysis else ""
        
        # For NO_MATCH, display a limited number of top candidates to keep the summary concise.
        # This is decoupled from args.kg_top_k, which can be larger for better linking recall.
        max_candidates_to_show = 10
        limited_candidates = linked_candidates[:max_candidates_to_show]
        formatted_limited_candidates = format_candidates_compact(limited_candidates)

        omitted_info = ""
        if len(linked_candidates) > max_candidates_to_show:
            omitted_count = len(linked_candidates) - max_candidates_to_show
            omitted_info = f"\n... (and {omitted_count} more candidates were omitted)"

        return (f"KG Exploration FAILED for Mention: \"{entity_mention}\"\n"
                f"- **Reason:** No suitable entity could be linked from the Knowledge Graph.\n"
                f"- **Detail:** After reviewing potential matches, the system determined that no available entity was a satisfactory fit for the mention '{entity_mention}' in the context of the question. "
                f"The top candidates considered were:\n{formatted_limited_candidates}{omitted_info}"
                f"{analysis_detail}")

    # Case 2: Any other type of failure during entity linking.
    if link_status != "SUCCESS":
        return f"KG Exploration FAILED for Mention: \"{entity_mention}\"\n- **Reason:** Entity linking failed. Detail: {str(link_result)}"

    # Case 3: Success. `link_result` is the selected entity dictionary. Proceed with exploration.
    # --- Step 2: Relation Discovery ---
    discovery_status, discovery_result = run_relation_discovery(
        question=question, analysis=analysis, query=query,
        linked_entity=link_result, client=client, args=args
    )

    if discovery_status != "SUCCESS":
        return f"KG Exploration FAILED for Entity: \"{link_result.get('label', entity_mention)}\"\n- **Reason:** Relation discovery failed. Detail: {discovery_result.get('reason', 'Unknown error')}"

    # --- Step 3: Fact Retrieval and Pruning ---
    pruning_status, final_report = run_fact_retrieval_and_pruning(
        question=question, analysis=analysis, query=query,
        entity_mention=entity_mention,
        linked_candidates=linked_candidates,
        linked_entity=link_result,
        discovery_result=discovery_result,
        client=client, args=args,
        max_retries=args.fact_pruning_retries,
    )

    if pruning_status != "SUCCESS":
        return f"KG Exploration FAILED for Entity: \"{link_result.get('label', entity_mention)}\"\n- **Reason:** Fact pruning failed. Detail: {final_report}"
        
    return final_report