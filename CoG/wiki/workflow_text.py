import re
import ast

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from wiki.api import get_text_under_section, get_full_page_details, retrieve_wiki_page
from wiki.prompt import prompt_extract_from_chunks, prompt_present_info
from wiki.format import format_retrieval_trace
from utils import generate_process


def parse_present_info_result(result_text):
    try:
        decision_match = re.search(r"Decision:\s*(EXTRACT_AND_EXPLORE|IRRELEVANT)", result_text)
        if not decision_match:
            print("Cannot find 'Decision' in result_text")
            return None
        
        decision = decision_match.group(1).strip()
        
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Info:|\nNew Query:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "not given."

        if decision == "EXTRACT_AND_EXPLORE":
            info_match = re.search(r"Extracted Info:(.*?)(?=\nRelevant Sections:|$)", result_text, re.DOTALL)
            extracted_info_raw = info_match.group(1).strip() if info_match else ""
            extracted_info = None if extracted_info_raw.lower() == 'none' else extracted_info_raw

            sections_match = re.search(r"Relevant Sections:(.*)", result_text, re.DOTALL)
            sections_str = sections_match.group(1).strip() if sections_match else "[]"
            relevant_sections = ast.literal_eval(sections_str)

            return {
                "decision": decision,
                "rationale": rationale,
                "extracted_info": extracted_info,
                "relevant_sections": relevant_sections
            }

        elif decision == "IRRELEVANT":
            query_match = re.search(r"New Query:(.*)", result_text, re.DOTALL)
            new_query = query_match.group(1).strip().strip('"') if query_match else ""
            
            return {
                "decision": decision,
                "rationale": rationale,
                "new_query": new_query
            }
            
    except Exception as e:
        print(f"Could not parse the result string: {result_text}\nerror: {e}")
        return None


def skim_wiki_page(page_title, page_summary, page_sections, question, analysis, query, args, max_retries=3):
    # 1. 准备 generate_process 所需的输入
    # 假设 list_top_level_sections 已定义
    template_inputs = {
        'question': question,
        'analysis': analysis,
        'query': query,
        'page_title': page_title,
        'page_summary': page_summary,
        'page_sections': page_sections
    }

    # 2. 调用 generate_process 来获取解析后的决策
    parsed_result = generate_process(
        step_name="Skim Wiki Page",
        prompt_template=prompt_present_info,
        template_inputs=template_inputs,
        parsing_function=parse_present_info_result,
        args=args,
        module='wiki',
        max_retries=max_retries
    )

    # 3. 如果未能获得有效决策，则中止
    if not parsed_result:
        print("\n--- [WORKFLOW FAILED] at step 'Skim Wiki Page'.---")
        return None
    return parsed_result


def parse_section_extraction_result(result_text):
    """
    Parses the output from the prompt_extract_from_chunks LLM call.
    It extracts the rationale and the information found in the text chunks.
    """
    try:
        # Extract rationale
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Info:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

        # Extract info
        info_match = re.search(r"Extracted Info:(.*)", result_text, re.DOTALL)
        extracted_info_raw = info_match.group(1).strip() if info_match else ""
        
        # Handle cases where no information is found
        if extracted_info_raw.lower() == 'none':
            extracted_info = None
        else:
            extracted_info = extracted_info_raw

        return {
            "rationale": rationale,
            "extracted_info": extracted_info
        }
    except Exception as e:
        print(f"Error parsing extraction result: {e}\nInput text: {result_text}")
        return None


def extract_info_from_sections(page, page_title, relevant_sections, rationale_for_exploration, question, analysis, query, embeddings, args, k=3):
    """
    Processes a list of Wikipedia sections using RAG to extract relevant information.
    Handles potential network errors during section text retrieval without halting the entire process.

    For each section, this function:
    1. Splits the section's text into chunks.
    2. Uses a vector store to find the most relevant chunks for a given query (RAG).
    3. Prompts an LLM with these chunks to extract key information.
    4. Parses the LLM's response and collects the results.

    Args:
        page (WikipediaPage): The Wikipedia page object.
        relevant_sections (list): A list of section titles to explore.
        rationale_for_exploration (str): The rationale for why these sections were chosen.
        question (str): The original user question.
        analysis (str): The overall plan to answer the question.
        query (str): The specific sub-query to guide the RAG and extraction.
        embeddings: The embedding model for creating vector representations.
        args: Configuration arguments for the LLM call.
        k (int): The number of top relevant chunks to retrieve.

    Returns:
        tuple: A tuple containing (status, data, has_errors).
               - On success: ("SUCCESS", list, bool), where the list contains dictionaries
                 of extracted information for each section, and the boolean indicates if any
                 non-fatal API errors occurred.
               - On critical failure: ("WIKI_API_ERROR", list, True), if all sections
                 failed to be retrieved due to network errors.
    """
    all_extracted_info_from_sections = []
    any_section_failed = False
    num_failures = 0

    if not relevant_sections:
        print("No sections were marked for exploration.")
        return "SUCCESS", all_extracted_info_from_sections, False

    for section_title in relevant_sections:
        print(f"\n--- Processing Section: '{section_title}' ---")

        # 1. Get the full text of the section with error handling
        status, section_text = get_text_under_section(page, section_title)
        
        if not status:
            # A network error occurred for this section. Log it, mark the failure, and continue.
            any_section_failed = True
            num_failures += 1
            error_message = section_text
            print(f"  -> WIKI_API_ERROR for section '{section_title}': {error_message}")
            all_extracted_info_from_sections.append({
                "section": section_title,
                "extracted_info": error_message,
                "rationale": "Failed to retrieve section text due to a network error."
            })
            continue # Continue to the next section

        if not section_text or not section_text.strip():
            print(f"  -> Section '{section_title}' is empty or could not be found. Skipping.")
            all_extracted_info_from_sections.append({
                "section": section_title,
                "extracted_info": f"Section '{section_title}' is empty or could not be found. Skipping.",
                "rationale": ''
            })
            continue

        # 2. Split the section text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(section_text)
        print(f"  -> Split section into {len(chunks)} chunks.")

        if not chunks:
            print(f"  -> No chunks were created from section '{section_title}'. Skipping.")
            continue

        # 3. Use RAG: create a vector store and retrieve the top-k most relevant chunks
        # print("  -> Creating vector store and retrieving chunks...")
        vector_store = InMemoryVectorStore.from_texts(texts=chunks, embedding=embeddings)

        num_chunks_to_retrieve = min(k, len(chunks))
        docs = vector_store.similarity_search(query, k=num_chunks_to_retrieve)
        context_chunks = "\n\n---\n\n".join([doc.page_content for doc in docs])
        print(f"  -> Retrieved top {len(docs)} relevant chunks for the query.")

        # 4. Call the LLM with the retrieved chunks to extract the final answer
        template_inputs = {
            'question': question,
            'analysis': analysis,
            'query': query,
            'page_title': page_title,
            'section_title': section_title,
            'exploration_rationale': rationale_for_exploration,
            'k': len(docs),
            'chunks': context_chunks
        }
        
        extraction_result = generate_process(
            step_name=f"Extract from Section '{section_title}' in {page_title}",
            prompt_template=prompt_extract_from_chunks,
            template_inputs=template_inputs,
            parsing_function=parse_section_extraction_result,
            args=args,
            module='wiki',
            max_retries=3
        )

        # 5. Process and store the result
        if extraction_result:
            print(f"  -> LLM Rationale: {extraction_result['rationale']}")
            print(f"  -> Extracted Info: {extraction_result['extracted_info']}")
            all_extracted_info_from_sections.append({
                "section": section_title,
                "extracted_info": extraction_result['extracted_info'],
                "rationale": extraction_result['rationale']
            })
        else:
            print(f"Error in section information extraction.")

    # If all sections failed due to network errors, it's a critical failure for this step.
    if len(relevant_sections) > 0 and num_failures == len(relevant_sections):
        return "WIKI_API_ERROR", all_extracted_info_from_sections, True

    return "SUCCESS", all_extracted_info_from_sections, any_section_failed


def retrieve_info_wikipedia(query: str, entity: str, question: str, analysis: str, args, embeddings):
    """
    Orchestrates the full process of retrieving and extracting information from Wikipedia for a single query.

    Args:
        query (str): The specific sub-query for this retrieval task.
        entity (str): The entity to search for on Wikipedia.
        question (str): The original user question for context.
        analysis (str): The high-level plan for answering the question.
        args: Configuration arguments for LLM calls.
        embeddings: The embedding model for RAG.

    Returns:
        dict: A dictionary containing the status and results of the process.
    """
    # Step 1: Find the correct Wikipedia page, handling disambiguation/not-found cases.
    status, result, trace = retrieve_wiki_page(entity, question, analysis, query, args, max_interactions=args.max_page_retrieval_interactions)
    if status == "SUCCESS":
        retrieval_trace_str = format_retrieval_trace(trace, status, result, max_options=5, max_results=5)
    else:
        retrieval_trace_str = format_retrieval_trace(trace, status, result)

    if status == "ERROR":  # This indicates a failure in query_wikipedia_api, likely a network error.
        return {
            "status": "WIKI_API_ERROR",
            "reason": result,
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": True,
        }
    elif status == "FAILED":  # This indicates a failure in the retrieval logic (e.g., LLM choice).
        return {
            "status": "RETRIEVAL_FAILED",
            "reason": result,
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": False,
        }
    
    page = result
    
    # --- New: Fetch all page details at once ---
    details_status, page_details = get_full_page_details(page, retries=5)
    if not details_status:
        # If fetching details fails, it's a critical API error. page_details contains the error message
        reason = f"Wikipedia API error when fetching page details for the query '{query}': {page_details}"
        print(reason)
        return {
            "status": "WIKI_API_ERROR",
            "reason": reason,
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": True,
        }
    
    page_title = page_details['title']
    page_summary = page_details['summary']
    page_sections = page_details['sections']
    
    # Step 2: Skim the retrieved page to decide on the next action.
    skim_result = skim_wiki_page(page_title, page_summary, page_sections, question, analysis, query, args)
    
    if not skim_result:
        return {
            "status": "SKIMMING_FAILED",
            "page_title": page_title,
            "reason": "The skimming process failed to produce a valid decision after multiple retries.",
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": False,
        }

    # Step 3: Process the skimming decision.
    decision = skim_result.get('decision')
    rationale = skim_result.get('rationale')

    if decision == "IRRELEVANT":
        return {
            "status": "IRRELEVANT_PAGE",
            "page_title": page_title,
            "rationale": rationale,
            "new_query_suggestion": skim_result.get('new_query'),
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": False,
        }

    if decision == "EXTRACT_AND_EXPLORE":
        summary_info = skim_result.get('extracted_info')
        relevant_sections = skim_result.get('relevant_sections', [])
        
        sections_info = []
        partial_wiki_error = False
        if relevant_sections:
            # Step 4: If sections are deemed relevant, extract info from them using RAG.
            sections_extraction_status, sections_extraction_result, sections_had_errors = extract_info_from_sections(
                page, page_title, relevant_sections, rationale, 
                question, analysis, query, embeddings, args, k=args.section_chunks
            )
            partial_wiki_error = sections_had_errors

            if sections_extraction_status == "WIKI_API_ERROR":
                # This means ALL sections failed, which is a critical error for this step.
                reason = f"Failed to extract text from all targeted sections for the page '{page_title}' due to network errors."
                print(reason)
                return {
                    "status": "WIKI_API_ERROR",
                    "reason": reason,
                    "retrieval_trace": retrieval_trace_str,
                    "partial_wiki_error": True,
                }
            
            sections_info = sections_extraction_result

        return {
            "status": "SUCCESS",
            "page_title": page_title,
            "summary_info": summary_info,
            "skim_rationale": rationale,
            "all_page_sections": page_sections,
            "selected_sections": relevant_sections,
            "sections_info": sections_info,
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": partial_wiki_error,
        }
    
    # Fallback for any unknown or unhandled decision from the skimming step.
    return {
        "status": "UNKNOWN_ERROR",
        "reason": f"An unknown decision '{decision}' was returned from the page skimming step.",
        "page_title": page_title,
        "retrieval_trace": retrieval_trace_str,
        "partial_wiki_error": False,
    }