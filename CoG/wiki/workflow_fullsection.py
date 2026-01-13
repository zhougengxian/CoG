import re

from wiki.api import get_text_under_section, get_full_page_details, retrieve_wiki_page
from wiki.workflow_text import skim_wiki_page
from wiki.workflow_table import skim_wiki_page_with_table, _serialize_full_table
from wiki.prompt import prompt_analyze_section_holistically
from wiki.table_api import get_wikipedia_tables
from wiki.format import format_retrieval_trace
from utils import generate_process


def parse_holistic_analysis_result(result_text):
    """
    Parses the output from the prompt_analyze_section_holistically LLM call.
    It extracts a unified rationale and the extracted information.
    """
    try:
        # Extract Rationale
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Info:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

        # Extract Info from the entire section (text + tables)
        info_match = re.search(r"Extracted Info:(.*)", result_text, re.DOTALL)
        extracted_info_raw = info_match.group(1).strip() if info_match else ""
        extracted_info = None if extracted_info_raw.lower() == 'none' else extracted_info_raw

        return {
            "rationale": rationale,
            "extracted_info": extracted_info
        }
    except Exception as e:
        print(f"Error parsing holistic analysis result: {e}\nInput text: {result_text}")
        return None


def extract_info_from_sections_holistically(
    page, page_title, relevant_sections, skim_rationale,
    question, analysis, query, args, tables_by_section,
    text_truncate_len=120000, table_truncate_len=20000
):
    """
    Orchestrates information extraction by performing a holistic, full-content analysis
    of each selected section, including its full text and all its tables.
    """
    all_sections_results = []
    any_section_failed = False
    num_failures = 0

    if not relevant_sections:
        print("No sections were marked for exploration.")
        return "SUCCESS", all_sections_results, False

    for section_title in relevant_sections:
        print(f"\n--- Processing Section Holistically: '{section_title}' ---")

        # 1. Get and intelligently handle section text
        status, section_text = get_text_under_section(page, section_title)
        if not status:
            # A network error occurred for this section. Log it, mark the failure, and continue.
            any_section_failed = True
            num_failures += 1
            error_message = section_text
            print(f"  -> WIKI_API_ERROR for section '{section_title}': {error_message}")
            all_sections_results.append({
                "section": section_title, "status": "WIKI_API_ERROR",
                "rationale": "Failed to retrieve section text due to a network error.",
                "extracted_info": error_message,
            })
            continue

        if not section_text or not section_text.strip():
            print(f"  -> Section '{section_title}' is empty or could not be found. Skipping.")
            all_sections_results.append({
                "section": section_title, "status": "EMPTY_SECTION", "rationale": "",
                "extracted_info": f"Section '{section_title}' is empty or could not be found.",
            })
            continue
            
        # Truncate text if it's too long
        if len(section_text) > text_truncate_len:
            section_text = section_text[:text_truncate_len] + "\n... [Section text truncated due to length] ..."
            print(f"  -> Section text is too long, truncated to {text_truncate_len} chars.")
        else:
            print(f"  -> Full section text length: {len(section_text)} chars.")

        # 2. Get and serialize all tables in the section
        tables_in_section = tables_by_section.get(section_title, [])
        all_tables_string = "No tables found in this section."
        if tables_in_section:
            print(f"  -> Found {len(tables_in_section)} tables. Serializing them.")
            serialized_tables = []
            for table_info in tables_in_section:
                table_str = _serialize_full_table(table_info, table_threshold=table_truncate_len)
                serialized_tables.append(f"Table Name: {table_info.get('table_name', '')}\n{table_str}")
            all_tables_string = "\n\n---\n\n".join(serialized_tables)
        
        # 3. Call LLM for holistic analysis
        template_inputs = {
            'question': question, 'analysis': analysis, 'query': query,
            'page_title': page_title, 'section_title': section_title,
            'exploration_rationale': skim_rationale,
            'section_text': section_text,
            'all_tables_string': all_tables_string
        }

        analysis_result = generate_process(
            step_name=f"Holistic analysis for Section '{section_title}'",
            prompt_template=prompt_analyze_section_holistically,
            template_inputs=template_inputs,
            parsing_function=parse_holistic_analysis_result,
            args=args, module='wiki', max_retries=3
        )

        if analysis_result:
            print(f"  -> LLM Rationale: {analysis_result['rationale']}")
            print(f"  -> Extracted Info: {analysis_result['extracted_info']}")
            all_sections_results.append({
                "section": section_title,
                "status": "SUCCESS",
                "rationale": analysis_result['rationale'],
                "extracted_info": analysis_result['extracted_info'],
            })
        else:
            print(f"LLM analysis Error! Cannot extract info from section '{section_title}' in holistic analysis.")
            all_sections_results.append({
                "section": section_title,
                "status": "ANALYSIS_FAILED",
                "rationale": "",
                "extracted_info": "LLM failed to produce a valid analysis for this section after multiple retries.",
            })

    if len(relevant_sections) > 0 and num_failures == len(relevant_sections):
        return "WIKI_API_ERROR", all_sections_results, True

    return "SUCCESS", all_sections_results, any_section_failed


def retrieve_info_wikipedia_full_section(query: str, entity: str, question: str, analysis: str, args):
    """
    Orchestrates the full process of retrieving and extracting information from Wikipedia
    using the holistic, full-section analysis workflow.

    Args:
        query (str): The specific sub-query for this retrieval task.
        entity (str): The entity to search for on Wikipedia.
        question (str): The original user question for context.
        analysis (str): The high-level plan for answering the question.
        args: Configuration arguments for LLM calls.

    Returns:
        dict: A dictionary containing the status and results of the process.
    """
    # Step 1: Find the correct Wikipedia page, handling disambiguation/not-found cases.
    status, result, trace = retrieve_wiki_page(entity, question, analysis, query, args, max_interactions=args.max_page_retrieval_interactions)
    if status == "SUCCESS":
        retrieval_trace_str = format_retrieval_trace(trace, status, result, max_options=5, max_results=5)
    else:
        retrieval_trace_str = format_retrieval_trace(trace, status, result)

    # Initialize a global error flag for this workflow run.
    # Any subsequent, non-critical Wikipedia API network error will set this to True.
    partial_wiki_error = False

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
    
    # Step 2: Fetch all page details and tables
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
    page_url = page_details['url']
    
    tables_status, tables_result = get_wikipedia_tables(page_url, retries=3)
    tables_by_section = {}
    
    if tables_status == "SUCCESS":
        for table_info in tables_result:
            section = table_info.get('top_section', 'Unknown')
            if section not in tables_by_section:
                tables_by_section[section] = []
            tables_by_section[section].append(table_info)
    else:
        print(f"Warning: Could not retrieve tables for page '{page_title}': {tables_result}")
        # A table retrieval error is a partial wiki error.
        partial_wiki_error = True

    # Step 3: Skim the page to decide on relevant sections
    summary_infobox_list = tables_by_section.get('Summary', [])
    
    skim_func = skim_wiki_page_with_table if summary_infobox_list else skim_wiki_page
    skim_args = {
        'page_title': page_title, 'page_summary': page_summary, 'page_sections': page_sections,
        'question': question, 'analysis': analysis, 'query': query, 'args': args
    }
    if summary_infobox_list:
        skim_args['infobox_table_info'] = summary_infobox_list[0]
        
    skim_result = skim_func(**skim_args)
    
    if not skim_result:
        return {
            "status": "SKIMMING_FAILED", "page_title": page_title,
            "reason": "The skimming process failed to produce a valid decision after multiple retries.",
            "retrieval_trace": retrieval_trace_str, "partial_wiki_error": partial_wiki_error,
        }

    # Step 4: Process skimming decision
    decision = skim_result.get('decision')
    rationale = skim_result.get('rationale')
    print(f"  -> LLM Decision in summary skim: {decision}")
    print(f"  -> LLM Rationale in summary skim: {rationale}")

    if decision == "IRRELEVANT":
        print(f"  -> suggested new query: {skim_result.get('new_query')}")
        return {
            "status": "IRRELEVANT_PAGE", "page_title": page_title, "rationale": rationale,
            "new_query_suggestion": skim_result.get('new_query'),
            "retrieval_trace": retrieval_trace_str, "partial_wiki_error": partial_wiki_error,
        }

    if decision == "EXTRACT_AND_EXPLORE":
        summary_info = skim_result.get('extracted_info')
        relevant_sections = skim_result.get('relevant_sections', [])
        print(f"  -> Extracted Info in summary skim: {summary_info}")
        print(f"  -> Selected Sections for holistic analysis: {relevant_sections}")
        
        sections_info = []
        if relevant_sections:
            # Step 5: Call the holistic extraction function
            sections_extraction_status, sections_extraction_result, sections_had_errors = extract_info_from_sections_holistically(
                page, page_title, relevant_sections, 
                skim_rationale=rationale,
                question=question, analysis=analysis, query=query, args=args,
                tables_by_section=tables_by_section
            )
            # Combine the error status from table fetching and section extraction.
            partial_wiki_error = partial_wiki_error or sections_had_errors

            if sections_extraction_status == "WIKI_API_ERROR":
                # This means ALL sections failed, which is a critical error for this step.
                reason = f"Failed to extract text from all targeted sections for the page '{page_title}' due to network errors."
                print(reason)
                return {"status": "WIKI_API_ERROR", "reason": reason, "retrieval_trace": retrieval_trace_str, "partial_wiki_error": True}
            
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
        "partial_wiki_error": partial_wiki_error,
    }