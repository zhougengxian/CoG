import re
import ast
from functools import partial

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from wiki.api import get_text_under_section, get_full_page_details, retrieve_wiki_page
from wiki.workflow_text import skim_wiki_page, parse_section_extraction_result, parse_present_info_result
from wiki.prompt import prompt_extract_from_chunks, prompt_present_info_with_table, prompt_analyze_section_with_tables, prompt_extract_from_table
from wiki.table_api import get_wikipedia_tables
from wiki.format import format_retrieval_trace
from utils import generate_process


def skim_wiki_page_with_table(page_title, page_summary, page_sections, infobox_table_info, question, analysis, query, args, max_retries=3, skim_threshold=5000):
    """
    Skims a Wikipedia page that includes a summary infobox to decide on the next action.
    This version provides the FULL infobox data, intelligently serialized to fit context.
    """
    # 1. Intelligently serialize the full infobox table
    infobox_df = infobox_table_info.get('dataframe')
    infobox_data_str = "No infobox data available."

    if infobox_df is not None:
        # Tier 1: Try full Markdown
        if infobox_table_info.get('markdown_total_len', float('inf')) <= skim_threshold:
            serialized_table = infobox_df.to_markdown(index=False)
            format_used = "Markdown"
        # Tier 2: Fallback to full CSV
        elif infobox_table_info.get('csv_total_len', float('inf')) <= skim_threshold:
            serialized_table = infobox_df.to_csv(index=False)
            format_used = "CSV"
        # Tier 3: Fallback to truncated CSV
        else:
            serialized_table = infobox_df.to_csv(index=False)[:skim_threshold]
            serialized_table += "\n... [Table content truncated due to length] ..."
            format_used = "Truncated CSV"

        infobox_data_str = f"Format: {format_used}\n{serialized_table}"

    # 2. Prepare generate_process inputs
    template_inputs = {
        'question': question,
        'analysis': analysis,
        'query': query,
        'page_title': page_title,
        'page_summary': page_summary,
        'page_sections': page_sections,
        'infobox_table': infobox_data_str
    }

    # 3. Call generate_process with the new prompt
    parsed_result = generate_process(
        step_name="Skim Wiki Page with Table",
        prompt_template=prompt_present_info_with_table,
        template_inputs=template_inputs,
        parsing_function=parse_present_info_result,
        args=args,
        module='wiki',
        max_retries=max_retries
    )

    if not parsed_result:
        print("\n--- [WORKFLOW FAILED] at step 'Skim Wiki Page with Table'.---")
        return None
    return parsed_result


def parse_table_extraction_result(result_text):
    """
    Parses the output from the prompt_extract_from_table LLM call.
    """
    try:
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Table Info:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

        info_match = re.search(r"Extracted Table Info:(.*)", result_text, re.DOTALL)
        extracted_info_raw = info_match.group(1).strip() if info_match else ""
        extracted_info = None if extracted_info_raw.lower() == 'none' else extracted_info_raw

        return {
            "rationale": rationale,
            "extracted_table_info": extracted_info
        }
    except Exception as e:
        print(f"Error parsing table extraction result: {e}\nInput text: {result_text}")
        return None


def parse_collaborative_analysis_result(result_text, available_table_names):
    """
    Parses the output from the prompt_analyze_section_with_tables LLM call.
    It extracts rationale, info from text, and a list of selected table names,
    and validates the selected tables against the available ones.
    """
    try:
        # Extract Rationale
        rationale_match = re.search(r"Rationale:(.*?)(?=\nExtracted Info:|$)", result_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

        # Extract Info from text chunks
        info_match = re.search(r"Extracted Info:(.*?)(?=\nSelected Tables:|$)", result_text, re.DOTALL)
        extracted_info_raw = info_match.group(1).strip() if info_match else ""
        extracted_info = None if extracted_info_raw.lower() == 'none' else extracted_info_raw

        # Extract and validate selected tables list
        tables_match = re.search(r"Selected Tables:(.*)", result_text, re.DOTALL)
        tables_str = tables_match.group(1).strip() if tables_match else "[]"
        selected_tables_raw = ast.literal_eval(tables_str)
        
        available_table_names_set = set(available_table_names)
        for table_name in selected_tables_raw:
            if table_name not in available_table_names_set:
                print(f"Error: Model hallucinated a non-existent table name: '{table_name}'. Failing parsing.")
                return None

        return {
            "rationale": rationale,
            "extracted_info": extracted_info,
            "selected_tables": selected_tables_raw
        }
    except Exception as e:
        print(f"Error parsing collaborative analysis result: {e}\nInput text: {result_text}")
        return None


def _serialize_full_table(table_info, table_threshold=10000):
    """
    Serializes a full table's data using a three-tier fallback strategy.
    """
    df = table_info.get('dataframe')
    if df is None or df.empty:
        return "No data available for this table."

    # Tier 1: Try full Markdown
    if table_info.get('markdown_total_len', float('inf')) <= table_threshold:
        serialized_table = df.to_markdown(index=False)
        format_used = "Markdown"
    # Tier 2: Fallback to full CSV
    elif table_info.get('csv_total_len', float('inf')) <= table_threshold:
        serialized_table = df.to_csv(index=False)
        format_used = "CSV"
    # Tier 3: Fallback to truncated CSV
    else:
        serialized_table = df.to_csv(index=False)[:table_threshold]
        serialized_table += "\n... [Table content truncated due to length] ..."
        format_used = "Truncated CSV"

    return f"Format: {format_used}\n{serialized_table}"


def _create_tables_preview(tables_in_section, preview_threshold=5000):
    """
    Creates a preview string for a list of tables using a three-tier fallback strategy,
    utilizing pre-calculated lengths to be efficient.
    """
    # Tier 1: Try Markdown previews if total length is within threshold
    markdown_total_len = sum(table.get('markdown_head_len', float('inf')) for table in tables_in_section)
    if markdown_total_len <= preview_threshold:
        table_previews = []
        for table in tables_in_section:
            df = table.get('dataframe')
            if df is not None and not df.empty:
                preview = df.head(5).to_markdown(index=False)
                table_previews.append(f"Table Name: {table['table_name']}\n{preview}")
        return "Preview Format: Markdown (Top 5 rows)\n\n" + "\n---\n".join(table_previews)

    # Tier 2: Fallback to CSV previews if total length is within threshold
    csv_total_len = sum(table.get('csv_head_len', float('inf')) for table in tables_in_section)
    if csv_total_len <= preview_threshold:
        table_previews = []
        for table in tables_in_section:
            df = table.get('dataframe')
            if df is not None and not df.empty:
                preview = df.head(5).to_csv(index=False)
                table_previews.append(f"Table Name: {table['table_name']}\n{preview}")
        return "Preview Format: CSV (Top 5 rows)\n\n" + "\n---\n".join(table_previews)

    # Tier 3: Fallback to metadata (name + columns)
    previews = []
    for table in tables_in_section:
        df = table.get('dataframe')
        if df is not None:
            columns = df.columns.tolist()
            preview = f"Table Name: {table['table_name']}\nColumns: {columns}"
            previews.append(preview)

    return "Preview Format: Metadata-only (Table Name and Columns)\n\n" + "\n---\n".join(previews)


def extract_info_from_sections_with_tables(
    page, page_title, relevant_sections, skim_rationale, summary_info,
    question, analysis, query, embeddings, args,
    tables_by_section, k=3
):
    """
    Orchestrates information extraction from sections, handling both text (via RAG)
    and tables (via preview and selection).
    """
    all_sections_results = []
    any_section_failed = False
    num_failures = 0

    if not relevant_sections:
        print("No sections were marked for exploration.")
        return "SUCCESS", all_sections_results, False

    for section_title in relevant_sections:
        print(f"\n--- Processing Section: '{section_title}' ---")

        # 1. Get section text with error handling
        status, section_text = get_text_under_section(page, section_title)
        if not status:
            # A network error occurred for this section. Log it, mark the failure, and continue.
            any_section_failed = True
            num_failures += 1
            error_message = section_text
            print(f"  -> WIKI_API_ERROR for section '{section_title}': {error_message}")
            all_sections_results.append({
                "section": section_title,
                "status": "WIKI_API_ERROR",
                "rationale": "Failed to retrieve section text due to a network error.",
                "extracted_info": error_message,
                "selected_tables": [],
                "extracted_tables_info": []
            })
            continue

        if not section_text or not section_text.strip():
            print(f"  -> Section '{section_title}' is empty or could not be found. Skipping.")
            all_sections_results.append({
                "section": section_title,
                "status": "EMPTY_SECTION",
                "rationale": "",
                "extracted_info": f"Section '{section_title}' is empty or could not be found.",
                "selected_tables": [],
                "extracted_tables_info": []
            })
            continue

        # 2. Text processing via RAG (common for both paths)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(section_text)
        print(f"  -> Split section into {len(chunks)} chunks.")

        if not chunks:
            print(f"  -> No chunks were created from section '{section_title}'. Skipping.")
            continue

        vector_store = InMemoryVectorStore.from_texts(texts=chunks, embedding=embeddings)
        num_chunks_to_retrieve = min(k, len(chunks))
        docs = vector_store.similarity_search(query, k=num_chunks_to_retrieve)
        context_chunks = "\n\n---\n\n".join([doc.page_content for doc in docs])
        print(f"  -> Retrieved top {len(docs)} relevant chunks for the query.")

        # 3. Check for tables and choose the execution path
        tables_in_section = tables_by_section.get(section_title, [])
        
        # PATH A: Section has tables -> Collaborative Analysis
        if tables_in_section:
            print(f"  -> Found {len(tables_in_section)} tables in this section. Starting collaborative analysis.")
            
            # 3a. Create table previews
            tables_preview_str = _create_tables_preview(tables_in_section)
            
            # 3b. Create a parsing function with the current section's context for table validation
            available_table_names = [t['table_name'] for t in tables_in_section]
            # Use partial to 'bake in' the list of available table names into the parsing function for validation.
            parsing_func = partial(parse_collaborative_analysis_result, available_table_names=available_table_names)

            # 3c. Call LLM for collaborative analysis
            template_inputs = {
                'question': question, 'analysis': analysis, 'query': query,
                'page_title': page_title, 'section_title': section_title,
                'exploration_rationale': skim_rationale,
                'k': len(docs), 'context_chunks': context_chunks,
                'tables_preview': tables_preview_str,
                'table_names_in_section': available_table_names
            }

            analysis_result = generate_process(
                step_name=f"Collaborative analysis for Section '{section_title}'",
                prompt_template=prompt_analyze_section_with_tables,
                template_inputs=template_inputs,
                parsing_function=parsing_func,
                args=args, module='wiki', max_retries=3
            )
            
            if analysis_result:
                print(f"  -> LLM Rationale: {analysis_result['rationale']}")
                print(f"  -> Extracted Info: {analysis_result['extracted_info']}")
                print(f"  -> Selected Tables for deep extraction: {analysis_result['selected_tables']}")

                extracted_tables_info = []
                selected_tables_for_extraction = analysis_result.get('selected_tables', [])

                if selected_tables_for_extraction:
                    for table_name in selected_tables_for_extraction:
                        # Find the full table object
                        table_to_extract = next((t for t in tables_in_section if t.get('table_name') == table_name), None)

                        if not table_to_extract:
                            print(f"  -> WARNING: LLM selected table '{table_name}' but it was not found in the section's table list.")
                            continue
                        
                        print(f"    -- Extracting from table: '{table_name}' --")

                        # Serialize full table data
                        full_table_str = _serialize_full_table(table_to_extract)
                        
                        # Prepare and run extraction prompt
                        table_extraction_inputs = {
                            'question': question, 'analysis': analysis, 'query': query,
                            'page_title': page_title,
                            'summary_content': summary_info or "None",
                            'summary_reading_rationale': skim_rationale or "None provided.",
                            'section_title': section_title,
                            'extracted_text_info': analysis_result.get('extracted_info') or "None",
                            'section_reading_rationale': analysis_result.get('rationale') or "None provided.",
                            'table_name': table_name,
                            'full_table_data': full_table_str
                        }

                        table_extraction_result = generate_process(
                            step_name=f"Extract from Table '{table_name}' in Section '{section_title}'",
                            prompt_template=prompt_extract_from_table,
                            template_inputs=table_extraction_inputs,
                            parsing_function=parse_table_extraction_result,
                            args=args, module='wiki', max_retries=3
                        )

                        if table_extraction_result:
                            print(f"      -> Rationale: {table_extraction_result['rationale']}")
                            print(f"      -> Extracted Table Info: {table_extraction_result['extracted_table_info']}")
                            extracted_tables_info.append({
                                "table_name": table_name,
                                "extracted_table_info": table_extraction_result['extracted_table_info'],
                                "table_extraction_rationale": table_extraction_result['rationale']
                            })
                        else:
                             print(f"      -> FAILED to extract info from table '{table_name}'.")
                             extracted_tables_info.append({
                                "table_name": table_name,
                                "extracted_table_info": "Extraction failed for this table.",
                                "table_extraction_rationale": "The LLM failed to produce a valid analysis or extraction for this table after multiple retries."
                            })

                all_sections_results.append({
                    "section": section_title,
                    "status": "SUCCESS",
                    "rationale": analysis_result['rationale'],
                    "extracted_info": analysis_result['extracted_info'],
                    "selected_tables": analysis_result['selected_tables'],
                    "extracted_tables_info": extracted_tables_info
                })
            else:
                print(f"LLM extraction Error! Cannot extract info from section '{section_title}' in collaborative analysis.")
                all_sections_results.append({
                    "section": section_title,
                    "status": "ANALYSIS_FAILED",
                    "rationale": "",
                    "extracted_info": "LLM failed to produce a valid analysis for this section after multiple retries.",
                    "selected_tables": [],
                    "extracted_tables_info": []
                })

        # PATH B: Section has no tables -> Pure Text RAG Extraction
        else:
            print("  -> No tables found. Proceeding with text-only RAG extraction.")
            template_inputs = {
                'question': question, 'analysis': analysis, 'query': query,
                'page_title': page_title, 'section_title': section_title,
                'exploration_rationale': skim_rationale,
                'k': len(docs), 'chunks': context_chunks
            }
        
            extraction_result = generate_process(
                step_name=f"Extract from Section '{section_title}'",
                prompt_template=prompt_extract_from_chunks,
                template_inputs=template_inputs,
                parsing_function=parse_section_extraction_result,
                args=args, module='wiki', max_retries=3
            )

            if extraction_result:
                print(f"  -> LLM Rationale: {extraction_result['rationale']}")
                print(f"  -> Extracted Info: {extraction_result['extracted_info']}")
                all_sections_results.append({
                    "section": section_title,
                    "status": "SUCCESS",
                    "rationale": extraction_result['rationale'],
                    "extracted_info": extraction_result['extracted_info'],
                    "selected_tables": [],
                    "extracted_tables_info": []
                })
            else:
                print(f"LLM extraction Error! Cannot extract info from section '{section_title}' in text-only RAG extraction.")
                all_sections_results.append({
                    "section": section_title,
                    "status": "ANALYSIS_FAILED",
                    "rationale": "",
                    "extracted_info": "LLM failed to produce a valid analysis for this section after multiple retries.",
                    "selected_tables": [],
                    "extracted_tables_info": []
                })

    if len(relevant_sections) > 0 and num_failures == len(relevant_sections):
        return "WIKI_API_ERROR", all_sections_results, True

    return "SUCCESS", all_sections_results, any_section_failed


def retrieve_info_wikipedia_with_tables(query: str, entity: str, question: str, analysis: str, args, embeddings):
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
    
    # --- Fetch all page details at once ---
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
    
    # --- Fetch and process tables ---
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

    # Step 2: Skim the retrieved page to decide on the next action.
    summary_infobox_list = tables_by_section.get('Summary', [])
    
    if summary_infobox_list:
        # Table-aware workflow if an infobox is present
        skim_result = skim_wiki_page_with_table(
            page_title, page_summary, page_sections, summary_infobox_list[0],
            question, analysis, query, args
        )
    else:
        # Fallback to original text-only workflow
        skim_result = skim_wiki_page(
            page_title, page_summary, page_sections, question, analysis, query, args
        )
    
    if not skim_result:
        return {
            "status": "SKIMMING_FAILED",
            "page_title": page_title,
            "reason": "The skimming process failed to produce a valid decision after multiple retries.",
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": partial_wiki_error,
        }

    # Step 3: Process the skimming decision.
    decision = skim_result.get('decision')
    rationale = skim_result.get('rationale')
    print(f"  -> LLM Decision in summary skim: {decision}")
    print(f"  -> LLM Rationale in summary skim: {rationale}")

    if decision == "IRRELEVANT":
        print(f"  -> suggested new query: {skim_result.get('new_query')}")
        return {
            "status": "IRRELEVANT_PAGE",
            "page_title": page_title,
            "rationale": rationale,
            "new_query_suggestion": skim_result.get('new_query'),
            "retrieval_trace": retrieval_trace_str,
            "partial_wiki_error": partial_wiki_error,
        }

    if decision == "EXTRACT_AND_EXPLORE":
        summary_info = skim_result.get('extracted_info')
        relevant_sections = skim_result.get('relevant_sections', [])
        print(f"  -> Extracted Info in summary skim: {summary_info}")
        print(f"  -> Selected Sections in summary skim: {relevant_sections}")
        
        sections_info = []
        if relevant_sections:
            # Step 4: If sections are deemed relevant, extract info from them using RAG.
            sections_extraction_status, sections_extraction_result, sections_had_errors = extract_info_from_sections_with_tables(
                page, page_title, relevant_sections, 
                skim_rationale=rationale, summary_info=summary_info,
                question=question, analysis=analysis, query=query, 
                embeddings=embeddings, args=args,
                tables_by_section=tables_by_section,
                k=args.section_chunks
            )
            # Combine the error status from table fetching and section extraction.
            partial_wiki_error = partial_wiki_error or sections_had_errors

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
        "partial_wiki_error": partial_wiki_error,
    }