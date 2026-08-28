from utils import indent_text


def format_retrieval_trace(trace: list, final_status: str, final_result, max_options=50, max_results=50) -> str:
    """
    将维基百科检索轨迹格式化为结构化的、精简的字符串，优化上下文长度，供大模型后续进行反思。

    Args:
        trace (list): 从 retrieve_wiki_page 函数返回的交互轨迹列表。
        final_status (str): 检索过程的最终状态 (例如, "SUCCESS", "FAILED")。
        final_result: 最终结果（成功时是 page 对象，失败时是原因字符串）。
        max_options (int): 要显示的最大歧义选项数。
        max_results (int): 要显示的最大相似页面结果数。

    Returns:
        str: 一个格式化后的精简字符串，详细描述了整个检索过程。
    """
    if not trace:
        return "Trace: No actions taken."

    lines = ["=== Wikipedia Retrieval Trace ==="]
    
    for log in trace:
        # 核心信息：搜索什么，得到了什么状态
        attempt_line = f"Attempt #{log['attempt']}: Search=\"{log['entity_searched']}\" -> Status={log['api_status']}"
        lines.append(attempt_line)

        # 根据状态添加关键的决策信息
        if log['api_status'] == "DISAMBIGUATION":
            options = log.get('options', [])
            options_display = options[:max_options]
            # 显示部分选项和剩余数量
            options_str = f"  - Options: {options_display}"
            if len(options) > max_options:
                options_str += f" ... ({len(options) - max_options} more)"
            lines.append(options_str)
            lines.append(f"  - LLM Choice: \"{log.get('llm_selected_entity', 'N/A')}\"")
        
        elif log['api_status'] == "NOT_FOUND":
            suggestion = log.get('suggestion', 'N/A')
            lines.append(f"  - Suggestion: \"{str(suggestion)}\"")
            
            results = log.get('search_results', [])
            results_display = results[:max_results]
            results_str = f"  - Similar: {results_display}"
            if len(results) > max_results:
                results_str += f" ... ({len(results) - max_results} more)"
            lines.append(results_str)
            lines.append(f"  - LLM Choice: \"{log.get('llm_selected_entity', 'N/A')}\"")

        elif log['api_status'] == "SUCCESS":
             lines.append(f"  - Page Found: \"{log.get('page_title', 'N/A')}\"")

        elif log['api_status'] == "ERROR":
            # 错误信息通常在 final_result 中，这里只标记事件
            lines.append(f"  - Error occurred during search.")

    # 最终结果
    lines.append("--- Final Outcome ---")
    lines.append(f"Status: {final_status}")
    if final_status == "SUCCESS":
        lines.append(f"Result: Page '{getattr(final_result, 'title', 'N/A')}' found.")
    else:
        lines.append(f"Result: {final_result}")
    
    return "\n".join(lines)


def format_wikipedia_retrieval(query: str, entity: str, result: dict, args=None, verbose: bool = True) -> str:
    """
    Formats the structured result from process_wiki_retrieval_for_query into a 
    human-readable string suitable for an LLM's context.

    Args:
        query (str): The sub-query that was executed.
        entity (str): The initial entity that was searched for.
        result (dict): The output dictionary from process_wiki_retrieval_for_query.

    Returns:
        str: A formatted string summarizing the entire retrieval and extraction process.
    """
    use_indent = args.use_indent if args else True
    lines = [
        f"Wikipedia Retrieval Results for entity: \"{entity}\"",
    ]

    status = result.get('status', 'UNKNOWN')
    retrieval_trace = result.get('retrieval_trace', 'No trace available.')
    
    if verbose:
        if status == "SUCCESS" and 'Attempt #2:' not in retrieval_trace:
            lines.append("**Information Extraction:**")
        else:
            lines.append("**1. Page Finding Trace:**")
            lines.append(retrieval_trace)
            lines.append("\n**2. Information Extraction:**")

        if status == "SUCCESS":
            lines.append(f"- **Page Title:** {result.get('page_title', 'N/A')}")
            lines.append(f"- **All Page Sections:** {result.get('all_page_sections', [])}")
            
            lines.append("\n- **A. Skimming Summary and Section Selection:**")
            skim_rationale = indent_text(result.get('skim_rationale', 'No rationale provided.'), "    ", add_space=True, use_indent=use_indent)
            lines.append(f"  - **Rationale:**{skim_rationale}")
            lines.append(f"  - **Selected Sections for Deeper Analysis:** {result.get('selected_sections', [])}")
            
            lines.append("\n- **B. Extracted Content:**")
            summary_info_text = result.get('summary_info') or "None"
            summary_info = indent_text(summary_info_text, "    ", add_space=True, use_indent=use_indent)
            lines.append(f"  - **From Summary:**{summary_info}")

            sections_info = result.get('sections_info', [])
            if sections_info:
                lines.append("  - **From Sections:**")
                for section_data in sections_info:
                    section_title = section_data.get('section', 'N/A')
                    # Main rationale from collaborative analysis or text-only extraction
                    rationale_text = section_data.get('rationale', 'No rationale provided.')
                    rationale = indent_text(rationale_text, "        ", add_space=True, use_indent=use_indent)
                    # Information from text
                    text_info_text = section_data.get('extracted_info') or "None"
                    text_info = indent_text(text_info_text, "        ", add_space=True, use_indent=use_indent)
                    
                    lines.append(f"    - **Section: \"{section_title}\"**")
                    lines.append(f"      - **Rationale for section processing:**{rationale}")
                    lines.append(f"      - **Extracted from Text:**{text_info}")

                    # Information from tables within the section
                    tables_info = section_data.get('extracted_tables_info', [])
                    if tables_info:
                        lines.append("      - **Extracted from Tables:**")
                        for table_data in tables_info:
                            table_name = table_data.get('table_name', 'N/A')
                            table_info_text = table_data.get('extracted_table_info') or "None"
                            table_info = indent_text(table_info_text, "            ", add_space=True, use_indent=use_indent)
                            table_rationale_text = table_data.get('table_extraction_rationale', 'No rationale provided.')
                            table_rationale = indent_text(table_rationale_text, "            ", add_space=True, use_indent=use_indent)
                            lines.append(f"        - **Table: \"{table_name}\"**")
                            lines.append(f"          - **Extracted:**{table_info}")
                            lines.append(f"          - **Rationale:**{table_rationale}")

            else:
                lines.append("  - **From Sections:** No information extracted from sections (either none were selected or they were empty).")
            
        elif status == "IRRELEVANT_PAGE":
            lines.append(f"- **Outcome:** Page '{result.get('page_title', 'N/A')}' was found but deemed IRRELEVANT.")
            rationale = indent_text(result.get('rationale', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Rationale:**{rationale}")
            if result.get('new_query_suggestion'):
                lines.append(f"- **Suggested New Query:** \"{result['new_query_suggestion']}\"")
                
        elif status == "RETRIEVAL_FAILED":
            lines.append(f"- **Outcome:** FAILED to retrieve a Wikipedia page.")
            reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Reason:**{reason}")

        elif status == "SKIMMING_FAILED":
            lines.append(f"- **Outcome:** FAILED to skim Wikipedia page '{result.get('page_title', 'N/A')}'.")
            reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Reason:**{reason}")

        else:
            lines.append(f"- **Outcome:** An unexpected error occurred.")
            lines.append(f"- **Status:** {status}")
            if result.get('reason'):
                reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
                lines.append(f"- **Reason:**{reason}")
    else:
        if status == "SUCCESS" and 'Attempt #2:' not in retrieval_trace:
            lines.append("**Information Extraction:**")
        else:
            lines.append("**1. Page Finding Trace:**")
            lines.append(retrieval_trace)
            lines.append("\n**2. Information Extraction:**")

        if status == "SUCCESS":
            lines.append(f"- **Page Title:** {result.get('page_title', 'N/A')}")
            
            all_sections = result.get('all_page_sections', [])
            selected_sections = result.get('selected_sections', [])
            
            if not all_sections:
                lines.append("- **Page Sections:** None")
            else:
                max_display = 12
                if len(all_sections) <= max_display:
                    lines.append(f"- **Page Sections:** {all_sections}")
                else:
                    display_sections = set(selected_sections)
                    for sec in all_sections:
                        if len(display_sections) >= max_display:
                            break
                        display_sections.add(sec)
                    final_display = [sec for sec in all_sections if sec in display_sections]
                    lines.append(f"- **Page Sections (Truncated):** {final_display} ... (and {len(all_sections) - len(final_display)} more)")
            
            lines.append(f"- **Selected Sections:** {selected_sections}")
            lines.append("\n- **Extracted Content:**")
            
            summary_info_text = result.get('summary_info') or "None"
            summary_info = indent_text(summary_info_text, "    ", add_space=True, use_indent=use_indent)
            lines.append(f"  - **From Summary:**{summary_info}")

            sections_info = result.get('sections_info', [])
            if sections_info:
                for section_data in sections_info:
                    section_title = section_data.get('section', 'N/A')
                    lines.append(f"  - **From Section \"{section_title}\":**")
                    
                    text_info_text = section_data.get('extracted_info') or "None"
                    text_info = indent_text(text_info_text, "      ", add_space=True, use_indent=use_indent)
                    lines.append(f"    - **Text:**{text_info}")

                    tables_info = section_data.get('extracted_tables_info', [])
                    if tables_info:
                        for table_data in tables_info:
                            table_name = table_data.get('table_name', 'N/A')
                            table_info_text = table_data.get('extracted_table_info') or "None"
                            table_info = indent_text(table_info_text, "      ", add_space=True, use_indent=use_indent)
                            lines.append(f"    - **Table \"{table_name}\":**{table_info}")
            else:
                lines.append("  - **From Sections:** No information extracted from sections (either none were selected or they were empty).")
                
        elif status == "IRRELEVANT_PAGE":
            lines.append(f"- **Outcome:** Page '{result.get('page_title', 'N/A')}' was found but deemed IRRELEVANT.")
            rationale = indent_text(result.get('rationale', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Rationale:**{rationale}")
            if result.get('new_query_suggestion'):
                lines.append(f"- **Suggested New Query:** \"{result['new_query_suggestion']}\"")
                
        elif status == "RETRIEVAL_FAILED":
            lines.append(f"- **Outcome:** FAILED to retrieve a Wikipedia page.")
            reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Reason:**{reason}")

        elif status == "SKIMMING_FAILED":
            lines.append(f"- **Outcome:** FAILED to skim Wikipedia page '{result.get('page_title', 'N/A')}'.")
            reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
            lines.append(f"- **Reason:**{reason}")

        else:
            lines.append(f"- **Outcome:** An unexpected error occurred.")
            lines.append(f"- **Status:** {status}")
            if result.get('reason'):
                reason = indent_text(result.get('reason', 'N/A'), "  ", add_space=True, use_indent=use_indent)
                lines.append(f"- **Reason:**{reason}")

    return "\n".join(lines) 


def format_wikipedia_retrieval_concise(query: str, entity: str, result: dict, args=None) -> str:
    """
    Formats the structured result from process_wiki_retrieval_for_query into a
    concise, human-readable string suitable for an LLM's context, with minimal markdown.
    
    Args:
        query (str): The sub-query that was executed.
        entity (str): The initial entity that was searched for.
        result (dict): The output dictionary from process_wiki_retrieval_for_query.

    Returns:
        str: A formatted string summarizing the entire retrieval and extraction process.
    """
    use_indent = args.use_indent if args else True
    lines = [
        f"Wikipedia Retrieval Results for entity: \"{entity}\"",
    ]

    status = result.get('status', 'UNKNOWN')
    retrieval_trace = result.get('retrieval_trace', 'No trace available.')

    if status == "SUCCESS" and 'Attempt #2:' not in retrieval_trace:
        lines.append("Information Extraction:")
    else:
        lines.append("1. Page Finding Trace:")
        lines.append(retrieval_trace)
        lines.append("\n2. Information Extraction:")

    if status == "SUCCESS":
        lines.append(f"Page Title: {result.get('page_title', 'N/A')}")
        lines.append(f"All Page Sections: {result.get('all_page_sections', [])}\n")
        
        lines.append("A. Skimming Summary and Section Selection:")
        skim_rationale = indent_text(result.get('skim_rationale', 'No rationale provided.'), "", add_space=True, use_indent=use_indent)
        lines.append(f"Rationale:{skim_rationale}")
        lines.append(f"Selected Sections for Deeper Analysis: {result.get('selected_sections', [])}\n")
        
        lines.append("B. Extracted Content:")
        summary_info = indent_text(result.get('summary_info') or "None", "", add_space=True, use_indent=use_indent)
        lines.append(f"From Summary:{summary_info}")

        sections_info = result.get('sections_info', [])
        if sections_info:
            lines.append("From Sections:")
            for section_data in sections_info:
                section_title = section_data.get('section', 'N/A')
                text_info = indent_text(section_data.get('extracted_info') or "None", "    ", add_space=True, use_indent=use_indent)
                rationale = indent_text(section_data.get('rationale', 'No rationale provided.'), "    ", add_space=True, use_indent=use_indent)
                lines.append(f"  Section \"{section_title}\":")
                lines.append(f"    Extracted from Text:{text_info}")
                lines.append(f"    Rationale for processing:{rationale}")

                tables_info = section_data.get('extracted_tables_info', [])
                if tables_info:
                    lines.append("    Extracted from Tables:")
                    for table_data in tables_info:
                        table_name = table_data.get('table_name', 'N/A')
                        table_info = indent_text(table_data.get('extracted_table_info') or "None", "        ", add_space=True, use_indent=use_indent)
                        table_rationale = indent_text(table_data.get('table_extraction_rationale', 'No rationale provided.'), "        ", add_space=True, use_indent=use_indent)
                        lines.append(f"      Table \"{table_name}\":")
                        lines.append(f"        Extracted:{table_info}")
                        lines.append(f"        Rationale:{table_rationale}")
        else:
            lines.append("From Sections: No information was extracted from page sections (either none were selected or they were empty).")
    
    elif status == "IRRELEVANT_PAGE":
        lines.append(f"Outcome: Page '{result.get('page_title', 'N/A')}' was found but deemed IRRELEVANT.")
        rationale = indent_text(result.get('rationale', 'N/A'), "", add_space=True, use_indent=use_indent)
        lines.append(f"Rationale:{rationale}")
        if result.get('new_query_suggestion'):
            lines.append(f"Suggested New Query: \"{result['new_query_suggestion']}\"")
            
    elif status == "RETRIEVAL_FAILED":
        lines.append(f"Outcome: FAILED to retrieve a Wikipedia page.")
        reason = indent_text(result.get('reason', 'N/A'), "", add_space=True, use_indent=use_indent)
        lines.append(f"Reason:{reason}")

    elif status == "SKIMMING_FAILED":
        lines.append(f"Outcome: FAILED to skim Wikipedia page '{result.get('page_title', 'N/A')}'.")
        reason = indent_text(result.get('reason', 'N/A'), "", add_space=True, use_indent=use_indent)
        lines.append(f"Reason:{reason}")

    else: # Handles UNKNOWN_ERROR and other potential statuses
        lines.append(f"Outcome: An unexpected error occurred.")
        lines.append(f"Status: {status}")
        if result.get('reason'):
            reason = indent_text(result.get('reason', 'N/A'), "", add_space=True, use_indent=use_indent)
            lines.append(f"Reason:{reason}")

    return "\n".join(lines)
