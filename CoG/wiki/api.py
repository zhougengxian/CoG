import wikipediaapi
import wikipedia
import re 
import time
import functools

from wiki.prompt import (
    prompt_disambiguate_with_reasoning,
    prompt_refine_search_with_reasoning,
    prompt_disambiguate_history,
    prompt_refine_search_history
)
from utils import generate_process

wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia.set_user_agent("CoG/1.0 (user@example.com)")

def query_wikipedia_api(query: str, lang: str = "en", retries: int = 3, backoff_factor: float = 1):
    """
    一个基于 `wikipedia-api` 库重构的、为程序或大模型设计的维基百科查询API函数。
    该函数增加了重试和指数回退机制。

    该函数接收一个查询词，返回一个包含状态和结果的元组。

    :param query: str, 需要查询的词条。
    :param lang: str, 维基百科的语言版本，默认为 "en"。
    :param retries: int, 失败时的重试次数，默认为 3。
    :param backoff_factor: float, 指数回退的基准秒数，默认为 0.3。
    :return: tuple, 包含查询结果的元组。
        - ("SUCCESS", page): 成功找到页面，返回页面对象。
        - ("DISAMBIGUATION", options): 页面为消歧义页，返回选项列表。
        - ("NOT_FOUND", suggestion, search_results): 未找到页面，返回建议和搜索结果。
        - ("ERROR", query, error_message): 发生错误。
    """
    # print(f"使用 'wikipedia-api' 查询: '{query}', 语言: '{lang}'")

    last_exception = None
    for attempt in range(retries):
        try:
            # 1. 初始化 wikipedia-api 对象，设置 User-Agent 是一个好习惯
            wiki_api = wikipediaapi.Wikipedia(
                user_agent="CoG-Bot/1.0 (Academic Research; +https://github.com/anonymous/CoG)",
                language=lang,
                # extract_format=wikipediaapi.ExtractFormat.WIKI  # 获取纯文本内容
            )

            page = wiki_api.page(query)

            if not page.exists():
                # 2. 页面未找到
                search_results = []
                suggestion = None
                search_results, suggestion = wikipedia.search(query, suggestion=True)

                return ("NOT_FOUND", suggestion, search_results)

            # 3. 检查页面是否为消歧义页面 (通过检查分类)
            # 这是一个尽力而为的检查，因为分类名称可能因语言而异
            disambiguation_markers = [
                'Category:Disambiguation pages',
                'Category:All article disambiguation pages'
            ]
            # 判断是否为消歧义页
            is_disambiguation = any(marker in page.categories for marker in disambiguation_markers)

            if is_disambiguation:
                options = [link for link in page.links.keys() if not link.startswith('Category:') and not link.startswith('Talk:')
                            and not link.startswith('File:') and not link.startswith('Template:') and not link.startswith('Help:')]
                return ("DISAMBIGUATION", options)
            page_title = page.title # cache the title to avoid multiple network requests from the lazy-loaded property.
            # print(f"Success: Found page '{page_title}'.")
            # 4. 成功找到页面
            return ("SUCCESS", page)

        except KeyError as e:
            # KeyError Indicates unexpected API response format (e.g. missing 'pages')
            # This often happens with special namespaces like 'Commons:'
            # Retry is useless for this error
            print(f"Error querying '{query}': KeyError (Non-network error): {e}")
            return ("ERROR", query, f"API Response Error: {e} (Likely invalid query format/namespace for this API)")

        except Exception as e:
            last_exception = e
            print(f"查询 '{query}' 时发生错误 (尝试 {attempt + 1}/{retries}): \n{e}")
            if attempt < retries - 1:
                # 指数回退
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"将在 {sleep_time:.2f} 秒后重试...")
                time.sleep(sleep_time)
    
    return ("ERROR", query, str(last_exception))
    

def list_top_level_sections(page):
    """
    从一个 WikipediaPage 对象中提取所有一级章节 (level 0) 的标题。

    Args:
        page (wikipediaapi.WikipediaPage): 来自 wikipedia-api 库的页面对象。

    Returns:
        list: 一个包含所有一级章节标题的字符串列表。
              如果页面对象无效或没有章节，则返回空列表。
    """
    # 确保 page 对象有效且包含 sections 属性
    if not hasattr(page, 'sections'):
        return []
        
    top_level_sections = []
    for section in page.sections:
        if section.level == 1:
            top_level_sections.append(section.title)
            
    return top_level_sections


def get_text_under_section(page, section_title, retries=3, backoff_factor=0.5):
    """
    获取维基百科页面中指定一级章节下的所有文本（包括所有子章节）。
    增加了重试和错误处理逻辑以应对网络问题。

    Args:
        page (wikipediaapi.WikipediaPage): wikipedia-api 的页面对象。
        section_title (str): 需要获取文本的一级章节的标题。
        retries (int): The number of times to retry on failure.
        backoff_factor (float): Factor to determine sleep time between retries.

    Returns:
        tuple: (status, data)
               - 成功: (True, str), data 是包含该章节及其所有子章节的完整文本。
               - 失败: (False, str), data 是错误信息。
               - 未找到: (True, None), 如果找不到该章节。
    """
    
    # 辅助函数：递归地从一个章节及其所有子章节中提取文本
    # This function will be called inside the try block, so any network error during recursion will be caught.
    def get_all_text(section):
        """Recursively get text from a section and its subsections."""
        # 将当前章节的标题和文本添加到结果中
        equals = "=" * (section.level)
        # 将 section.text 中连续2个或以上的换行符替换为1个
        cleaned_text = re.sub(r'\n{2,}', '\n', section.text)
        full_text = f"{equals} {section.title} {equals}\n{cleaned_text}"
        if len(cleaned_text) > 0:
            full_text += "\n\n"
        # 递归地处理所有子章节
        for sub_section in section.sections:
            full_text += get_all_text(sub_section)
        return full_text

    last_exception = None
    for attempt in range(retries):
        try:
            # 1. 在页面的一级章节中寻找标题匹配的章节对象 (page.sections triggers network call)
            target_section = None
            for section in page.sections:
                if section.title == section_title:
                    target_section = section
                    break
            
            # 2. 如果找到了目标章节，就使用辅助函数提取所有文本
            if target_section:
                # get_all_text triggers network calls for section.text and sub_section.sections
                full_section_text = get_all_text(target_section)
                # The original code had [:-2] to remove a trailing newline. Let's be safer.
                if full_section_text.endswith("\n\n"):
                    full_section_text = full_section_text[:-2]
                return True, full_section_text
            else:
                # 3. 如果未找到，这不是一个错误，而是正常情况
                return True, None

        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{retries} failed to get section '{section_title}': {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

    # All retries failed
    error_message = f"Failed to fetch section '{section_title}' after {retries} attempts. Last error:\n{last_exception}. It may be temporary network issues. Please try again."
    return False, error_message

def get_full_page_details(page, retries=3, backoff_factor=0.5):
    """
    Fetches title, summary, and top-level sections from a WikipediaPage object
    in a single call with robust error handling and retry logic for network requests.

    Args:
        page (wikipediaapi.WikipediaPage): The page object from which to fetch details.
        retries (int): The number of times to retry on failure.
        backoff_factor (float): Factor to determine sleep time between retries.

    Returns:
        tuple: A tuple containing (status, data).
               - If successful, `status` is True and `data` is a dictionary:
                 {'title': str, 'summary': str, 'sections': list[str]}.
               - If unsuccessful, `status` is False and `data` is a string
                 containing the error message.
    """
    last_exception = None
    for attempt in range(retries):
        try:
            # These attributes are lazy-loaded and will trigger network requests on first access.
            # We fetch them all here to consolidate network activity and error handling.
            title = page.title
            summary = page.summary
            url = page.fullurl
            # Re-use the safe function to get sections
            sections = list_top_level_sections(page)

            return True, {
                "title": title,
                "summary": summary,
                "sections": sections,
                "url": url
            }
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{retries} failed to fetch page details for '{getattr(page, 'title', 'Unknown Page')}': {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

    # All retries failed
    error_message = f"Failed to fetch page details after {retries} attempts. Last error:\n{last_exception}. It may be temporary network issues. Please try again."
    return False, error_message


def _format_search_history(history: list, max_options: int = 5) -> str:
    """Formats the search history into a readable string for the LLM prompt."""
    if not history:
        return "No search attempts have been made yet."

    history_str = ""
    for entry in history:
        history_str += f"- **Attempt {entry['attempt']}:**\n"
        history_str += f"  - Searched Query: \"{entry['searched_query']}\"\n"
        history_str += f"  - Result: {entry['result_type']}\n"
        
        if entry['result_type'] == 'Disambiguation':
            options = entry.get('options_provided', [])
            if len(options) > max_options:
                history_str += f"  - Options Provided: {options[:max_options]} ... ({len(options) - max_options} more)\n"
            else:
                history_str += f"  - Options Provided: {options}\n"

        elif entry['result_type'] == 'Not Found':
            history_str += f"  - Wikipedia Suggestion: \"{entry.get('suggestion')}\"\n"
            search_results = entry.get('search_results_provided', [])
            if len(search_results) > max_options:
                history_str += f"  - Search Results Provided: {search_results[:max_options]} ... ({len(search_results) - max_options} more)\n"
            else:
                history_str += f"  - Search Results Provided: {search_results}\n"

        history_str += f"  - Your Rationale: {entry['llm_rationale']}\n"
        history_str += f"  - Your Choice: \"{entry['llm_choice']}\"\n\n"
        
    return history_str.strip()


def parse_disambiguation_selection(llm_output: str, valid_options: list) -> dict:
    """Parses the LLM output for disambiguation selection."""
    try:
        rationale_match = re.search(r"Rationale:(.*?)(?=\nChosen Page:|$)", llm_output, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else ""

        choice_match = re.search(r"Chosen Page:(.*)", llm_output, re.DOTALL)
        choice = choice_match.group(1).strip() if choice_match else ""

        if rationale and choice:
            # First, check for structural correctness, then validate content.
            if choice == "NO_MATCH":
                # "NO_MATCH" is a valid instruction, pass validation.
                pass
            elif choice not in valid_options:
                # Content validation failed: LLM hallucinated an option.
                max_display = 20
                if len(valid_options) > max_display:
                    display_options = f"{valid_options[:max_display]} ... ({len(valid_options) - max_display} more)"
                else:
                    display_options = valid_options
                print(f"Parsing Warning: LLM chose '{choice}', which is not in the provided valid options. This is a hallucination. Valid options were: {display_options}")
                return None
            
            # If validation passes, return the parsed data.
            return {'rationale': rationale, 'choice': choice}
        
        # Structural validation failed: couldn't parse both fields.
        return None
    except Exception:
        return None


def parse_refine_search_selection(llm_output: str) -> dict:
    """Parses the LLM output for search refinement selection."""
    try:
        rationale_match = re.search(r"Rationale:(.*?)(?=\nNew Query:|$)", llm_output, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else ""

        choice_match = re.search(r"New Query:(.*)", llm_output, re.DOTALL)
        choice = choice_match.group(1).strip() if choice_match else ""

        if rationale and choice:
            return {'rationale': rationale, 'choice': choice}
        return None
    except Exception:
        return None


def retrieve_wiki_page(initial_entity: str, question: str, analysis: str, query: str, args, max_interactions: int = 3):
    """
    通过与 LLM 交互来检索维基百科页面，处理歧义和页面未找到的情况。

    该函数尝试为初始实体找到一个有效的维基百科页面。如果初始查询导致消歧义页面或未找到，
    它将使用 LLM 从维基百科 API 提供的建议中选择一个更好的查询。此过程将重复进行，
    直到达到最大交互次数。函数会记录每次尝试的详细轨迹。

    Args:
        initial_entity (str): 要搜索的第一个实体字符串。
        question (str): 原始用户问题，用于提供上下文。
        analysis (str): 用于回答问题的顶层计划。
        query (str): 导致本次实体搜索的具体子查询。
        args: 一个包含模型配置的参数对象 (例如，用于 run_llm)。
        max_interactions (int): 查找页面的最大尝试次数。

    Returns:
        tuple: 一个包含最终状态、结果和交互轨迹的元组。
            - ("SUCCESS", page, trace): 如果成功找到页面。`page` 是一个 WikipediaPage 对象。
            - ("FAILED", reason, trace): 如果过程失败。`reason` 是解释原因的字符串。
            - ("ERROR", reason, trace): failure caused by WikiPedia API.
            `trace` 是一个字典列表，每个字典记录了一次交互尝试的事件。
    """
    current_entity = initial_entity
    trace = []
    search_history = []

    if not current_entity:
        return "FAILED", "No valid entity provided for Wikipedia search.", trace

    for attempt in range(max_interactions):
        interaction_log = {
            "attempt": attempt + 1,
            "entity_searched": current_entity,
        }
        print(f"--- Wikipedia Search for: '{current_entity}' (Attempt #{attempt + 1}/{max_interactions})---")

        # 步骤 1: 查询 Wikipedia API
        status, *result = query_wikipedia_api(current_entity, retries=5)
        interaction_log["api_status"] = status

        # 步骤 2: 处理 API 响应
        if status == "SUCCESS":
            page = result[0]
            # Pre-fetch the title to avoid multiple network requests from the lazy-loaded property.
            page_title = page.title
            outcome = f"Success: Found page '{page_title}'."
            interaction_log["outcome"] = outcome
            interaction_log["page_title"] = page_title
            trace.append(interaction_log)
            print(outcome)
            return "SUCCESS", page, trace

        elif status == "DISAMBIGUATION":
            options = result[0]
            print(f"Disambiguation: Found {len(options)} options: {options}")
            interaction_log["options"] = options
            
            # 根据交互轮次动态选择 Prompt 和输入
            if attempt == 0:
                prompt_template = prompt_disambiguate_with_reasoning
                template_inputs = {
                    'question': question, 'analysis': analysis, 'query': query,
                    'entity': current_entity, 'options_list': options
                }
            else:
                formatted_history = _format_search_history(search_history)
                prompt_template = prompt_disambiguate_history
                template_inputs = {
                    'question': question, 'analysis': analysis, 'query': query,
                    'entity': current_entity, 'options_list': options,
                    'search_history': formatted_history
                }

            # Use functools.partial to pass the 'options' list to the parser for validation
            parser_with_context = functools.partial(parse_disambiguation_selection, valid_options=options)
            
            llm_decision = generate_process(
                step_name=f"Disambiguation for '{current_entity}' (Attempt {attempt + 1})",
                prompt_template=prompt_template,
                template_inputs=template_inputs,
                parsing_function=parser_with_context,
                args=args, module='wiki', max_retries=3
            )

            if not llm_decision:
                reason = f"LLM failed to produce a valid selection for disambiguation after multiple retries."
                interaction_log["outcome"] = f"Failure: {reason}"
                trace.append(interaction_log)
                print(reason)
                return "FAILED", reason, trace
                
            new_entity = llm_decision['choice']
            rationale = llm_decision['rationale']
            interaction_log["llm_rationale"] = rationale
            interaction_log["llm_selected_entity"] = new_entity
            print(f"LLM Rationale: {rationale}")
            print(f"LLM selected new entity: '{new_entity}'")

            history_entry = {
                "attempt": attempt + 1,
                "searched_query": current_entity,
                "result_type": "Disambiguation",
                "options_provided": options,
                "llm_rationale": rationale,
                "llm_choice": new_entity
            }
            search_history.append(history_entry)

            if not new_entity or new_entity.strip() == "NO_MATCH":
                reason = "LLM decided that no options were relevant from the disambiguation page."
                interaction_log["outcome"] = f"Failure: {reason}"
                trace.append(interaction_log)
                print(reason)
                return "FAILED", reason, trace
            
            # 为下一次尝试做准备
            current_entity = new_entity.strip()
            interaction_log["outcome"] = f"Refining search with new entity: '{current_entity}'."

        elif status == "NOT_FOUND":
            suggestion, search_results = result
            print(f"Not Found.\nSuggestion: '{suggestion}'.\nFound {len(search_results)} similar pages: {search_results}")
            interaction_log["suggestion"] = suggestion
            interaction_log["search_results"] = search_results

            if attempt == 0:
                prompt_template = prompt_refine_search_with_reasoning
                template_inputs = {
                    'question': question, 'analysis': analysis, 'query': query,
                    'entity': current_entity, 'suggestion': suggestion, 
                    'search_results_list': search_results
                }
            else:
                formatted_history = _format_search_history(search_history)
                prompt_template = prompt_refine_search_history
                template_inputs = {
                    'question': question, 'analysis': analysis, 'query': query,
                    'entity': current_entity, 'suggestion': suggestion, 
                    'search_results_list': search_results,
                    'search_history': formatted_history
                }

            llm_decision = generate_process(
                step_name=f"Search Refinement for '{current_entity}' (Attempt {attempt + 1})",
                prompt_template=prompt_template,
                template_inputs=template_inputs,
                parsing_function=parse_refine_search_selection,
                args=args, module='wiki', max_retries=3
            )

            if not llm_decision:
                reason = f"LLM failed to produce a valid query refinement after multiple retries."
                interaction_log["outcome"] = f"Failure: {reason}"
                trace.append(interaction_log)
                print(reason)
                return "FAILED", reason, trace

            new_entity = llm_decision['choice']
            rationale = llm_decision['rationale']
            interaction_log["llm_rationale"] = rationale
            interaction_log["llm_selected_entity"] = new_entity
            print(f"LLM Rationale: {rationale}")
            print(f"LLM selected new entity: '{new_entity}'")

            history_entry = {
                "attempt": attempt + 1,
                "searched_query": current_entity,
                "result_type": "Not Found",
                "suggestion": suggestion,
                "search_results_provided": search_results,
                "llm_rationale": rationale,
                "llm_choice": new_entity
            }
            search_history.append(history_entry)

            if not new_entity or new_entity.strip() == "NO_MATCH":
                reason = "LLM decided that no suggestions were relevant from the search results."
                interaction_log["outcome"] = f"Failure: {reason}"
                trace.append(interaction_log)
                print(reason)
                return "FAILED", reason, trace
            
            # 为下一次尝试做准备
            current_entity = new_entity.strip()
            interaction_log["outcome"] = f"Refining search with new entity: '{current_entity}'."
            
        elif status == "ERROR":
            _query, error_message = result
            if "API Response Error" in error_message:
                reason = f"Wikipedia API format error for query '{_query}': {error_message}. Please check if the query namespace is supported."
            else:
                reason = f"Wikipedia API error for query '{_query}': {error_message}. It may be temporary network issues. Please try again."
            
            interaction_log["outcome"] = f"Failure: {reason}"
            trace.append(interaction_log)
            print(reason)
            return "ERROR", reason, trace
        
        trace.append(interaction_log)
        # time.sleep(1) # 在两次尝试之间短暂延迟

    # 如果循环完成，意味着在达到最大交互次数后仍未成功
    final_reason = f"Failed to retrieve a valid page after {max_interactions} interactions. Last entity tried: '{current_entity}'."
    print(final_reason)
    return "FAILED", final_reason, trace