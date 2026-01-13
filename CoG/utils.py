from string import Template
import json
import time
import openai
import textwrap
import re
from typing import Any, List
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers

class E5InstructEmbeddings(HuggingFaceEmbeddings):
    def _embed(self, texts: List[str], encode_kwargs: dict[str, Any]) -> List[List[float]]:
        # 重写 _embed 以跳过 text.replace("\n", " ") 操作
        
        # 直接调用模型进行编码，保留原始文本格式（包括换行符）
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # 为 embed_query 增加默认指令逻辑
        # 如果文本没有显式的 "Instruction:" 前缀，则添加默认检索指令
        if not text.startswith("Instruction:"):
            # 默认指令，用于 workflow_table.py 等通用检索场景
            default_instruction = "Instruction: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            text = f"{default_instruction}{text}"
        
        return super().embed_query(text)

def run_llm(prompt, args, module='main', n=1, **generation_params):
    base_url = args.base_url
    model_name = args.model
    if module == 'kg':
        base_url = args.kg_base_url if args.kg_base_url else args.base_url
        model_name = args.kg_model if args.kg_model else args.model
    elif module == 'wiki':
        base_url = args.wiki_base_url if args.wiki_base_url else args.base_url
        model_name = args.wiki_model if args.wiki_model else args.model

    client = openai.OpenAI(base_url=base_url, timeout=600 * 6)

    sys_prompt = '''You are a helpful assistant'''
    messages = [{"role":"system","content":sys_prompt}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    params = {
        'model': model_name,
        'messages': messages,
        'n': n,
    }
    if 'gpt' in model_name:
        max_tokens = generation_params.pop('max_tokens', None)
        if max_tokens:
            params['max_completion_tokens'] = max_tokens
    params.update(generation_params)
    
    if 'Qwen3-32B' in model_name:
        if args.enable_thinking:
            params['temperature'] = 0.6
            params['top_p'] = 0.95
        else:
            params['temperature'] = 0.7
            params['top_p'] = 0.8
        params['extra_body'] = {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
        
    f = 4
    error_repr = "Unknown error after 4 retries."
    while(f > 0):
        try:
            response = client.chat.completions.create(**params)

            if 'Qwen3-32B' in model_name: # Qwen3-32B models
                if n > 1:
                    for choice in response.choices:
                        if "</think>" in choice.message.content:
                            choice.message.content = choice.message.content.split("</think>")[-1].strip()
                    return response, None
                else:
                    result = response.choices[0].message.content
                    if "</think>" in result:
                        result = result.split("</think>")[-1].strip()
                    return result, None
            else: # openAI models
                if n > 1:
                    return response, None
                else:
                    result = response.choices[0].message.content
                return result, None
        except Exception as e:
            error_repr = repr(e)
            print('ERROR fail to run llm: ',error_repr)
            time.sleep(10)
            f -= 1
    return '', error_repr


def generate_process(step_name: str, prompt_template: Template, template_inputs: dict, parsing_function, args, module: str = 'main', max_retries: int = 3, **llm_params):
    """
    A generic function for executing a single "generate-process" step with retry logic.

    This function encapsulates a standard processing flow:
    1. Constructs a prompt using the provided template and inputs.
    2. Calls the large language model (LLM) to get the result.
    3. Uses the specified parsing function to process the LLM's output.
    4. If the LLM call or parsing fails, it will be retried.
    """
    last_llm_error = None
    for attempt in range(max_retries):
        # print(f"--- Running Step: {step_name} (Attempt {attempt + 1}/{max_retries}) ---")
        prompt = prompt_template.substitute(**template_inputs)
        result_text, llm_error = run_llm(prompt, args, module=module, **llm_params)
        
        if llm_error:
            last_llm_error = llm_error
            print(f"Attempt {attempt + 1} failed: LLM call returned an error. Retrying in 5 seconds...")
            time.sleep(5)
            continue
            
        # Sanitize common LLM formatting like fenced code blocks before parsing
        sanitized_text = strip_markdown_code_block(result_text)
        result = parsing_function(sanitized_text)
        if not result and sanitized_text != result_text:
            # Fallback: try parsing the raw text if sanitized failed
            result = parsing_function(result_text)
        if result:
            return result
        else:
            print(f"Attempt {attempt + 1} failed: Could not parse LLM output.")
            # print("Generated Result:\n", result_text)
            # time.sleep(5)
            
    print(f"Failed to get a valid result for step '{step_name}' after {max_retries} attempts.")
    if last_llm_error:
        if hasattr(args, 'current_llm_errors') and isinstance(args.current_llm_errors, set):
            args.current_llm_errors.add(last_llm_error)
    return None


def prepare_dataset(dataset_name):

    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'hotpot_e':
        with open('../data/hotpotadv_dev.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/webqsp_test.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == '2wiki':
        with open('../data/2wikimultihopqa.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'KGQAGen':
        with open('../data/KGQAGen-10k.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'musique':
        with open('../data/musique.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("Dataset not found.")
        exit(-1)
    return datas, question_string


def indent_text(text: str, indent: str, add_space: bool = False, use_indent: bool = True) -> str:
    """
    Formats and indents text based on whether it's single or multi-line.
    - For single-line text, returns the text with an optional leading space.
    - For multi-line text, adds a newline at the start and indents all lines.
    """
    if not use_indent:
        # If indentation is disabled, return the original, cleaned text
        if not isinstance(text, str):
            text = str(text)
        return textwrap.dedent(text).strip()

    if not isinstance(text, str):
        text = str(text)
    
    # Clean up the text by removing leading/trailing whitespace and dedenting
    cleaned_text = textwrap.dedent(text).strip()
    
    # If the text is empty, return an empty string
    if not cleaned_text:
        return ""

    # Check if the cleaned text contains a newline character
    if '\n' in cleaned_text:
        # For multi-line text, indent the whole block and add a newline before it
        indented_text = textwrap.indent(cleaned_text, indent)
        return f"\n{indented_text}"
    else:
        # For single-line text, just add a leading space if requested
        return f" {cleaned_text}" if add_space else cleaned_text


def strip_markdown_code_block(text: str) -> str:
    """
    Remove surrounding or embedded markdown code fences from LLM output.
    - Handles opening fences like ```json, ```text, ```text/json, etc.
    - If the whole content is fenced, returns the inner content.
    - If a fenced block appears inside a larger message, returns the first block's content.
    - Otherwise returns the trimmed original string.
    """
    if not isinstance(text, str):
        return text

    stripped_text = text.strip()

    # Case 1: Entire content is one fenced block
    lines = stripped_text.split('\n')
    if len(lines) >= 2 and lines[0].startswith('```') and lines[-1].strip() == '```':
        inner = '\n'.join(lines[1:-1]).strip()
        return inner

    # # Case 2: Extract the first fenced block inside a larger text
    # fence_pattern = re.compile(r"```[^\n]*\n([\s\S]*?)\n```", re.MULTILINE)
    # match = fence_pattern.search(stripped_text)
    # if match:
    #     return match.group(1).strip()

    # Case 3: No fences found, return as-is (trimmed)
    return stripped_text