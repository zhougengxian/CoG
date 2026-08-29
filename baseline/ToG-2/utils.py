"""LLM helpers used by the CoG ToG-2 baseline.

Adapted from IDEA-FinAI/ToG-2. Token, timing, and document accounting were
removed for the CoG release.
"""

import re
import time

from prompt_list import *  # noqa: F403


def run_llm(
    prompt,
    temperature,
    max_tokens,
    api_key,
    engine,
    n=1,
    enable_thinking=False,
    base_url=None,
):
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=3600)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(4):
        try:
            kwargs = {
                "model": engine,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": n,
            }
            if "qwen3" in engine.lower():
                kwargs["temperature"] = 0.6 if enable_thinking else 0.7
                kwargs["top_p"] = 0.95 if enable_thinking else 0.8
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            response = client.chat.completions.create(**kwargs)
            if n > 1:
                if "qwen3" in engine.lower():
                    for choice in response.choices:
                        content = choice.message.content or ""
                        if "</think>" in content:
                            choice.message.content = content.split("</think>")[-1].strip()
                return response
            result = response.choices[0].message.content or ""
            if "</think>" in result:
                result = result.split("</think>")[-1].strip()
            return result
        except Exception as exc:
            print(f"LLM request failed ({attempt + 1}/4): {exc}")
            if attempt < 3:
                time.sleep(10)
    return ""


def extract_answer(text):
    match = re.search(r"\{(.*?)\}", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def if_true(prompt):
    return prompt.lower().strip().replace(" ", "") == "yes"


def generate_only_with_gpt(question, args):
    if args.gpt_only == "io":
        prompt = generate_directly + "\n\nQ: " + question + "\nA:"  # noqa: F405
    else:
        prompt = cot_prompt + "\n\nQ: " + question + "\nA:"  # noqa: F405
    return run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.model,
        base_url=args.base_url,
    )


def self_consistency(question, data, idx, args):
    del idx
    prompt = hotpotqa_s1_prompt_demonstration + "Q: " + question.strip() + "\nA: "  # noqa: F405
    responses = run_llm(
        prompt,
        0.7,
        args.max_length,
        args.api_key,
        args.model,
        n=11,
        base_url=args.base_url,
    )
    if responses is None:
        raise RuntimeError("Self-consistency LLM request failed")

    texts = [(choice.message.content or "").strip() for choice in responses.choices]
    answers = []
    answer_texts = []
    for text in texts:
        if "The answer is" in text:
            answers.append(text.split("The answer is", 1)[1].strip().lower())
            answer_texts.append(text)
    if answers:
        majority = max(set(answers), key=answers.count)
        majority_index = answers.index(majority)
        score = answers.count(majority) / 11
        response = answer_texts[majority_index]
    else:
        score = 0
        response = "No answer found"

    data_point = dict(data)
    data_point["cot_sc_score"] = score
    data_point["cot_sc_response"] = response
    return data_point


def if_finish_list(values):
    remaining = [value for value in values if value != "[FINISH_ID]"]
    return not remaining, remaining
