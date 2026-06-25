from typing import Iterable, List
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
import json

try:
    from llama_cpp import LlamaGrammar
except ImportError:
    LlamaGrammar = None


def make_enum_grammar(labels: Iterable[str], add_other: bool = True) -> LlamaGrammar:
    """
    if LlamaGrammar is None:
        raise ImportError(
            "llama-cpp-python is required for make_enum_grammar(). "
            "Install it only if you need local llama.cpp grammar support."
        )

    Create a llama.cpp GBNF grammar that forces output to be exactly one of the given labels.

    Example output:
        root ::= intent
        intent ::= "a" | "b" | "c" | "other"
    """
    # normalize, deduplicate, keep order
    seen = set()
    items: List[str] = []
    for x in labels:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            items.append(x)

    if add_other and "other" not in seen:
        items.append("other")

    if not items:
        raise ValueError("labels is empty after normalization")

    def gbnf_quote(s: str) -> str:
        # Escape for GBNF string literal
        # (quotes and backslashes are the main ones you may hit)
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{s}\""

    options = " | ".join(gbnf_quote(x) for x in items)

    grammar_str = f"""
root ::= intent
intent ::= {options}
""".strip()
    
    grammar = LlamaGrammar.from_string(grammar_str)

    return grammar



def make_messages(system_prompt_path: str, memory: str, user_prompt: str, max_retry: int = 10) -> List:
    # Retrieve system prompt + calculate length
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    system_prompt_length = len(system_prompt)
    # TODO: Identify memory

    memory = ""

    # TODO: Limit the length of input to 4000????? letters

    user_prompt_length = len(user_prompt)
    for _ in range(max_retry):
        total_length = system_prompt_length + len(memory) + user_prompt_length
        if total_length < 4000:
            break
        else:
            memory = memory - ""
    else:
        raise NameError
    system_prompt += memory

    # TODO: Message configuration 
    ## Order: System!! - memory - user
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

Role = Literal["system", "user", "assistant"]
@dataclass
class Message:
    role: Role
    content: str

class WatsonxLLM:
    def __init__(self, model):
        self.model = model

    def chat(self, messages, temperature=0.1, max_tokens=4000):

        response = self.model.chat(messages=messages)

        return response["choices"][0]["message"]["content"]
    
    def detect(self, messages, tools: List[dict] = None):

        response = self.model.chat(messages=messages, tools = tools)

        return response["choices"][0]["message"]["tool_calls"]
    
    def judge(self, messages, tools: List[dict] = None) -> list:

        res = self.model.chat(messages=messages, tools = tools)

        message = res["choices"][0]["message"]
        finish_reason = res["choices"][0].get("finish_reason")
        try:
            if finish_reason == "tool_calls" or message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if function_name == "request_user_clarification":
                    question = arguments["question"]
                    reason = arguments.get("reason", "")

                    return {
                        "action": "ask_user",
                        "reason": reason,
                        "question": question
                        }
            else:
                content = message.get("content", "")
                decision = json.loads(content)

                if decision.get("need_clarification") is False:
                    return {
                        "action": "continue_generation",
                        "reason": decision.get("reason", ""),
                        "assumptions": decision.get("assumptions", [])
                    }

                return {
                    "action": "continue_generation",
                    "reason": "No clarification tool call was returned.",
                    "assumptions": []
                }
        except Exception:
            print("Error occured within WatsonxLLM.judge")

        raise ValueError(f"Unrecognized response format: {type(res)} -> {res}")
 
