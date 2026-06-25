from tools import WatsonxLLM
from datetime import datetime, timezone
import json
""" 
labels = {
    "Process structure",
    "Distribution of cases over paths",
    "Throughput time of cases",
    "Resource utilization rate",
    "other",
}
"""

def intent_detection(
        system_prompt: str,
        user_prompt: str,
        llm
) -> str:
    messages = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    res = llm.chat(messages)

    return res 

def get_intent(system_prompt: str, user_input: str, his_conversation: list, llm: WatsonxLLM) -> str:
    """ 
    user_input: the pure str of user's input.
    """
    his_input= []
    if his_conversation:
        for log in his_conversation:
            his_input.append(
                {
                    "role": log["role"],
                    "content": log["content"],
                }
            )
    his_input.append(
        {
            "role": "user",
            "content": user_input,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    )
    user_prompt = f"""
        The following messages are sorted chronologically, with the oldest message at the beginning:

        {json.dumps(his_input, ensure_ascii=False, indent=2)}
    """
    intent = intent_detection(system_prompt, user_prompt, llm)
    
    return intent