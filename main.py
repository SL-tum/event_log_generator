from intent_detection.intent_detection import get_intent
from generation.prompt_construction import f_prompt
from generation.generation import generation_loop
from graph_rag.c_local_search import local_search
import yaml
import pandas as pd
from graphrag.config.models.graph_rag_config import GraphRagConfig
import json
import os
from pathlib import Path
from memory.main import Memory
from tools import WatsonxLLM
from huggingface_hub.utils import disable_progress_bars
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as GenParams
import asyncio
from datetime import datetime

disable_progress_bars()

root_folder_path = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    load_dotenv(root_folder_path / ".env")


# === Memory ===
db_path = str(root_folder_path / "history.db")
memory_summary_prompt_path = fr"{root_folder_path}/prompt/memory_summary_prompt.txt"
memory_judge_prompt_path = fr"{root_folder_path}/prompt/memory_judge_prompt.txt"

# === Intent Detection ===
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
watsonx_api_key = os.getenv("WATSONX_API_KEY")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
general_model_id = os.getenv("WATSONX_GENERAL_MODEL_ID", "openai/gpt-oss-120b")
judge_model_id = os.getenv("WATSONX_JUDGE_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
id_path = fr"{root_folder_path}/intent_detection"
id_system_prompt_path = f"{root_folder_path}/prompt/system_prompt_id.txt"
#id_QA_PATH = f"{id_path}/QA_4_100.json"
id_OUT_PATH = f"{id_path}/results_{RUN_TS}.json"

# === Graph RAG ===
graph_rag_path = fr"{root_folder_path}/graph_rag"
graph_rag_qa_path = fr"{graph_rag_path}/QA.json"
graph_rag_setting_path = fr"{graph_rag_path}/settings.yaml"

# === Judge ===
judge_system_prompt_path = fr"{root_folder_path}/prompt/judge_system_prompt.txt"
judge_user_prompt_path = fr"{root_folder_path}/prompt/judge_user_prompt.txt"

# === Detector ===
detector_system_prompt_path = fr"{root_folder_path}/prompt/detector_system_prompt.txt"
detector_user_prompt_path = fr"{root_folder_path}/prompt/detector_user_prompt.txt"

# === Process Model ===
pm_folder_path = fr"{root_folder_path}/data/input"
event_log_folder_path = fr"{root_folder_path}/event_log"


def require_env(name: str, value: str | None) -> str:
    if value:
        return value
    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        "Copy .env.example to .env and fill in your credentials."
    )


def load_graph_rag_tables() -> dict[str, pd.DataFrame]:
    output_dir = Path(graph_rag_path) / "output"
    required_files = {
        "documents": output_dir / "documents.parquet",
        "text_units": output_dir / "text_units.parquet",
        "entities": output_dir / "entities.parquet",
        "relationships": output_dir / "relationships.parquet",
        "communities": output_dir / "communities.parquet",
        "community_reports": output_dir / "community_reports.parquet",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "GraphRAG index files are missing. Run GraphRAG indexing first, "
            f"or restore the prebuilt files. Missing: {missing}"
        )
    return {name: pd.read_parquet(path) for name, path in required_files.items()}


def expand_env_values(value):
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


async def get_knowledge(root_folder_path: str = "", user_input: str = "") -> list[str]:
    graph_rag_tables = load_graph_rag_tables()

    with open(graph_rag_setting_path, 'r', encoding='utf-8') as f:
        graph_rag_setting = yaml.load(f.read(), Loader=yaml.FullLoader)
    graph_rag_setting = expand_env_values(graph_rag_setting)

    graph_rag_setting['root_dir'] = str(Path(graph_rag_path).resolve())

    graphRagConfig = GraphRagConfig(**graph_rag_setting)

    answer, context = await local_search(
    root_folder_path=root_folder_path,
    config=graphRagConfig,
    entities=graph_rag_tables["entities"],
    communities=graph_rag_tables["communities"],
    community_reports=graph_rag_tables["community_reports"],
    text_units=graph_rag_tables["text_units"],
    relationships=graph_rag_tables["relationships"],
    community_level=2,
    response_type="Single Paragraph",
    query=user_input,
    )

    return answer, context

async def main():
    print("The Event Log Generation Framework is now running (type 'exit' to quit).")
    RUN_TS_I = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem = Memory(db_path, memory_summary_prompt_path, memory_judge_prompt_path)
    round_messages = []
    running = True
    current_session_id = f"a_{RUN_TS_I}"
    require_env("WATSONX_API_KEY", watsonx_api_key)
    require_env("WATSONX_PROJECT_ID", watsonx_project_id)
    credentials = Credentials(
    url = watsonx_url,
    api_key = watsonx_api_key)
    client = APIClient(credentials)
    general_params = {
        GenParams.MAX_TOKENS: 10000,
        GenParams.MAX_COMPLETION_TOKENS: 8048
    }
    general_model = ModelInference(
        model_id=general_model_id,
        api_client=client,
        params=general_params,
        project_id=watsonx_project_id)
    general_llm = WatsonxLLM(general_model)

    pm_folder = Path(pm_folder_path)
    fp = f_prompt()
    bpmn_files = list(pm_folder.glob("*.bpmn"))
    pm = "No Process Model given."
    if bpmn_files:
        print("Found bpmn files.")
        with open(bpmn_files[0], "r", encoding = "utf-8") as f:
            pm = f.read()
    else:
        print("Please only provide one process model. Or no process model is given.")

    judge_params = {
        GenParams.MAX_TOKENS: 10000,
        GenParams.MAX_COMPLETION_TOKENS: 8048
    }
    with open(detector_system_prompt_path, "r", encoding= "utf-8") as f:
        detector_system_prompt_template = f.read()

    detector_tools = [
    {
        "type": "function",
        "function": {
            "name": "set_generation_stage_decision",
            "description": "Return the decision about whether the user intends to enter the event log generation stage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": [
                            "ENTER_GENERATION_STAGE",
                            "DO_NOT_ENTER_GENERATION_STAGE"
                        ],
                        "description": "ENTER_GENERATION_STAGE if the user clearly wants to start, continue, retry, or finalize event log generation now. DO_NOT_ENTER_GENERATION_STAGE otherwise."
                    }
                },
                "required": [
                    "decision"
                ],
                "additionalProperties": False
            }
        }
    }
]
    judge_model = ModelInference(
        model_id=judge_model_id,
        api_client=client,
        params=judge_params,
        project_id=watsonx_project_id,
        verify = False)
    judge_llm = WatsonxLLM(judge_model)
    print("Hello, How can I help you today?")
    while running:
        user_input = input("\n>>> ")
        if user_input.lower() in ['exit', 'quit']:
            round_messages.append({"role": "user", "content": user_input})
            mem.add(messages = round_messages, session_id=current_session_id, model = general_llm)
            round_messages.clear()
            print("User quit.")
            break

        # === Detector === 
        detector_messages = [
            {"role": "system", "content": detector_system_prompt_template},
            {"role": "user", "content": [{
                "type": "text",
                "text": user_input
            }]},
        ]
        detecto_res = judge_llm.detect(messages=detector_messages, tools = detector_tools)
        decision = json.loads(detecto_res[0]["function"]["arguments"])["decision"]
        if decision == "ENTER_GENERATION_STAGE":
            round_messages.append({"role": "user", "content": user_input})
            mem.add(messages = round_messages, session_id=current_session_id, model = general_llm)
            round_messages.clear()
            print("Start generation.")
            requirements_table = mem.get_memory("requirement", session_id=current_session_id)
            data_table = mem.get_memory("data", session_id=current_session_id)
            result = await generation_loop(
                mem=mem,
                round_messages=round_messages,
                current_session_id=current_session_id,
                general_llm=general_llm,
                f_prompt_instance=fp,
                process_model=pm,
                config=data_table,
                special_requirements=", ".join(requirements_table) if requirements_table else "",
                output_base_dir= event_log_folder_path,
            )
            print(f"Done. Files in: {result['run_dir']}")
            break
        round_messages.append({"role": "user", "content": user_input})
        # === Intent Detection ===
        print("Intent Detection Started.")
        with open(id_system_prompt_path, "r", encoding= "utf-8") as f:
            id_system_prompt = f.read()
        his_conversation = mem.get_history(session_id=current_session_id)
        cleaned_history = [{"role": d["role"], "content": d["event"]} for d in his_conversation]
        intent = get_intent(id_system_prompt, user_input, cleaned_history, general_llm)
        print(f"Intent: {intent} Detected.")
        for item in ["other", "Other", "others", "Others"]:
            if (item in intent) and len(intent) >= 10:
                intent = intent.replace(item, "")
            elif (item in intent):
                item = None
            else:
                continue
        mem.update_intent(intent, current_session_id)
        
        # === Knowledge Retrieval ===
        print("Knowledge Retrieval Started.")
        try:
            knowledge, context = await get_knowledge(root_folder_path, user_input)
        except Exception as exc:
            knowledge, context = "", {}
            print(f"Knowledge retrieval skipped: {exc}")
        mem.update_knowledge(knowledge, current_session_id)
        print(f"Knowledge Retrieved.")

        # === Judge ===
        print("Judge Started.")
        with open(judge_system_prompt_path, "r", encoding= "utf-8") as f:
            judge_system_prompt_template = f.read()
        with open(judge_user_prompt_path, "r", encoding= "utf-8") as f:
            judge_user_prompt_template = f.read()

        mem.add(messages = round_messages, session_id=current_session_id, model = general_llm)
        round_messages.clear()

        requirements_table = mem.get_memory("requirement", session_id=current_session_id)
        data_table = mem.get_memory("data", session_id=current_session_id)
        perference_table = mem.get_memory("preference", session_id=current_session_id)
        constrain_table = mem.get_memory("constrain", session_id=current_session_id)
        next_his_conversation = mem.get_history(session_id=current_session_id)
        cleaned_next_history = [{"role": d["role"], "content": d["event"]} for d in next_his_conversation]
        judge_user_prompt = judge_user_prompt_template.format(
            intent = intent,
            knowledge = knowledge,
            requirements_table = ", ".join(requirements_table) if requirements_table else "",
            data_table = ", ".join(data_table) if data_table else "",
            perference_table = ", ".join(perference_table) if perference_table else "",
            constrain_table = ", ".join(constrain_table) if constrain_table else "",
            history = cleaned_next_history
        )

        judge_messages = [
        {"role": "system", "content": judge_system_prompt_template},
        {"role": "user", "content": [{
            "type": "text",
            "text": judge_user_prompt
        }]},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "request_user_clarification",
                    "description": "Trigger a user clarification step when the event log generation request is incomplete, ambiguous, or conflicting.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                            "type": "string",
                            "description": "Brief explanation of why clarification is required."
                            },
                            "missing_or_ambiguous_information": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Key information items that are missing, ambiguous, or conflicting."
                            },
                            "question": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Clarification question to ask the user. Only ONE question is allowed."
                            }
                        },
                        "required": [
                            "reason", "question"
                        ]
                    }
                }
            }
        ]

        judge_result = judge_llm.judge(messages = judge_messages, tools = tools)
        #print(f"This is judge_result: {judge_result}")
        if isinstance(judge_result["question"], list):
            judge_content = " ".join(judge_result["question"])
        else:
            judge_content = judge_result["question"]

        if judge_result["action"] != "ask_user":

            round_messages.append({"role": "assistant", "content": judge_content})
            mem.add(messages = round_messages, session_id=current_session_id, model = general_llm)
            round_messages.clear()
            print(f"Judgement finished. Start Generation.")

            result = await generation_loop(
                mem=mem,
                round_messages=round_messages,
                current_session_id=current_session_id,
                general_llm=general_llm,
                f_prompt_instance=fp,
                process_model=pm,
                config=", ".join(data_table) if data_table else "",
                special_requirements=", ".join(requirements_table) if requirements_table else "",
                output_base_dir= event_log_folder_path,
            )
            print(f"Done. Files in: {result['run_dir']}")
            break

        round_messages.append({"role": "assistant", "content": judge_content})
        mem.add(messages = round_messages, session_id=current_session_id, model = general_llm)
        round_messages.clear()
        print(f"Judgement finished. Need Further Clarification.")
        print(judge_content)



        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
