import asyncio
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from memory.utils import WatsonxLLM
from pydantic import ValidationError
from memory.storage import SQLiteManager
#from factory import VectorStoreFactory
logger = logging.getLogger(__name__)

class Memory():
    def __init__(self, db_path, summary_prompt_path, judge_prompt_path):
        self.db = SQLiteManager(db_path)

        with open(summary_prompt_path, "r", encoding = "utf-8") as f:
            self.summary_prompt = f.read()

        with open(judge_prompt_path, "r", encoding = "utf-8") as f:
            self.judge_prompt = f.read()
        return None

    def _add_history(self, data: str, session_id: str, metadata=None):
        logger.debug(f"Creating memory with {data=}")

        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        #new_metadata["updated_at"] = new_metadata["created_at"]

        self.db.add_history(
            session_id,
            data,
            created_at=new_metadata.get("created_at"),
            role=new_metadata.get("role"),
        )
        return None
    
    def add_session_history(self, messages: list[dict], session_id: str):
        """ OUTER FUNCTION"""
        for message_dict in messages:
            if (
                not isinstance(message_dict, dict)
                or message_dict.get("role") is None
                or message_dict.get("content") is None
            ):
                logger.warning(f"Skipping invalid message format: {message_dict}")
                continue

            if message_dict["role"] == "system":
                continue

            per_msg_meta = {}
            per_msg_meta["role"] = message_dict["role"]
            msg_content = message_dict["content"]

            self._add_history(msg_content, session_id, per_msg_meta)
            print(f"The following chat history is stored: {message_dict}")

        return None

    def _update_memory(self, key, content, session_id, model):
        if not content:
            print("No info should be updated to Database.")
            return None
        #print(f"DEBUG: Attempting to access table: [{key}]")
        table = self.db.get_table(key, session_id)
        #print(type(table))
        #print(f"This is the old {key} table: {table}")
        if not table:
            if isinstance(content, list):
                str_content = " ".join(map(str, content)) 
            else:
                str_content = str(content) if content is not None else ""
            self.db.update_table(key, session_id, str_content)
            print(f"First item is added to Table {key}: {str_content}.")
            rt = self.db.get_table(key, session_id)
            return rt
        #print(f"type of content in update_memory is {type(content)}, content is {content}, len is {len(content)}")
        if isinstance(content, list):
            str_content = " ".join(map(str, content)) 
        else:
            str_content = str(content) if content is not None else ""
        #str_content = " ".join(content) if isinstance(content, list) and len(content) > 1 else str(content[0])
        #print(f"This is str_content:{str_content}")
        if isinstance(table, list):
            formatted_table = " ".join(map(str, table))
        else:
            formatted_table = str(table) if table is not None else ""
        #formatted_table = " ".join(table) if isinstance(table, list) and len(table) > 1 else str(table[0])
        #print(f"This is formatted_table:{formatted_table}")
        judge_message = [
                {
                    "role": "system",
                    "content": self.judge_prompt,
                },
                {
                    "role": "user",
                    "content": formatted_table + str_content
                }
            ]
        try:
            result = model.chat(judge_message)
            #print(type(result))
            #print(f"This is result:{result}")
        except Exception as e:
                logger.error(f"The LLM respones are not strict list of string: {e}")
                raise
        try:
                processed_result = json.loads(result)
        except json.JSONDecodeError:
                processed_result = [item.strip() for item in result.split(".") if item.strip()]
        if isinstance(processed_result, list):
            self.db.clear_table(key)
            for item in processed_result:
                self.db.update_table(key, session_id, item)
            #self.db.update_table(key, session_id, processed_result)
        
        """if isinstance(result, list):
            for item in result:
                self.db.update_table(key, session_id, item)
        elif isinstance(result, str):
            result = [item.strip() for item in result.split(".") if item.strip()]
            if len(result) > 1:
                for item in result:
                    self.db.update_table(key, session_id, item)
            else:
                self.db.update_table(key, session_id, str(result[0]))"""
        
        new_table = self.db.get_table(key, session_id)
        return new_table

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            print(f"Cannot get history, session id is {session_id}.")
        history = self.db.get_history(session_id = session_id)
        return history    

    def update_intent(
            self,
            intent: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> None:
            if intent:
                self.db.clear_table("intent")
                self.db.update_table("intent", session_id, intent)
                print(f"Intent: {intent} Stored.")
            else:
                print(f"In Update_Intent: intent is {intent}")
            return None

    def update_knowledge(
            self,
            knowledge: Optional[list] = None,
            session_id: Optional[str] = None,
    ) -> None:
            #self.db.clear_table("intent")
            if not knowledge:
                return None
            if isinstance(knowledge, str):
                knowledge = [knowledge]
            for item in knowledge:
                self.db.update_table("knowledge", session_id, item)
            return None

    def add(
            self,
            messages,
            *,
            session_id: Optional[str] = None,
            model: Optional[WatsonxLLM] = None,
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            return "messages must be str, dict, or list[dict]"
        
        # === Session History ===
        self.add_session_history(messages, session_id)

        # === Classification ===
        json_object = json.dumps(messages, ensure_ascii=False)
        sum_message = [
                    {
                        "role": "system",
                        "content": self.summary_prompt,
                    },
                    {
                        "role": "user",
                        "content": json_object
                    }
                ]
        res = model.chat(sum_message)
        #print(res)
        if isinstance(res, str):
            result = json.loads(res)
        else:
            result = res
        #result = json.loads(res)
        
        if isinstance(result["data"], str):
            result["data"] = json.loads(result["data"])
        elif isinstance(result["data"], dict):
            # 如果已经是 dict，就保持原样，不做处理
            pass
        else:
            result["data"] = None
        # === Update Storage ===
        res_list = []
        for key, value in result.items():
            #print(f"{key} started")
            #print(f"Content is {value}.")
            new_table = self._update_memory(key, value, session_id, model)
            #print(f"New Table is {new_table}") 
            res_list.append(new_table)
        return res_list
    
    def get_memory(self, key: str = "", session_id: str = ""):

        table = self.db.get_table(key, session_id)
        #print(f"Table type is: {type(table)}. Get the {key} table: {table}")
        return table
