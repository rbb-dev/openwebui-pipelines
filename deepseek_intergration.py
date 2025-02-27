"""
title: Deepseek Intergration Chat and Image
authors: brammittendorff
author_url: https://github.com/brammittendorff/openwebui-pipelines
funding_url: https://github.com/open-webui
version: 0.0.1
required_open_webui_version: 0.3.20
license: MIT
environment_variables:
    - DEEPSEEK_API_KEY (required)
"""
import json
import logging
import os
import re
import time
from typing import AsyncIterator, Dict, List, Union

import httpx
from open_webui.utils.misc import pop_system_message
from pydantic import BaseModel, Field


class Pipe:
    REQUEST_TIMEOUT = (3.05, 60)
    MODEL_MAX_TOKENS = {
        "deepseek-chat": 8192,
        "deepseek-reasoner": 8192,
    }

    class Valves(BaseModel):
        DEEPSEEK_BASE_URL: str = Field(
            default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            description="DeepSeek API URL",
        )
        DEEPSEEK_API_KEY: str = Field(
            default=os.getenv("DEEPSEEK_API_KEY", ""),
            description="DeepSeek API Key",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "deepseek"
        self.name = "deepseek/"
        self.valves = self.Valves()
        self.request_id = None
        self.clean_pattern = re.compile(r"<details>.*?</details>\n\n", flags=re.DOTALL)
        self.buffer_size = 3
        self.thinking_state = -1  # -1: Not started, 0: Thinking, 1: Answered

    @staticmethod
    def get_model_id(model_name: str) -> str:
        return model_name.replace(".", "/").split("/")[-1]

    def get_deepseek_models(self) -> List[Dict[str, str]]:
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            with httpx.Client() as client:
                response = client.get(
                    f"{self.valves.DEEPSEEK_BASE_URL}/models",
                    headers=headers,
                    timeout=10,
                )
            response.raise_for_status()
            models_data = response.json()
            return [
                {"id": model["id"], "name": model["id"]}
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            logging.error(f"Error getting models: {e}")
            return []

    def pipes(self) -> List[dict]:
        return self.get_deepseek_models()

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> AsyncIterator[str]:
        buffer = []
        self.thinking_state = -1
        last_status_time = time.time()
        status_dots = 0
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                            
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                            
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse data line: {data_str}, error: {e}")
                            continue
                            
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        reasoning = delta.get("reasoning_content") or ""
                        content = delta.get("content") or ""
                        finish_reason = choice.get("finish_reason")
                        
                        if self.thinking_state == -1 and reasoning:
                            self.thinking_state = 0
                            buffer.append("<details>\n<summary>Thinking Process</summary>\n\n")
                        elif self.thinking_state == 0 and not reasoning and content:
                            self.thinking_state = 1
                            buffer.append("\n</details>\n\n")
                            
                        if self.thinking_state == 0 and model_id == "deepseek-reasoner":
                            current_time = time.time()
                            if current_time - last_status_time > 1:
                                status_dots = (status_dots % 3) + 1
                                last_status_time = current_time
                                
                        if reasoning:
                            buffer.append(reasoning.replace("\n", "\n> "))
                        elif content:
                            buffer.append(content)
                            
                        if finish_reason == "stop":
                            if self.thinking_state == 0:
                                buffer.append("\n</details>\n\n")
                            break
                            
                        if len(buffer) >= self.buffer_size or "\n" in (reasoning + content):
                            yield "".join(buffer)
                            buffer.clear()
                            
                    if buffer:
                        yield "".join(buffer)
        except Exception as e:
            error_msg = f"Stream Error: {str(e)}"
            yield error_msg

    async def _regular_request(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    original_content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")
                    
                    if reasoning:
                        processed_content = (
                            f"<details>\n<summary>Thinking Process</summary>\n\n"
                            f"{reasoning}\n</details>\n\n{original_content}"
                        )
                        processed_content = self.clean_pattern.sub("", processed_content).strip()
                        data["choices"][0]["message"]["content"] = processed_content
                        data["choices"][0]["message"]["reasoning_content"] = reasoning
                        
                return data
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Regular request failed: {error_msg}")
            return {"error": error_msg, "choices": []}

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[AsyncIterator[str], dict]:
        if not self.valves.DEEPSEEK_API_KEY:
            return {"error": "DEEPSEEK_API_KEY is required", "choices": []}
            
        try:
            system_message, messages = pop_system_message(body.get("messages", []))
            
            for msg in messages:
                if msg.get("role") == "assistant" and "content" in msg:
                    msg["content"] = self.clean_pattern.sub("", msg["content"]).strip()
                    
            model_id = self.get_model_id(body["model"])
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_id, 8192)
            
            if system_message:
                messages.insert(0, {"role": "system", "content": str(system_message)})
                
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                ),
                "temperature": float(body.get("temperature", 0.7)),
                "stream": body.get("stream", False),
            }
            
            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            
            if payload["stream"]:
                return self._stream_response(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )
            else:
                return await self._regular_request(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Pipe processing failed: {error_msg}")
            return {"error": error_msg, "choices": []}
