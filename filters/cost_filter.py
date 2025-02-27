"""
title: Cost Tracker
description: Manages and calculates costs for model usage in an Open WebUI
author: maki
version: 1.0.0
license: MIT
requirements: requests, tiktoken, cachetools, pydantic
environment_variables:
disclaimer: Provided as-is without warranties. 
            You must ensure it meets your needs.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import requests
import tiktoken
from cachetools import TTLCache, cached
from pydantic import BaseModel, Field

DEFAULT_MODEL_PRICING = {
    "anthropic/claude-3-haiku": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-haiku-20240307": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },

    "anthropic/claude-3-opus": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-opus-20240229": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },

    "anthropic/claude-3-sonnet": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-sonnet-20240229": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },

    "anthropic/claude-3.5-haiku": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-5-haiku-20241022": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-5-haiku-latest": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic/claude-3.5-sonnet": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-5-sonnet-20240620": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-5-sonnet-20241022": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-5-sonnet-latest": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic/claude-3.7-sonnet": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-7-sonnet-20250219": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "anthropic_integration_chat.claude-3-7-sonnet-latest": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "chatgpt-4o-latest": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0125": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-1106": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-instruct": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-instruct-0914": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-4": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-0125-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-0613": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-1106-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-turbo-2024-04-09": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-turbo-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-2024-05-13": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-2024-08-06": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-2024-11-20": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-audio-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-audio-preview-2024-10-01": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-audio-preview-2024-12-17": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini-2024-07-18": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini-audio-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini-audio-preview-2024-12-17": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini-realtime-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-mini-realtime-preview-2024-12-17": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-realtime-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-realtime-preview-2024-10-01": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4o-realtime-preview-2024-12-17": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1-2024-12-17": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1-mini": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1-mini-2024-09-12": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1-preview": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o1-preview-2024-09-12": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o3-mini": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "o3-mini-2025-01-31": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "omni-moderation-2024-09-26": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "omni-moderation-latest": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "deepseek/deepseek-chat": {
        "input_cost_per_token": 0.00000055,
        "output_cost_per_token": 0.00000219,
    },
    "deepseek_intergration_chat.deepseek-chat": {
        "input_cost_per_token": 0.00000055,
        "output_cost_per_token": 0.00000219,
    },
    "deepseek/deepseek-reasoner": {
        "input_cost_per_token": 0.00000055,
        "output_cost_per_token": 0.00000219,
    },
    "deepseek_intergration_chat.deepseek-reasoner": {
        "input_cost_per_token": 0.00000055,
        "output_cost_per_token": 0.00000219,
    },
}


class Config:
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, ".cache")
    USER_COST_FILE = os.path.join(DATA_DIR, f"costs-{datetime.now().year}.json")

    # If you want to fetch from a remote URL, specify it here
    REMOTE_JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

    CACHE_TTL = 432000  # e.g., 5 days
    CACHE_MAXSIZE = 8
    DECIMALS = "0.00000001"
    DEBUG = False

def debug_print(msg: str):
    if Config.DEBUG:
        print("[COST TRACKER DEBUG] " + msg)

def get_encoding_for_model(model_name: str):
    """
    Safely get a tiktoken encoding for the given model_name,
    falling back to 'cl100k_base' if unknown.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        debug_print(f"Unknown encoding for model={model_name}, using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")


class UserCostManager:
    def __init__(self, cost_file_path: str):
        self.cost_file_path = cost_file_path
        self._ensure_cost_file_exists()

    def _ensure_cost_file_exists(self):
        if not os.path.exists(self.cost_file_path):
            with open(self.cost_file_path, "w", encoding="UTF-8") as f:
                json.dump({}, f)

    def _read_costs(self) -> dict:
        with open(self.cost_file_path, "r", encoding="UTF-8") as f:
            return json.load(f)

    def _write_costs(self, costs: dict):
        with open(self.cost_file_path, "w", encoding="UTF-8") as f:
            json.dump(costs, f, indent=2)

    def update_user_cost(
        self,
        user_email: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_cost: Decimal,
    ):
        """
        Append a cost record for the given user & model to the JSON file.
        """
        costs_data = self._read_costs()
        if user_email not in costs_data:
            costs_data[user_email] = []

        costs_data[user_email].append({
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": str(total_cost)
        })
        self._write_costs(costs_data)

cache = TTLCache(maxsize=Config.CACHE_MAXSIZE, ttl=Config.CACHE_TTL)


class ModelCostManager:
    _lock = Lock()

    def __init__(self, remote_url: str, fallback_dict: dict):
        self.remote_url = remote_url
        self.fallback_dict = fallback_dict
        self.cached_data = None  # Will store the remote JSON data (if downloaded)
        self.cache_file = os.path.join(Config.CACHE_DIR, "model_prices.json")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    @cached(cache=cache)
    def fetch_remote_data(self) -> dict:
        """
        Attempt to download remote JSON. If that fails, return empty dict.
        """
        if not self.remote_url:
            return {}
        try:
            debug_print(f"Attempting to fetch remote cost data from {self.remote_url}")
            resp = requests.get(self.remote_url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            debug_print(f"Failed to fetch remote data: {e}")
            return {}

    def get_model_data(self, model: str) -> dict:
        """
        Return the cost data for a specific model. 
        1) Try remote data 
        2) If not found, fallback to DEFAULT_MODEL_PRICING.
        """
        # 1) Attempt to load remote data
        if not self.cached_data:
            with self._lock:
                if not self.cached_data:
                    self.cached_data = self.fetch_remote_data()

        # If remote data is found and has the model
        if self.cached_data and model in self.cached_data:
            return self.cached_data[model]

        # 2) Otherwise fallback
        if model in self.fallback_dict:
            return self.fallback_dict[model]

        debug_print(f"No cost data found for '{model}'. Using zero cost.")
        return {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }


class CostCalculator:
    def __init__(self, user_cost_manager: UserCostManager, model_cost_manager: ModelCostManager):
        self.user_cost_manager = user_cost_manager
        self.model_cost_manager = model_cost_manager

    def calculate_costs(
        self, model: str, input_tokens: int, output_tokens: int, compensation: float
    ) -> Decimal:
        """
        Calculate the cost of input_tokens + output_tokens for the given model.
        Use 'compensation' as a multiplier (e.g. 1.2 for 20% markup).
        """
        cost_data = self.model_cost_manager.get_model_data(model)
        in_cpt = Decimal(str(cost_data.get("input_cost_per_token", 0)))
        out_cpt = Decimal(str(cost_data.get("output_cost_per_token", 0)))

        input_cost = Decimal(input_tokens) * in_cpt
        output_cost = Decimal(output_tokens) * out_cpt
        raw_total = input_cost + output_cost
        final_cost = raw_total * Decimal(compensation)

        # Round to the nearest DECIMALS
        return final_cost.quantize(Decimal(Config.DECIMALS), rounding=ROUND_HALF_UP)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=15)
        compensation: float = Field(
            default=1.0, description="Price multiplier"
        )
        show_elapsed_time: bool = True
        show_tokens: bool = True
        show_tokens_per_second: bool = True
        debug: bool = False

    def __init__(self):
        self.valves = self.Valves()
        Config.DEBUG = self.valves.debug

        self.user_cost_manager = UserCostManager(Config.USER_COST_FILE)
        self.model_cost_manager = ModelCostManager(
            remote_url=Config.REMOTE_JSON_URL,
            fallback_dict=DEFAULT_MODEL_PRICING
        )
        self.cost_calculator = CostCalculator(self.user_cost_manager, self.model_cost_manager)

        self.input_tokens = 0
        self.start_time = None

    def _remove_roles(self, text: str) -> str:
        """
        Remove lines that begin with 'SYSTEM:', 'USER:', 'ASSISTANT:', or 'PROMPT:'.
        """
        roles = ("SYSTEM:", "USER:", "ASSISTANT:", "PROMPT:")
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            # If line starts with one of those roles, remove it
            if any(line.startswith(r) for r in roles):
                cleaned.append(line.split(":", 1)[1].strip())
            else:
                cleaned.append(line)
        return "\n".join(cleaned).strip()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Called before the main generation step.
        - Count input tokens
        - Possibly store user email
        - Mark start_time
        """
        # Let's parse messages from body
        messages = body.get("messages", [])
        content_str = "\n".join([m.get("content", "") for m in messages])
        cleaned_text = self._remove_roles(content_str)

        enc = get_encoding_for_model(body.get("model", "unknown-model"))
        self.input_tokens = len(enc.encode(cleaned_text))

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {
                "description": f"Input tokens: {self.input_tokens}",
                "done": False
            }})

        # If there's user info
        if __user__ and "email" in __user__:
            body["user"] = __user__["email"]

        self.start_time = time.time()
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Called after the generation step.
        - Count output tokens
        - Compute cost
        - Save cost
        - Emit stats
        """
        end_time = time.time()
        elapsed = end_time - self.start_time

        # Count output tokens
        messages = body.get("messages", [])
        last_msg_content = messages[-1].get("content", "") if messages else ""
        enc = get_encoding_for_model(body.get("model", "unknown-model"))
        output_tokens = len(enc.encode(last_msg_content))

        # Compute cost
        model_name = body.get("model", "unknown-model")
        total_cost = self.cost_calculator.calculate_costs(
            model=model_name,
            input_tokens=self.input_tokens,
            output_tokens=output_tokens,
            compensation=self.valves.compensation
        )

        # Save cost
        user_email = None
        if __user__:
            user_email = __user__.get("email")
        elif "user" in body:
            user_email = body["user"]

        if user_email:
            try:
                self.user_cost_manager.update_user_cost(
                    user_email,
                    model_name,
                    self.input_tokens,
                    output_tokens,
                    total_cost
                )
            except Exception as e:
                debug_print(f"Error updating user cost: {e}")

        # Prepare stats
        total_tokens = self.input_tokens + output_tokens
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        stats_list = []
        if self.valves.show_elapsed_time:
            stats_list.append(f"{elapsed:.2f} s")
        if self.valves.show_tokens_per_second:
            stats_list.append(f"{tokens_per_sec:.2f} T/s")
        if self.valves.show_tokens:
            stats_list.append(f"{total_tokens} Tokens")

        # format cost
        cost_str = (
            f"${total_cost:.2f}"
            if float(total_cost) < float(Config.DECIMALS)
            else f"${total_cost:.6f}"
        )
        stats_list.append(cost_str)

        stats_string = " | ".join(stats_list)

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {
                "description": stats_string,
                "done": True
            }})

        return body
