"""
title: Anthropic Integration Chat and Image
authors: brammittendorff
author_url: https://github.com/brammittendorff/openwebui-pipelines
funding_url: https://github.com/open-webui
version: 0.0.1
required_open_webui_version: 0.3.20
license: MIT
environment_variables:
    - ANTHROPIC_API_KEY (required)
"""
import os
import requests
import json
import time
import logging
from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    # Constants
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    MAX_TOTAL_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total limit
    REQUEST_TIMEOUT = (3.05, 60)
    STREAM_DELAY = 0.01
    
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Anthropic API Key"
        )
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", "")
        )
    
    def get_anthropic_models(self) -> List[Dict[str, str]]:
        """Return available Anthropic models."""
        return [
            {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "claude-3.5-haiku"},
            {"id": "claude-3-5-haiku-latest", "name": "claude-3.5-haiku"},
            {"id": "claude-3-5-sonnet-20240620", "name": "claude-3.5-sonnet"},
            {"id": "claude-3-5-sonnet-20241022", "name": "claude-3.5-sonnet"},
            {"id": "claude-3-5-sonnet-latest", "name": "claude-3.5-sonnet"},
            {"id": "claude-3-7-sonnet-20250219", "name": "claude-3.7-sonnet"},
            {"id": "claude-3-7-sonnet-latest", "name": "claude-3.7-sonnet"},
        ]
    
    def pipes(self) -> List[Dict[str, str]]:
        """Return available models for the pipe."""
        return self.get_anthropic_models()
    
    def process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image data with size validation.
        
        Args:
            image_data: Image data from the request
            
        Returns:
            Processed image data in Anthropic format
            
        Raises:
            ValueError: If image size exceeds limits
        """
        if image_data["image_url"]["url"].startswith("data:image"):
            # Handle base64 encoded images
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            
            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )
                
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # Handle URL images
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            
            if response.status_code != 200:
                raise ValueError(f"Failed to access image URL: {url}, status code: {response.status_code}")
                
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                )
                
            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }
    
    def prepare_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Prepare messages for Anthropic API format.
        
        Args:
            messages: List of message objects
            
        Returns:
            Tuple of (processed_messages, total_image_size)
        """
        processed_messages = []
        total_image_size = 0
        
        for message in messages:
            processed_content = []
            
            # Handle content that is a list (multimodal)
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)
                        
                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            
                            if total_image_size > self.MAX_TOTAL_IMAGE_SIZE:
                                raise ValueError("Total size of images exceeds 100 MB limit")
            else:
                # Handle simple text content
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]
                
            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
            
        return processed_messages, total_image_size
    
    def get_headers(self) -> Dict[str, str]:
        """Return headers for Anthropic API requests."""
        return {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
    
    def pipe(self, body: Dict[str, Any]) -> Union[str, Generator, Iterator]:
        """
        Process a request through the Anthropic API.
        
        Args:
            body: Request body containing messages and parameters
            
        Returns:
            Response from Anthropic API, either streamed or complete
        """
        if not self.valves.ANTHROPIC_API_KEY:
            return "Error: ANTHROPIC_API_KEY is required"
            
        try:
            # Extract system message and process messages
            system_message, messages = pop_system_message(body["messages"])
            processed_messages, _ = self.prepare_messages(messages)
            
            # Extract model name from full model ID
            model_name = body["model"][body["model"].find(".") + 1:]
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
                "stream": body.get("stream", False),
            }
            
            # Add system message if present
            if system_message:
                payload["system"] = str(system_message)
                
            headers = self.get_headers()
            
            # Choose streaming or non-streaming response
            if body.get("stream", False):
                return self.stream_response(self.API_URL, headers, payload)
            else:
                return self.non_stream_response(self.API_URL, headers, payload)
                
        except ValueError as e:
            logging.error(f"Validation error: {e}")
            return f"Error: {e}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            return f"Error: {e}"
    
    def stream_response(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Stream response from Anthropic API.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Yields:
            Text chunks from the response
        """
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=self.REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                        
                    try:
                        data = json.loads(line[6:])
                        
                        if data["type"] == "content_block_start":
                            yield data["content_block"]["text"]
                        elif data["type"] == "content_block_delta":
                            yield data["delta"]["text"]
                        elif data["type"] == "message_stop":
                            break
                        elif data["type"] == "message":
                            for content in data.get("content", []):
                                if content["type"] == "text":
                                    yield content["text"]
                                    
                        # Small delay to avoid overwhelming the client
                        time.sleep(self.STREAM_DELAY)
                        
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse JSON: {line}")
                    except KeyError as e:
                        logging.error(f"Unexpected data structure: {e}, data: {data}")
                        
        except requests.exceptions.RequestException as e:
            logging.error(f"Stream request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            logging.error(f"Error in stream_response: {e}", exc_info=True)
            yield f"Error: {e}"
    
    def non_stream_response(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
        """
        Get complete response from Anthropic API.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Returns:
            Complete text response
        """
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            res = response.json()
            return res["content"][0]["text"] if "content" in res and res["content"] else ""
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Non-stream request failed: {e}")
            return f"Error: {e}"
        except (KeyError, IndexError) as e:
            logging.error(f"Unexpected response format: {e}, response: {response.text}")
            return f"Error: Unexpected response format: {e}"
