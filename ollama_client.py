# ./llm-telemetry-toolkit/ollama_client.py
"""
Ollama API Client.
Provides a simple wrapper for interacting with a local Ollama server,
extracting metadata, and ensuring consistent telemetry hooks.

Inputs: Base URL, Model Name, Prompts.
Outputs: Generated text and raw API metadata.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(
        self, model: str, prompt: str, system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a completion from Ollama.
        Returns the raw JSON response containing 'response', 'context', and stats.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(
                req, timeout=300
            ) as response:  # High timeout for large models
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"error": str(e)}

    def show_model_info(self, model: str) -> Dict[str, Any]:
        """Fetches model details (modelfile, details, etc)."""
        url = f"{self.base_url}/api/show"
        payload = {"name": model}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}

    def check_connection(self) -> bool:
        """Pings the Ollama server to verify reachability."""
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5)
            return True
        except Exception:
            return False
