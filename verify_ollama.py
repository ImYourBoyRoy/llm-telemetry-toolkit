# ./verify_ollama.py
"""
Run end-to-end verification against a live Ollama host with telemetry capture.
Used for manual confidence checks across multiple local/remote Ollama models.
Run: `python verify_ollama.py` from project root with reachable Ollama endpoint.
Inputs: Environment settings and optional local `settings.cfg` values plus test prompts.
Outputs: Telemetry log files under `ollama_telemetry_logs` and console status lines.
Side effects: Performs model inference requests and writes session log artifacts.
Operational notes: Intended for manual validation, not deterministic CI execution.
"""

import sys
import time
import os
from configparser import ConfigParser
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_telemetry_toolkit import (
    LLMLogger,
    TelemetryConfig,
    LLMInteraction,
)
from llm_telemetry_toolkit.providers import OllamaClient


def _load_local_ollama_settings() -> dict[str, str]:
    """Load private local settings from ./settings.cfg when present."""
    settings_path = Path("settings.cfg")
    if not settings_path.exists():
        return {}

    parser = ConfigParser()
    parser.read(settings_path, encoding="utf-8")
    if not parser.has_section("ollama"):
        return {}

    return {
        key.strip(): value.strip()
        for key, value in parser.items("ollama")
        if key and value
    }


def _to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "no", "off", ""}


LOCAL_SETTINGS = _load_local_ollama_settings()

# Configuration
OLLAMA_HOST = os.getenv(
    "OLLAMA_BASE_URL", LOCAL_SETTINGS.get("base_url", "http://localhost:11434")
).strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", LOCAL_SETTINGS.get("api_key", "")).strip()
DEFAULT_MODELS = [
    "tinyllama:latest",
    "mistral-small3.2:latest",
    "qwen3:8b",
]
configured_models = os.getenv("OLLAMA_VERIFY_MODELS", "").strip() or LOCAL_SETTINGS.get(
    "verify_models", ""
)
TARGET_MODELS = [
    model.strip()
    for model in (configured_models or ",".join(DEFAULT_MODELS)).split(",")
    if model.strip()
]
USE_CHAT_API = _to_bool(
    os.getenv("OLLAMA_VERIFY_USE_CHAT", LOCAL_SETTINGS.get("use_chat", "1"))
)


def main():
    print(f"--- Starting Ollama Verification against {OLLAMA_HOST} ---", flush=True)

    client = OllamaClient(OLLAMA_HOST, api_key=OLLAMA_API_KEY or None)

    # 0. Connection Check

    print("  > Checking connection...", end=" ", flush=True)
    if client.check_connection():
        print("OK", flush=True)
    else:
        print("FAIL: Could not connect to Ollama", flush=True)
        return

    # 1. Setup Telemetry
    config = TelemetryConfig(
        session_id=f"ollama_test_{datetime.now().strftime('%H%M%S')}",
        base_log_dir=Path("ollama_telemetry_logs"),
        enable_entity_logging=False,
        output_formats=["json", "md"],
        max_content_length=None,
    )
    logger = LLMLogger(config)
    print(f"Logs will be written to: {config.base_log_dir}", flush=True)

    # 2. Resolve installed models and iterate targets
    installed_payload = client.list_models()
    installed_models = {
        str(item.get("name"))
        for item in installed_payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }
    if not installed_models:
        print("  ! Could not retrieve installed model list from /api/tags.", flush=True)

    for model in TARGET_MODELS:
        if installed_models and model not in installed_models:
            print(f"\n[Skipping Model - not installed]: {model}", flush=True)
            continue

        print(f"\n[Testing Model]: {model}", flush=True)

        # A. Fetch Info
        print("  > Fetching metadata...", flush=True)
        model_info = client.show_model_info(model)
        details = model_info.get("details", {})

        # B. Run Inference
        prompt = "Explain why the sky is blue in one concise sentence."
        print(f"  > Sending prompt: '{prompt}'", flush=True)

        start_t = time.time()
        if USE_CHAT_API:
            result = client.chat(
                model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.2},
                think=False,
            )
            if "error" in result:
                print("  ! /api/chat failed; falling back to /api/generate", flush=True)
                result = client.generate(
                    model,
                    prompt,
                    stream=False,
                    options={"temperature": 0.2},
                    think=False,
                )
        else:
            result = client.generate(
                model, prompt, stream=False, options={"temperature": 0.2}, think=False
            )
        latency = time.time() - start_t

        if "error" in result:
            print(f"  X Error: {result['error']}", flush=True)
            # Log failure
            interaction = LLMInteraction(
                session_id=config.session_id,
                model_name=model,
                prompt=prompt,
                response="ERROR",
                response_time_seconds=latency,
                error_message=result["error"],
                validation_passed=False,
            )
            logger.log(interaction)
            continue

        if "message" in result and isinstance(result.get("message"), dict):
            response_text = str(result["message"].get("content", ""))
            response_role = str(result["message"].get("role", "assistant"))
            tool_calls = result["message"].get("tool_calls", [])
        else:
            response_text = str(result.get("response", ""))
            response_role = "assistant"
            tool_calls = []
        print(
            f"  > Response received ({len(response_text)} chars). Time: {latency:.2f}s",
            flush=True,
        )

        # C. Telemetry Logging
        # Extract native Ollama stats
        eval_count = result.get("eval_count", 0)
        prompt_eval_count = result.get("prompt_eval_count", 0)
        total_duration_ns = result.get("total_duration", 0)

        interaction = LLMInteraction(
            session_id=config.session_id,
            model_name=model,
            provider="ollama_local",
            prompt=prompt,
            response=response_text,
            response_time_seconds=latency,
            token_count_response=eval_count,
            token_count_prompt=prompt_eval_count,
            interaction_type="verification_test",
            request_messages=[{"role": "user", "content": prompt}]
            if USE_CHAT_API
            else None,
            response_message_role=response_role,
            finish_reason=result.get("done_reason"),
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
            metadata={
                "ollama_details": details,
                "ollama_stats": {
                    "total_duration_ns": total_duration_ns,
                    "load_duration_ns": result.get("load_duration", 0),
                    "model_family": details.get("family", "unknown"),
                },
                "ollama_response": {
                    "done": result.get("done"),
                },
            },
        )

        log_result = logger.log(interaction)
        if log_result.success:
            print(
                f"  > Telemetry logged interaction {log_result.interaction_id}",
                flush=True,
            )
            if interaction.thought_process:
                print("  ! <think> block detected and extracted!", flush=True)

    print("\nFinishing background writes...", flush=True)
    logger.shutdown()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
