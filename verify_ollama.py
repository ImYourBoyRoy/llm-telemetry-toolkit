# ./llm-telemetry-toolkit/verify_ollama.py
"""
Ollama Integration Verification Script.
Connects to a real Ollama server, performs inference, and logs detailed telemetry.
Target Host: 192.168.1.21
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_telemetry_toolkit import (
    LLMLogger,
    TelemetryConfig,
    LLMInteraction,
)
from ollama_client import OllamaClient

# Configuration
OLLAMA_HOST = "http://192.168.1.21:11434"
TARGET_MODELS = [
    "tinyllama:latest",
    "gpt-oss:20b",
    "ministral-3:latest",
    "mistral-small3.2:latest",
    "huihui_ai/orchestrator-abliterated:latest",
    "devstral:latest",
]


def main():
    print(f"--- Starting Ollama Verification against {OLLAMA_HOST} ---", flush=True)

    client = OllamaClient(OLLAMA_HOST)

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

    # 2. Iterate Models
    for model in TARGET_MODELS:
        print(f"\n[Testing Model]: {model}", flush=True)

        # A. Fetch Info
        print("  > Fetching metadata...", flush=True)
        model_info = client.show_model_info(model)
        details = model_info.get("details", {})

        # B. Run Inference
        prompt = "Explain why the sky is blue in one concise sentence."
        print(f"  > Sending prompt: '{prompt}'", flush=True)

        start_t = time.time()
        result = client.generate(model, prompt)
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

        response_text = result.get("response", "")
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
            metadata={
                "ollama_details": details,
                "ollama_stats": {
                    "total_duration_ns": total_duration_ns,
                    "load_duration_ns": result.get("load_duration", 0),
                    "model_family": details.get("family", "unknown"),
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
