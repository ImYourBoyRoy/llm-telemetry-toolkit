# llm-telemetry-toolkit/verify_toolkit.py
import sys
import shutil
import time
from pathlib import Path

# Fix path to include src/
sys.path.append(str(Path(__file__).parent / "src"))

from llm_telemetry_toolkit import (
    LLMLogger,
    TelemetryConfig,
    SessionContext,
    monitor_interaction,
    LLMInteraction,
)


def main():
    print("Verifying ADVANCED Toolkit...")

    # Clean prev output
    out = Path("test_output")
    try:
        if out.exists():
            shutil.rmtree(out)
    except Exception:
        pass

    # 1. Advanced Config
    config = TelemetryConfig(
        session_id="adv_session_001",
        base_log_dir=out,
        enable_entity_logging=True,
        output_formats=["json", "md", "csv"],  # <--- Multiple formats
        max_content_length=50,  # <--- Truncation test (Lowered to force truncation)
        mask_pii=True,  # <--- Enable PII Redaction
    )
    print("! Config Initialized")

    # 2. Async Logger
    logger = LLMLogger(config)
    print("! Logger Initialized (Async)")

    # 2. Context Manager Implicit Session
    with SessionContext("context_session_001"):
        print("! Inside Context")

        # PII Test Prompt
        pii_prompt = (
            "My email is roy@example.com, IP is 192.168.1.1, phone +1 (555) 123-4567. "
            "Card: 4111 2222 3333 4444"
        )

        interaction = LLMInteraction(
            session_id="context_session_001",  # Explicitly match our context for this manual test
            model_name="test-model-v1",
            prompt=pii_prompt,
            response="I will keep your data safe, <think>Parsing ID...</think> don't worry.",
            response_time_seconds=0.5,
            token_count_prompt=10,
            token_count_response=20,
            cost_usd=0.002,
            entity_label="CompA",  # <--- Entity routing
        )
        logger.log(interaction)  # Log the PII interaction
        print("! PII Interaction logged.")

    # 3. Test Monitor Decorator & Implicit Context
    @monitor_interaction(logger, interaction_type="decorated_test")
    def expensive_func(name):
        time.sleep(0.1)
        return f"Response for {name}" * 10

    with SessionContext("context_session_001"):
        print("! Inside Context Manager")
        result = expensive_func("User")
        print(f"! Function returned: {result[:20]}...")

    # 4. Wait for Async Write (since we can't await it smoothly here without a queue check, just sleep)
    print("! Waiting for background thread...")
    time.sleep(1.0)

    # 5. Verify Files
    # Expecting context_session_001 directory because the context manager overrides the config default
    session_dir = out / "llm_interactions" / "context_session_001"

    if not session_dir.exists():
        print(f"X Session dir missing: {session_dir}")
        # Fallback check
        print(f"  Existing dirs: {list(out.glob('**/*'))}")
        sys.exit(1)

    # Check for JSON, MD, CSV
    files = list(session_dir.glob("*"))
    print(f"! Found {len(files)} files in output.")

    extensions = {f.suffix for f in files}
    if {".json", ".md", ".csv"}.issubset(extensions):
        print("! All formats (json, md, csv) generated successfully.")

    # Check for session configuration file
    config_file = session_dir / "session_config.json"
    if config_file.exists():
        print("! Session config file created successfully.")
    else:
        print("X Session config file missing.")

    # Check Truncation in JSON
    json_file = next(f for f in files if f.suffix == ".json")
    with open(json_file, "r") as f:
        content = f.read()
        if "TRUNCATED" in content:
            print("! Truncation Logic verified.")
        else:
            print(f"X Truncation failed in {json_file}")

    # 6. Show Rich Dashboard
    print("\n--- Displaying Rich Dashboard for 'context_session_001' ---")
    import subprocess

    cmd = [
        sys.executable,
        "-W",
        "ignore::RuntimeWarning",  # Suppress runpy warnings
        "-m",
        "llm_telemetry_toolkit.interface.cli",
        "view",
        "--session",
        "context_session_001",
        "--dir",
        str(out),  # Use the test output dir
    ]

    # Need to pass PYTHONPATH to subprocess so it finds the src package
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")

    subprocess.run(cmd, env=env)

    print("\n! Verification Complete!")
    logger.shutdown()


if __name__ == "__main__":
    main()
