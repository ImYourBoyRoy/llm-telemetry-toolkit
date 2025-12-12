# 🤖 LLM Telemetry Toolkit

[![PyPI Version](https://img.shields.io/pypi/v/llm-telemetry-toolkit.svg)](https://pypi.org/project/llm-telemetry-toolkit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-telemetry-toolkit.svg)](https://pypi.org/project/llm-telemetry-toolkit/)
[![License](https://img.shields.io/pypi/l/llm-telemetry-toolkit.svg)](https://opensource.org/licenses/MIT)
[![Typing: PEP 561](https://img.shields.io/badge/Typing-PEP%20561-informational.svg)](https://peps.python.org/pep-0561/)

> A standardized, **async-first** observability system for LLM Agents. Zero-latency logging, beautiful metrics, and deep telemetry for production AI systems.

---

## ✨ Design Goals

*   **Zero Latency Impact**
    Logging happens in a background thread via `queue.Queue`. Your Agent's "Process Loop" is never blocked by file I/O.

*   **"Masterpiece" Developer Experience**
    From beautiful Rich-powered CLI dashboards to zero-friction `@monitor_interaction` decorators, every interaction feels expertly crafted.

*   **Smart Parsing**
    Automatically detects reasoning traces (`<think>` tags) and segregates them from the final response. Handles PII redaction intelligently.

*   **Implicit Context**
    Uses `contextvars` to track sessions across deep function stacks without passing `session_id` arguments everywhere.

---

## ⭐ Features

*   **Multi-Format Logging**: Generate `JSON` dossiers, human-readable `Markdown` reports, and `CSV` datasets simultaneously.
*   **Rich CLI Dashboard**: Visualize your agent's performance with a beautiful terminal UI (`view` and `stats` commands).
*   **Smart PII Redaction**: "Smart Masking" for Emails (`r***@example.com`), IPs, Phones, and Credit Cards. (**Opt-in**)
*   **Ollama Native Support**: Captures deep metadata (load duration, eval count, quantization) from local Ollama providers.
*   **Robust Schema**: Built on Pydantic v2 for strict validation and typing.

---

## 🚀 Installation

```bash
pip install llm-telemetry-toolkit
# Ensure you have 'rich' for the CLI experience (installed by default)
```

---

## 🧪 Quick Start

The toolkit uses "Smart Defaults" so you can start logging with zero configuration.

```python
from llm_telemetry_toolkit import LLMLogger, TelemetryConfig, monitor_interaction

# 1. Zero-Config initialization
# - session_id auto-generated (session_YYYYMMDD_HHMMSS)
# - logs written to ./logs
config = TelemetryConfig() 

logger = LLMLogger(config)

# 2. Decorate your LLM function
@monitor_interaction(logger)
def chat_completion(prompt: str) -> str:
    # Simulate LLM call
    return "The sky is blue because of Rayleigh scattering."

# 3. Run!
print(chat_completion("Why is the sky blue?"))
# -> Logs are automatically written to ./logs/llm_interactions/session_.../
```

---

## 🛠️ Command-Line Interface (CLI)

The toolkit comes with a "sexy" CLI for viewing logs and calculating costs.

### View Logs
Displays color-coded panels with syntax highlighting for prompts and responses.

```bash
python -m llm_telemetry_toolkit.interface.cli view --session ollama_test_004910
```

### Calculate Stats
Aggregates token counts, latency, and estimated cost.

```bash
python -m llm_telemetry_toolkit.interface.cli stats --session ollama_test_004910
```

**Example Output:**
```text
┌──────────────────── Session Statistics: ollama_test_004910 ────────────────────┐
│ Metric               │ Value                                                │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Total Interactions   │ 7                                                    │
│ Total Latency        │ 65.57s                                               │
│ Avg Latency          │ 9.37s                                                │
│ Total Cost           │ $0.00000                                             │
│ Total Tokens         │ 3268                                                 │
│  - Prompt            │ 2454                                                 │
│  - Response          │ 814                                                  │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

The `TelemetryConfig` object controls all behavior.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `session_id` | `str` | `session_Timestamp` | Unique identifier for the logging session. |
| `base_log_dir` | `Path` | `./logs` | Root directory for storing logs. |
| `enable_session_logging` | `bool` | `True` | Group logs by session ID. |
| `enable_entity_logging` | `bool` | `False` | Fork logs into entity folders (e.g., 'Company A'). |
| `output_formats` | `List[str]` | `['json']` | Supported: `'json'`, `'md'`, `'csv'`. |
| `mask_pii` | `bool` | `False` | Enable regex-based PII redaction. (**Opt-in**) |
| `max_content_length` | `int` | `None` | Truncate prompts/responses longer than this chars. |
| `json_indent` | `int` | `2` | Indentation for JSON files. |

---

## 🤖 Agents & MCP Server Integration

This toolkit is designed to be the "Telemetry Layer" for Agentic frameworks or MCP Servers.

### 1. Manual Logging (for Traceability)
If you can't use decorators, use the manual API.

```python
from llm_telemetry_toolkit import LLMInteraction

# ... inside your MCP tool execution ...
interaction = LLMInteraction(
    session_id="agent_run_x",
    model_name="deepseek:r1",
    prompt=tool_input,
    response=tool_output,
    response_time_seconds=latency,
    # Capture Deep Metadata from Ollama/Providers
    metadata={
        "ollama_stats": {
            "total_duration_ns": 1500000000,
            "load_duration_ns": 500000000,
            "prompt_eval_count": 45,
            "eval_count": 120
        },
        "tool_name": "web_scraper",
        "url": "example.com"
    }
)

# Non-blocking write
logger.log(interaction)
```

### 2. Context Management
Use `SessionContext` to implicitly pass session IDs through your call stack.

```python
from llm_telemetry_toolkit import SessionContext

def my_deeply_nested_function():
    # Automatically picks up "user_session_abc" from context
    logger.log(interaction) 

with SessionContext("user_session_abc"):
    my_deeply_nested_function()
```

---

## 🔎 Smart Parsing & PII

### reasoning-trace extraction (<think>)
If your model (e.g., DeepSeek R1, Groq) returns `<think>` blocks, the toolkit automatically extracts them into a separate `thought_process` field in the JSON log, keeping the `response` clean.

### Smart Masking (PII)
Enable via `TelemetryConfig(mask_pii=True)`. This is **Opt-in** and disabled by default.
When enabled, sensitive data is redacted while preserving debug context:
*   Email: `roy@example.com` -> `r***@example.com`
*   IP: `192.168.1.1` -> `192.xxx.xxx.xxx`
*   Credit Card: `4111 2222 3333 4444` -> `4111 xxxx xxxx 4444`
*   Phone: `+1 (555) 123-4567` -> `+1 (555) ***-**67`

---

## 📦 Example Log Output (Ollama)

```json
{
  "interaction_id": "ollama_test_004910_llm_0003",
  "session_id": "ollama_test_004910",
  "timestamp_utc": "2025-12-12T07:49:34.550928+00:00",
  "model_name": "ministral-3:latest",
  "provider": "ollama_local",
  "prompt": "Explain why the sky is blue in one concise sentence.",
  "response": "The sky appears blue because Earth's atmosphere scatters shorter (blue) wavelengths of sunlight more than other colors due to a phenomenon called **Rayleigh scattering**.",
  "response_time_seconds": 7.78,
  "token_count_prompt": 551,
  "token_count_response": 33,
  "metadata": {
    "ollama_details": {
      "format": "gguf",
      "family": "mistral3",
      "parameter_size": "8.9B",
      "quantization_level": "Q4_K_M"
    },
    "ollama_stats": {
      "total_duration_ns": 7757972341,
      "load_duration_ns": 6139982699,
      "eval_count": 33
    }
  }
}
```

---

## 📦 License

MIT. Created by **Roy Dawson IV**.

*   GitHub: [https://github.com/imyourboyroy](https://github.com/imyourboyroy)
*   PyPi: [https://pypi.org/user/ImYourBoyRoy/](https://pypi.org/user/ImYourBoyRoy/)
