# LLM Telemetry Toolkit

[![PyPI Version](https://img.shields.io/pypi/v/llm-telemetry-toolkit.svg)](https://pypi.org/project/llm-telemetry-toolkit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-telemetry-toolkit.svg)](https://pypi.org/project/llm-telemetry-toolkit/)
[![License](https://img.shields.io/pypi/l/llm-telemetry-toolkit.svg)](https://opensource.org/licenses/MIT)
[![Typing: PEP 561](https://img.shields.io/badge/Typing-PEP%20561-informational.svg)](https://peps.python.org/pep-0561/)

## Use Case Synopsis

`llm-telemetry-toolkit` is a production-focused telemetry layer for LLM agents and tools.  
It captures structured interactions with async queue writes (or true sync writes when needed), sanitizes content, and emits JSON/Markdown/CSV logs for local debugging, analytics, and release-grade observability.

---

## Installation

```bash
pip install llm-telemetry-toolkit
```

Optional OpenTelemetry export support:

```bash
pip install "llm-telemetry-toolkit[otel]"
```

---

## Integration Guide

### 1) Standalone usage

```python
from llm_telemetry_toolkit import LLMInteraction, LLMLogger, TelemetryConfig

config = TelemetryConfig(
    session_id="demo_session",
    output_formats=["json", "md", "csv"],
    ensure_ascii=False,
)
logger = LLMLogger(config)

result = logger.log(
    LLMInteraction(
        session_id="demo_session",
        model_name="my-model",
        prompt="Explain Rayleigh scattering in one sentence.",
        response="The sky looks blue because shorter blue wavelengths scatter more in the atmosphere.",
        response_time_seconds=0.52,
    ),
    sync=True,
)

print(result.model_dump())
logger.shutdown()
```

### 2) Agentic / MCP usage

Use this package as your telemetry boundary around tool execution:

```python
from llm_telemetry_toolkit import LLMInteraction, LLMLogger, SessionContext, TelemetryConfig

logger = LLMLogger(TelemetryConfig(session_id="agent_run_001", mask_pii=True))

def run_tool(tool_name: str, tool_input: str, tool_output: str, latency: float) -> None:
    logger.log(
        LLMInteraction(
            session_id="agent_run_001",
            model_name="deepseek:r1",
            interaction_type="tool_call",
            tool_name=tool_name,
            prompt=tool_input,
            response=tool_output,
            response_time_seconds=latency,
            metadata={"trace_layer": "mcp_server"},
        )
    )

with SessionContext("user_session_abc"):
    run_tool("web_scraper", "https://example.com", "200 OK", 0.23)

logger.shutdown()
```

### 3) Ollama adapters (sync + async, package-native)

```python
import asyncio
from llm_telemetry_toolkit import AsyncOllamaClient, OllamaClient, OllamaTransportConfig

transport = OllamaTransportConfig(max_retries=2, backoff_initial_seconds=0.2)

sync_client = OllamaClient("http://localhost:11434", transport_config=transport)
print(sync_client.list_models())
sync_client.close()

async def run() -> None:
    async with AsyncOllamaClient("http://localhost:11434", transport_config=transport) as client:
        payload = await client.chat(
            model="tinyllama:latest",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        print(payload)

asyncio.run(run())
```

### 4) Ollama streaming chunk API (sync + async)

```python
import asyncio
from llm_telemetry_toolkit import AsyncOllamaClient, OllamaClient

with OllamaClient("http://localhost:11434") as client:
    for event in client.stream_generate("tinyllama:latest", "Count 1 to 3"):
        print(event)

async def run_stream() -> None:
    async with AsyncOllamaClient("http://localhost:11434") as client:
        async for event in client.stream_chat(
            model="tinyllama:latest",
            messages=[{"role": "user", "content": "Say hello in 3 words"}],
        ):
            print(event)

asyncio.run(run_stream())
```

### 5) Optional OpenTelemetry export

```python
from llm_telemetry_toolkit import LLMLogger, TelemetryConfig

logger = LLMLogger(
    TelemetryConfig(
        session_id="otel_demo",
        enable_otel_export=True,
        otel_tracer_name="research_tool",
        otel_span_name="llm.interaction",
        otel_service_name="llm-telemetry-toolkit",
        otel_auto_configure=True,
        otel_otlp_endpoint="http://localhost:4318/v1/traces",
        otel_sampler_ratio=1.0,
    )
)
```

> Note: OTel export is additive to file logging and uses `opentelemetry-api/sdk` + OTLP exporter when auto-configure is enabled.

---

## Commands & Arguments

### CLI entry points

- `llm-telemetry-toolkit ...`
- `python -m llm_telemetry_toolkit.interface.cli ...`

### Commands

#### `view`

Render recent interactions for a session.

```bash
llm-telemetry-toolkit view --session <SESSION_ID> [--dir ./logs] [--limit 5]
```

Arguments:

- `--session` (required): session id folder under `llm_interactions/`
- `--dir` (optional): base log directory (default: `./logs`)
- `--limit` (optional): number of latest interactions to display (default: `5`)

#### `stats`

Aggregate totals (latency, token counts, cost, model usage).

```bash
llm-telemetry-toolkit stats --session <SESSION_ID> [--dir ./logs]
```

Arguments:

- `--session` (required): session id folder under `llm_interactions/`
- `--dir` (optional): base log directory (default: `./logs`)

---

## Configuration (TelemetryConfig)

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `session_id` | `str` | `session_YYYYMMDD_HHMMSS` | Required to be non-empty; used in folder routing. |
| `base_log_dir` | `Path` | `./logs` | Root location for outputs. |
| `enable_session_logging` | `bool` | `True` | Enables primary session logs. |
| `enable_entity_logging` | `bool` | `False` | Enables entity-routed logs under `entity_log_subdir`. |
| `session_log_subdir` | `str` | `llm_interactions` | Session output folder name. |
| `entity_log_subdir` | `str` | `entity_llm_interactions` | Entity output folder name. |
| `output_formats` | `List[str]` | `["json"]` | Supported only: `json`, `md`, `csv`. |
| `filename_template` | `str` | `{timestamp}_{interaction_id}_{type}.{ext}` | Supports `{timestamp}`, `{interaction_id}`, `{type}`, `{ext}`, `{session_id}`, `{model_name}`, `{tool_name}`. |
| `json_indent` | `int` | `2` | Pretty indent for JSON payloads. |
| `ensure_ascii` | `bool` | `False` | Controls JSON unicode escaping. |
| `max_content_length` | `Optional[int]` | `None` | Truncates prompt/response/thought content. |
| `mask_pii` | `bool` | `False` | Redacts email/IP/phone/card patterns. |
| `enable_otel_export` | `bool` | `False` | Emits optional OTel spans in addition to file logs. |
| `otel_tracer_name` | `str` | `llm_telemetry_toolkit` | Tracer name used for emitted spans. |
| `otel_span_name` | `str` | `llm.interaction` | Span operation name for each interaction. |
| `otel_service_name` | `str` | `llm-telemetry-toolkit` | Service name attribute on spans. |
| `otel_auto_configure` | `bool` | `False` | Bootstraps OTel SDK provider/exporter when no provider is set. |
| `otel_exporter` | `str` | `otlp_http` | Auto-config exporter (`otlp_http` or `none`). |
| `otel_otlp_endpoint` | `str` | `http://localhost:4318/v1/traces` | OTLP/HTTP destination endpoint. |
| `otel_otlp_headers` | `Dict[str, str]` | `{}` | Optional headers (auth/tenant). |
| `otel_otlp_timeout_seconds` | `float` | `10.0` | OTLP export timeout in seconds. |
| `otel_enable_console_export` | `bool` | `False` | Emit spans to console in addition to OTLP. |
| `otel_sampler_ratio` | `float` | `1.0` | Trace sampling ratio for auto-configured SDK. |
| `otel_resource_attributes` | `Dict[str, str]` | `{}` | Extra resource attributes on spans. |

### Structured interaction fields (LLMInteraction)

In addition to free-form `metadata`, the schema now supports explicit optional fields for downstream consistency:

- Chat/tooling: `request_messages`, `response_message_role`, `finish_reason`, `tool_calls`
- Embeddings: `embedding_input_count`, `embedding_vector_count`, `embedding_dimensions`
- Typed helper models: `ChatMessage`, `ToolCall`, `ToolFunctionCall`

---

## Examples

### Decorator-based logging (sync + async)

```python
import asyncio
from llm_telemetry_toolkit import LLMLogger, TelemetryConfig, monitor_interaction

logger = LLMLogger(TelemetryConfig(session_id="decorator_demo"))

@monitor_interaction(logger, interaction_type="sync_call")
def multiply(x: int, y: int) -> int:
    return x * y

@monitor_interaction(logger, interaction_type="async_call")
async def async_multiply(x: int, y: int) -> int:
    await asyncio.sleep(0.05)
    return x * y

print(multiply(3, 4))
print(asyncio.run(async_multiply(5, 6)))
logger.shutdown()
```

### True sync writes (for strict confirmation paths)

```python
from llm_telemetry_toolkit import LLMInteraction, LLMLogger, TelemetryConfig

logger = LLMLogger(TelemetryConfig(session_id="sync_demo"))
result = logger.log(
    LLMInteraction(
        session_id="sync_demo",
        model_name="sync-model",
        prompt="hello",
        response="world",
        response_time_seconds=0.1,
    ),
    sync=True,
)
print(result.success, result.queued, result.write_confirmed, result.created_files)
logger.shutdown()
```

---

## Testing

### Standard local test run

```powershell
python -B -m pytest -q
```

### Optional live Ollama integration test (private/local only)

1. Copy `settings.example.cfg` to `settings.cfg`.
2. Update `settings.cfg` with your private local server values.
3. Run:

```powershell
$env:OLLAMA_INTEGRATION = "1"
python -m pytest tests/test_ollama_integration.py -q
```

---

## Packaging & Release Notes

- Versioning is VCS-driven via `hatch-vcs` and git tags (`vX.Y.Z`).
- Console script entry point is published as `llm-telemetry-toolkit`.
- PyPI publish workflow is **tag-triggered** (`push` on `v*` tags).
- Optional TestPyPI publish can be run manually via GitHub Actions workflow dispatch.

Recommended release sequence:

1. Dry run preflight:

```powershell
python .\tools\publish_release.py --project .\llm-telemetry-toolkit --version 0.2.0 --dry-run
```

2. Real release tag push:

```powershell
python .\tools\publish_release.py --project .\llm-telemetry-toolkit --version 0.2.0
```

3. (Optional) Manual TestPyPI publish from Actions UI:
   - Workflow: `Publish to PyPI`
   - Input: `publish_target = testpypi`

---

## Author & Links

Created by **Roy Dawson IV**

- GitHub: [https://github.com/imyourboyroy](https://github.com/imyourboyroy)
- PyPi: [https://pypi.org/user/ImYourBoyRoy/](https://pypi.org/user/ImYourBoyRoy/)

---

## License

MIT License
