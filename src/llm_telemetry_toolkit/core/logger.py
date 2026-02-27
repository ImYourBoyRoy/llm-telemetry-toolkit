# ./src/llm_telemetry_toolkit/core/logger.py
"""
Implement the telemetry logger write pipeline and background worker lifecycle.
Used by application code to persist `LLMInteraction` records in configured formats.
Run: Imported as a library component or via package-level helper objects.
Inputs: `TelemetryConfig` at construction and `LLMInteraction` items via `log()`.
Outputs: Per-call `LogResult` plus JSON/MD/CSV files on disk by configured session/entity.
Side effects: Creates directories/files and runs a daemon thread for async writes.
Operational notes: Supports true sync writes, queue draining on shutdown, and path-safe routing.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

from ..io.formatters import FormatterFactory
from ..io.parser import ContentParser
from ..io.utils import sanitize_path_component
from ..models.config import TelemetryConfig
from ..models.results import LogResult
from ..models.schema import ChatMessage, LLMInteraction, ToolCall

logger = logging.getLogger(__name__)


class InteractionExporter(Protocol):
    """Protocol for optional downstream interaction exporters."""

    def export(self, interaction: LLMInteraction) -> None:
        """Export one prepared interaction."""

    def shutdown(self) -> None:
        """Flush/release exporter resources when supported."""


class LLMLogger:
    """Thread-safe logger supporting asynchronous queue writes and sync fallback writes."""

    _instance_lock = threading.Lock()
    _instances: Dict[str, "LLMLogger"] = {}

    def __new__(cls, config: TelemetryConfig) -> "LLMLogger":
        key = cls._build_instance_key(config)
        with cls._instance_lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, config: TelemetryConfig):
        if getattr(self, "_initialized", False):
            return

        self.config = config
        self._instance_key = self._build_instance_key(config)

        self._state_lock = threading.Lock()
        self._error_lock = threading.Lock()

        self._log_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"llm-telemetry-writer-{sanitize_path_component(config.session_id)}",
            daemon=True,
        )
        self._exporters: List[InteractionExporter] = []

        self.counter = 0
        self._recent_interactions: List[LLMInteraction] = []
        self._write_errors: Dict[str, List[str]] = {}
        self._is_shutdown = False

        self._setup_directories()
        self._setup_exporters()
        self._worker_thread.start()
        self._initialized = True
        logger.info(
            "LLMLogger initialized for session '%s' at '%s'",
            self.config.session_id,
            self.config.base_log_dir,
        )

    @staticmethod
    def _build_instance_key(config: TelemetryConfig) -> str:
        resolved_base = str(Path(config.base_log_dir).expanduser().resolve())
        output_formats = ",".join(config.output_formats)
        return "|".join(
            [
                config.session_id,
                resolved_base,
                config.session_log_subdir,
                config.entity_log_subdir,
                output_formats,
                config.filename_template,
                str(config.enable_session_logging),
                str(config.enable_entity_logging),
                str(config.mask_pii),
                str(config.max_content_length),
                str(config.enable_otel_export),
                config.otel_tracer_name,
                config.otel_span_name,
                config.otel_service_name,
                str(config.otel_auto_configure),
                config.otel_exporter,
                config.otel_otlp_endpoint,
                json.dumps(config.otel_otlp_headers, sort_keys=True),
                str(config.otel_otlp_timeout_seconds),
                str(config.otel_enable_console_export),
                str(config.otel_sampler_ratio),
                json.dumps(config.otel_resource_attributes, sort_keys=True),
            ]
        )

    @classmethod
    def clear_instances(cls) -> None:
        """Clear multiton instances for isolated test runs."""
        with cls._instance_lock:
            cls._instances.clear()

    def _setup_directories(self) -> None:
        safe_session_name = sanitize_path_component(
            self.config.session_id, fallback="session"
        )
        self.session_dir = (
            self.config.base_log_dir
            / self.config.session_log_subdir
            / safe_session_name
        )

        if self.config.enable_session_logging:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._write_session_config(self.session_dir, self.config.session_id)

        self.entity_base_dir: Optional[Path] = None
        if self.config.enable_entity_logging:
            self.entity_base_dir = (
                self.config.base_log_dir / self.config.entity_log_subdir
            )
            self.entity_base_dir.mkdir(parents=True, exist_ok=True)

    def _setup_exporters(self) -> None:
        """Initialize optional exporters from config without breaking file logging."""
        if not self.config.enable_otel_export:
            return

        try:
            from ..observability import OpenTelemetryInteractionExporter

            self._exporters.append(
                OpenTelemetryInteractionExporter(
                    tracer_name=self.config.otel_tracer_name,
                    span_name=self.config.otel_span_name,
                    service_name=self.config.otel_service_name,
                    auto_configure=self.config.otel_auto_configure,
                    exporter=self.config.otel_exporter,
                    otlp_endpoint=self.config.otel_otlp_endpoint,
                    otlp_headers=self.config.otel_otlp_headers,
                    otlp_timeout_seconds=self.config.otel_otlp_timeout_seconds,
                    enable_console_export=self.config.otel_enable_console_export,
                    sampler_ratio=self.config.otel_sampler_ratio,
                    resource_attributes=self.config.otel_resource_attributes,
                )
            )
            logger.info(
                "OpenTelemetry interaction exporter enabled (tracer=%s, span=%s).",
                self.config.otel_tracer_name,
                self.config.otel_span_name,
            )
        except Exception as exc:
            logger.warning(
                "OpenTelemetry export requested but unavailable: %s. "
                "File logging will continue.",
                exc,
            )

    def register_exporter(self, exporter: InteractionExporter) -> None:
        """Register a custom interaction exporter (for example metrics/tracing sinks)."""
        self._exporters.append(exporter)

    def _write_session_config(self, directory: Path, effective_session_id: str) -> None:
        """Write a stable session configuration payload for each session directory."""
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "session_config.json"
        if config_path.exists():
            return

        try:
            payload = self.config.model_dump(mode="json")
            payload["session_id"] = effective_session_id
            content = json.dumps(
                payload,
                indent=self.config.json_indent,
                ensure_ascii=self.config.ensure_ascii,
            )
            config_path.write_text(content, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive write guard
            logger.exception(
                "Failed to write session config '%s': %s", config_path, exc
            )

    def log(self, interaction: LLMInteraction, sync: bool = False) -> LogResult:
        """
        Log one interaction.
        - sync=False: queue write and return immediately.
        - sync=True: perform write inline and return confirmed file results.
        """
        start_time = time.perf_counter()

        if self._is_shutdown:
            latency = (time.perf_counter() - start_time) * 1000
            return LogResult(
                success=False,
                interaction_id=interaction.interaction_id,
                latency_ms=latency,
                errors=["Logger is shut down and cannot accept new interactions."],
                queued=False,
                write_confirmed=True,
            )

        prepared = self._prepare_interaction(interaction.model_copy(deep=True))

        if sync:
            return self._write_sync(prepared, start_time)

        self._log_queue.put(prepared)
        latency = (time.perf_counter() - start_time) * 1000
        return LogResult(
            success=True,
            interaction_id=prepared.interaction_id,
            latency_ms=latency,
            warnings=[
                "Queued for background write. Use sync=True for immediate disk confirmation."
            ],
            queued=True,
            write_confirmed=False,
        )

    def _prepare_interaction(self, interaction: LLMInteraction) -> LLMInteraction:
        """Normalize and enrich an interaction before disk writes."""
        if not interaction.session_id:
            interaction.session_id = self.config.session_id

        thought, clean_response = ContentParser.extract_thought_process(
            interaction.response
        )
        if thought:
            interaction.thought_process = thought
            interaction.response = clean_response

        if self.config.mask_pii:
            interaction.prompt = ContentParser.redact_pii(interaction.prompt)
            interaction.response = ContentParser.redact_pii(interaction.response)
            if interaction.thought_process:
                interaction.thought_process = ContentParser.redact_pii(
                    interaction.thought_process
                )
            if interaction.request_messages is not None:
                interaction.request_messages = cast(
                    List[ChatMessage],
                    ContentParser.redact_pii_recursive(interaction.request_messages),
                )
            if interaction.tool_calls is not None:
                interaction.tool_calls = cast(
                    List[ToolCall],
                    ContentParser.redact_pii_recursive(interaction.tool_calls),
                )
            interaction.metadata = cast(
                Dict[str, Any],
                ContentParser.redact_pii_recursive(interaction.metadata),
            )

        interaction.prompt = ContentParser.clean_and_truncate(
            interaction.prompt, self.config
        )
        interaction.response = ContentParser.clean_and_truncate(
            interaction.response, self.config
        )
        if interaction.thought_process:
            interaction.thought_process = ContentParser.clean_and_truncate(
                interaction.thought_process, self.config
            )

        with self._state_lock:
            self.counter += 1
            if self._looks_like_uuid(interaction.interaction_id):
                interaction.interaction_id = (
                    f"{interaction.session_id}_llm_{self.counter:04d}"
                )
            self._recent_interactions.append(interaction)
            if len(self._recent_interactions) > 100:
                self._recent_interactions.pop(0)

        return interaction

    @staticmethod
    def _looks_like_uuid(value: str) -> bool:
        try:
            uuid.UUID(value)
        except ValueError:
            return False
        return True

    def _write_sync(self, interaction: LLMInteraction, start_time: float) -> LogResult:
        """Write one interaction immediately and return deterministic result status."""
        try:
            created_files = self._write_to_disk(interaction)
            export_warnings = self._run_exporters(interaction)
            latency = (time.perf_counter() - start_time) * 1000
            primary_file = created_files[0] if created_files else None
            return LogResult(
                success=True,
                interaction_id=interaction.interaction_id,
                primary_log_path=primary_file,
                created_files=created_files,
                latency_ms=latency,
                warnings=export_warnings,
                queued=False,
                write_confirmed=True,
            )
        except Exception as exc:
            self._record_write_error(interaction.interaction_id, str(exc))
            latency = (time.perf_counter() - start_time) * 1000
            return LogResult(
                success=False,
                interaction_id=interaction.interaction_id,
                latency_ms=latency,
                errors=[str(exc)],
                queued=False,
                write_confirmed=True,
            )

    def _worker_loop(self) -> None:
        """Consume queued interactions and write each item to disk."""
        while True:
            item = self._log_queue.get()
            try:
                if item is None:
                    return
                self._write_to_disk(item)
                export_warnings = self._run_exporters(item)
                for warning in export_warnings:
                    self._record_write_error(item.interaction_id, warning)
            except Exception as exc:  # pragma: no cover - worker safety fallback
                interaction_id = (
                    item.interaction_id
                    if isinstance(item, LLMInteraction)
                    else "unknown_interaction"
                )
                self._record_write_error(interaction_id, str(exc))
                logger.exception(
                    "Worker failed writing interaction '%s': %s",
                    interaction_id,
                    exc,
                )
            finally:
                self._log_queue.task_done()

    def _write_to_disk(self, interaction: LLMInteraction) -> List[Path]:
        """Write one interaction to all configured targets and return created paths."""
        created_files: List[Path] = []

        if self.config.enable_session_logging:
            session_directory = self._session_directory_for(interaction.session_id)
            session_directory.mkdir(parents=True, exist_ok=True)
            self._write_session_config(session_directory, interaction.session_id)
            created_files.extend(
                self._write_files(
                    interaction=interaction,
                    base_dir=session_directory,
                    use_entity_subdir=False,
                    entity_key=None,
                )
            )

        if self.config.enable_entity_logging and (
            interaction.entity_id or interaction.entity_label
        ):
            if self.entity_base_dir is None:
                raise RuntimeError(
                    "Entity logging is enabled but entity base directory is unavailable."
                )
            entity_key = interaction.entity_label or interaction.entity_id or "entity"
            created_files.extend(
                self._write_files(
                    interaction=interaction,
                    base_dir=self.entity_base_dir,
                    use_entity_subdir=True,
                    entity_key=entity_key,
                )
            )

        return created_files

    def _run_exporters(self, interaction: LLMInteraction) -> List[str]:
        """Run optional exporters and return non-fatal warning messages."""
        warnings: List[str] = []
        if not self._exporters:
            return warnings

        for exporter in self._exporters:
            exporter_name = exporter.__class__.__name__
            try:
                exporter.export(interaction)
            except Exception as exc:  # pragma: no cover - exporter safety guard
                message = (
                    f"Exporter '{exporter_name}' failed for interaction "
                    f"'{interaction.interaction_id}': {exc}"
                )
                warnings.append(message)
                logger.warning(message)

        return warnings

    def _session_directory_for(self, session_id: str) -> Path:
        safe_session_name = sanitize_path_component(session_id, fallback="session")
        return (
            self.config.base_log_dir
            / self.config.session_log_subdir
            / safe_session_name
        )

    def _write_files(
        self,
        interaction: LLMInteraction,
        base_dir: Path,
        use_entity_subdir: bool,
        entity_key: Optional[str] = None,
    ) -> List[Path]:
        target_dir = base_dir
        if use_entity_subdir:
            safe_entity = sanitize_path_component(entity_key or "entity")
            safe_session = sanitize_path_component(
                interaction.session_id, fallback="session"
            )
            target_dir = base_dir / safe_entity / safe_session

        target_dir.mkdir(parents=True, exist_ok=True)

        created_files: List[Path] = []
        for format_name in self.config.output_formats:
            formatter = FormatterFactory.get_formatter(format_name)
            extension = formatter.file_extension()
            filename = self._render_output_filename(interaction, extension)
            file_path = target_dir / filename
            content = formatter.format(interaction, self.config)
            file_path.write_text(content, encoding="utf-8")
            created_files.append(file_path)

        return created_files

    def _render_output_filename(
        self, interaction: LLMInteraction, extension: str
    ) -> str:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        template_values = {
            "timestamp": timestamp_str,
            "interaction_id": interaction.interaction_id,
            "type": interaction.interaction_type or "generic",
            "ext": extension,
            "session_id": interaction.session_id,
            "model_name": interaction.model_name,
            "tool_name": interaction.tool_name or "tool",
        }

        try:
            rendered_name = self.config.filename_template.format(**template_values)
        except KeyError as exc:
            raise ValueError(
                f"Invalid filename_template placeholder '{exc.args[0]}'."
            ) from exc

        candidate_name = Path(rendered_name).name
        if not candidate_name:
            candidate_name = f"{interaction.interaction_id}.{extension}"

        expected_suffix = f".{extension.lower()}"
        if not candidate_name.lower().endswith(expected_suffix):
            candidate_name = f"{candidate_name}{expected_suffix}"

        parsed = Path(candidate_name)
        safe_stem = sanitize_path_component(
            parsed.stem, fallback=sanitize_path_component(interaction.interaction_id)
        )
        return f"{safe_stem}{expected_suffix}"

    def _record_write_error(self, interaction_id: str, error_message: str) -> None:
        with self._error_lock:
            self._write_errors.setdefault(interaction_id, []).append(error_message)

    def get_write_errors(
        self, interaction_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Return write errors for all interactions or one specific interaction id."""
        with self._error_lock:
            if interaction_id is None:
                return {key: errors[:] for key, errors in self._write_errors.items()}
            if interaction_id not in self._write_errors:
                return {}
            return {interaction_id: self._write_errors[interaction_id][:]}

    def get_recent_interactions(self, limit: int = 10) -> List[LLMInteraction]:
        """Return the most recent prepared interactions from in-memory cache."""
        with self._state_lock:
            if limit <= 0:
                return []
            return self._recent_interactions[-limit:]

    def shutdown(self) -> None:
        """Flush pending queue items and stop the background worker cleanly."""
        if self._is_shutdown:
            return

        self._is_shutdown = True
        self._log_queue.join()
        self._log_queue.put(None)
        self._worker_thread.join()
        for exporter in self._exporters:
            shutdown = getattr(exporter, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:  # pragma: no cover - defensive cleanup guard
                    logger.warning("Exporter shutdown failed: %s", exc)
        self._exporters.clear()
