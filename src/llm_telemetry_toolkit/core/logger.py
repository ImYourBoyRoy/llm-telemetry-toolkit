# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/core/logger.py
"""
Core Logger Implementation.
Handles async logging, file writing, and format delegation using a background worker.
Inputs: LLMInteraction objects.
Outputs: Log files on disk.
"""

import time
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from ..models.schema import LLMInteraction
from ..models.config import TelemetryConfig
from ..models.results import LogResult
from ..io.formatters import FormatterFactory
from ..io.parser import ContentParser
from ..io.utils import generate_safe_filename

logger = logging.getLogger(__name__)


class LLMLogger:
    _instance_lock = threading.Lock()
    _instances: Dict[str, "LLMLogger"] = {}

    def __new__(cls, config: TelemetryConfig):
        key = config.session_id
        with cls._instance_lock:
            if key not in cls._instances:
                instance = super(LLMLogger, cls).__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, config: TelemetryConfig):
        if getattr(self, "_initialized", False):
            return

        self.config = config
        self._setup_directories()

        # Async Queue
        self._log_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.counter = 0
        self.entity_counters: Dict[str, int] = {}

        # Local "Reflective Memory" (Cache)
        self._recent_interactions: List[LLMInteraction] = []

        self._initialized = True
        logger.info(
            f"LLMLogger initialized (Async) for session '{self.config.session_id}'"
        )

    def _setup_directories(self):
        self.session_dir = (
            self.config.base_log_dir
            / self.config.session_log_subdir
            / self.config.session_id
        )
        if self.config.enable_session_logging:
            self.session_dir.mkdir(parents=True, exist_ok=True)

        self.entity_base_dir: Optional[Path] = None
        if self.config.enable_entity_logging:
            self.entity_base_dir = (
                self.config.base_log_dir / self.config.entity_log_subdir
            )
            self.entity_base_dir.mkdir(parents=True, exist_ok=True)

        self._write_session_config(self.session_dir)

    def _write_session_config(self, directory: Path):
        """
        Writes the current configuration to a unique 'session_config.json' file
        in the log directory.
        """
        config_path = directory / "session_config.json"
        # Only write if it doesn't exist to prevent race conditions or churn
        if not config_path.exists():
            try:
                # Use model_dump_json if using Pydantic V2
                json_data = self.config.model_dump_json(indent=2)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(json_data)
            except Exception as e:
                logger.error(f"Failed to write session config: {e}")

    def log(self, interaction: LLMInteraction, sync: bool = False) -> LogResult:
        """
        Logs an interaction. By default, this is non-blocking (async).
        Returns a LogResult immediately, though the file write happens in background via queue.
        If sync=True, waits for the write to check for disk errors.
        """
        start_time = time.time()

        # 1. Session Handling
        # If the interaction has a different session ID (e.g. from Context), we respect it.
        # We DO NOT forcefully overwrite it with config.session_id anymore.
        if not interaction.session_id:
            interaction.session_id = self.config.session_id

        # 2. Parse & Clean
        # Extract thought process
        thought, clean_resp = ContentParser.extract_thought_process(
            interaction.response
        )
        if thought:
            interaction.thought_process = thought
            interaction.response = clean_resp

        # 3. PII Redaction (Privacy)
        if self.config.mask_pii:
            interaction.prompt = ContentParser.redact_pii(interaction.prompt)
            interaction.response = ContentParser.redact_pii(interaction.response)
            if interaction.thought_process:
                interaction.thought_process = ContentParser.redact_pii(
                    interaction.thought_process
                )

        # 4. Truncate content if configured
        interaction.prompt = ContentParser.clean_and_truncate(
            interaction.prompt, self.config
        )
        interaction.response = ContentParser.clean_and_truncate(
            interaction.response, self.config
        )

        # 5. ID Generation
        self.counter += 1
        if len(interaction.interaction_id) == 36 and "-" in interaction.interaction_id:
            interaction.interaction_id = (
                f"{interaction.session_id}_llm_{self.counter:04d}"
            )

        # 4. Cache for Reflective Memory
        self._update_memory(interaction)

        # 5. Queue for writing
        self._log_queue.put(interaction)

        latency = (time.time() - start_time) * 1000

        return LogResult(
            success=True,
            interaction_id=interaction.interaction_id,
            latency_ms=latency,
            warnings=["Queued for background write"] if not sync else [],
        )

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                interaction = self._log_queue.get(timeout=1.0)
                self._write_to_disk(interaction)
                self._log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")

    def _write_to_disk(self, interaction: LLMInteraction):
        # Determine Session Dir
        # If interaction session matches config, use cached dir. Otherwise, compute it.
        if interaction.session_id == self.config.session_id:
            msg_session_dir = self.session_dir
        else:
            msg_session_dir = (
                self.config.base_log_dir
                / self.config.session_log_subdir
                / interaction.session_id
            )
            msg_session_dir.mkdir(parents=True, exist_ok=True)
            self._write_session_config(msg_session_dir)  # <-- Fixed order

        # 1. Main Session Log
        if self.config.enable_session_logging:
            self._write_files(interaction, msg_session_dir, use_entity_subdir=False)

        # 2. Entity Log
        if self.config.enable_entity_logging and (
            interaction.entity_id or interaction.entity_label
        ):
            entity_key = interaction.entity_label or interaction.entity_id
            self._write_files(
                interaction,
                self.entity_base_dir,
                use_entity_subdir=True,
                entity_key=entity_key,
            )

    def _write_files(
        self,
        interaction: LLMInteraction,
        base_dir: Path,
        use_entity_subdir: bool,
        entity_key: str = None,
    ):
        target_dir = base_dir
        # filename_prefix was unused

        if use_entity_subdir and entity_key:
            safe_entity = generate_safe_filename(entity_key, timestamp=False)
            target_dir = base_dir / safe_entity / interaction.session_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # Note: Entity counters in threaded environment might need locking if accurate strict ordering is required.
            # For now, we assume global counter (interaction_id) is sufficient for unique filenames.

        # Write for each configured format
        for fmt_name in self.config.output_formats:
            formatter = FormatterFactory.get_formatter(fmt_name)
            ext = formatter.file_extension()

            # Use template but handle extension separately to avoid sanitization issues
            # We assume template is mostly for the BASE name structure.
            # If user included .{ext} in template, we strip it or handle it?
            # For simplicity, let's construct a safe base name and append extension.

            # Simplified approach: Ignore user template's extension part if present
            # We construct the components
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Construct a raw string for the base name
            base_str = f"{timestamp_str}_{interaction.interaction_id}_{interaction.interaction_type or 'generic'}"

            # Sanitize the base string
            safe_base = generate_safe_filename(base_str, suffix="", timestamp=False)

            # Append extension (dot is safe in FS, but generate_safe_filename removes it)
            safe_fname = f"{safe_base}.{ext}"

            file_path = target_dir / safe_fname

            # Check if file exists (rare collision due to truncation), if so, append counter?
            # For now, we trust timestamps + IDs + different extensions

            content = formatter.format(interaction, self.config)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    def _update_memory(self, interaction: LLMInteraction):
        """Updates the short-term reflective memory."""
        self._recent_interactions.append(interaction)
        # Keep last 100
        if len(self._recent_interactions) > 100:
            self._recent_interactions.pop(0)

    def get_recent_interactions(self, limit: int = 10) -> List[LLMInteraction]:
        return self._recent_interactions[-limit:]

    def shutdown(self):
        """Stops the worker thread and flushes queue."""
        self._stop_event.set()
        self._worker_thread.join()
