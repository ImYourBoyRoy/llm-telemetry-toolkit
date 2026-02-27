# ./src/llm_telemetry_toolkit/core/context.py
"""
Manage implicit session context for telemetry logging across call stacks.
Used by decorators and manual logging flows to avoid passing session_id everywhere.
Run: Imported by logger/decorator consumers; no standalone CLI execution.
Inputs: Session IDs passed to `SessionContext`.
Outputs: Context manager state and `get_current_session_id()` lookup values.
Side effects: Updates a context-local variable for the active execution scope.
Operational notes: Context values are isolated per task/thread via `contextvars`.
"""

import contextvars
from typing import Optional

# Global context variable to store the current session ID
_current_session_id = contextvars.ContextVar("current_session_id", default=None)


class SessionContext:
    """
    Context manager to set the current session ID for implicit logging.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.token = None

    def __enter__(self):
        self.token = _current_session_id.set(self.session_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _current_session_id.reset(self.token)


def get_current_session_id() -> Optional[str]:
    """Returns the currently active session ID from context."""
    return _current_session_id.get()
