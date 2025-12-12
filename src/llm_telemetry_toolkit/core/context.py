# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/core/context.py
"""
Context Management for implicit session handling.
Uses Python's contextvars to maintain session state across call stacks.
Inputs: Session IDs.
Outputs: Context managers and current state.
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
