# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/core/decorators.py
"""
Decorators for automatic interaction logging.
Provides easy-to-use wrappers like @monitor_interaction.
Inputs: Functions to decorate.
Outputs: Wrapped functions with logging hooks.
"""

import time
import functools
import traceback
from typing import Optional, Callable

from .logger import LLMLogger
from ..models.schema import LLMInteraction
from .context import get_current_session_id


def monitor_interaction(
    logger: LLMLogger,
    interaction_type: str = "function_call",
    tool_name: Optional[str] = None,
    log_errors: bool = True,
):
    """
    Decorator to automatically log function calls as log interactions.
    Captures input (args/kwargs) as prompt and result as response.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            response_str = ""

            try:
                result = func(*args, **kwargs)
                response_str = str(result)
                return result
            except Exception as e:
                error = e
                response_str = f"Error: {str(e)}\n{traceback.format_exc()}"
                if not log_errors:
                    raise e
                raise e
            finally:
                end_time = time.time()
                latency = end_time - start_time

                # Construct Interaction
                # Resolving session ID: Use context content if available, else logger's default
                session_id = get_current_session_id() or logger.config.session_id

                interaction = LLMInteraction(
                    session_id=session_id,
                    model_name="decorated_function",  # Placeholder
                    response_time_seconds=latency,
                    prompt=f"Args: {args}\nKwargs: {kwargs}",
                    response=response_str,
                    interaction_type=interaction_type,
                    tool_name=tool_name or func.__name__,
                    error_message=str(error) if error else None,
                    validation_passed=not bool(error),
                )

                logger.log(interaction)

        return wrapper

    return decorator
