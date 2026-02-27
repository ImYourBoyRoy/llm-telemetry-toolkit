# ./src/llm_telemetry_toolkit/core/decorators.py
"""
Provide decorator helpers that auto-log sync and async function interactions.
Used by agent/tool functions to capture invocation context without manual log wiring.
Run: Imported as a library utility; decorators are applied in user application code.
Inputs: A configured `LLMLogger`, function args/kwargs, and optional interaction metadata.
Outputs: Original function return values plus telemetry interactions emitted to logger.
Side effects: Serializes call arguments/results into prompt/response fields for observability.
Operational notes: `log_errors=False` suppresses failure-event logging while still re-raising errors.
"""

from __future__ import annotations

import functools
import inspect
import time
import traceback
from typing import Any, Callable, Optional, TypeVar, cast

from .context import get_current_session_id
from .logger import LLMLogger
from ..models.schema import LLMInteraction

F = TypeVar("F", bound=Callable[..., Any])


def monitor_interaction(
    logger: LLMLogger,
    interaction_type: str = "function_call",
    tool_name: Optional[str] = None,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorate a callable and auto-log invocation inputs, latency, and output/error payloads.
    Supports both sync and async functions while preserving original behavior.
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            return cast(
                F,
                _build_async_wrapper(
                    func=cast(Callable[..., Any], func),
                    logger=logger,
                    interaction_type=interaction_type,
                    tool_name=tool_name,
                    log_errors=log_errors,
                ),
            )

        return cast(
            F,
            _build_sync_wrapper(
                func=cast(Callable[..., Any], func),
                logger=logger,
                interaction_type=interaction_type,
                tool_name=tool_name,
                log_errors=log_errors,
            ),
        )

    return decorator


def _build_sync_wrapper(
    *,
    func: Callable[..., Any],
    logger: LLMLogger,
    interaction_type: str,
    tool_name: Optional[str],
    log_errors: bool,
) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            if log_errors:
                _emit_log(
                    logger=logger,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    latency_seconds=time.perf_counter() - start_time,
                    interaction_type=interaction_type,
                    tool_name=tool_name,
                    response_text=f"Error: {error}\n{traceback.format_exc()}",
                    error=error,
                )
            raise

        _emit_log(
            logger=logger,
            func=func,
            args=args,
            kwargs=kwargs,
            latency_seconds=time.perf_counter() - start_time,
            interaction_type=interaction_type,
            tool_name=tool_name,
            response_text=str(result),
            error=None,
        )
        return result

    return wrapper


def _build_async_wrapper(
    *,
    func: Callable[..., Any],
    logger: LLMLogger,
    interaction_type: str,
    tool_name: Optional[str],
    log_errors: bool,
) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
        except Exception as error:
            if log_errors:
                _emit_log(
                    logger=logger,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    latency_seconds=time.perf_counter() - start_time,
                    interaction_type=interaction_type,
                    tool_name=tool_name,
                    response_text=f"Error: {error}\n{traceback.format_exc()}",
                    error=error,
                )
            raise

        _emit_log(
            logger=logger,
            func=func,
            args=args,
            kwargs=kwargs,
            latency_seconds=time.perf_counter() - start_time,
            interaction_type=interaction_type,
            tool_name=tool_name,
            response_text=str(result),
            error=None,
        )
        return result

    return wrapper


def _emit_log(
    *,
    logger: LLMLogger,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    latency_seconds: float,
    interaction_type: str,
    tool_name: Optional[str],
    response_text: str,
    error: Optional[Exception],
) -> None:
    session_id = get_current_session_id() or logger.config.session_id
    interaction = LLMInteraction(
        session_id=session_id,
        model_name="decorated_function",
        response_time_seconds=latency_seconds,
        prompt=_build_prompt(args=args, kwargs=kwargs),
        response=response_text,
        interaction_type=interaction_type,
        tool_name=tool_name or func.__name__,
        error_message=str(error) if error else None,
        validation_passed=error is None,
    )
    logger.log(interaction)


def _build_prompt(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return f"Args: {repr(args)}\nKwargs: {repr(kwargs)}"
