"""请求级追踪上下文，不保存客户消息或工具参数。"""
import logging
from collections.abc import Callable
from contextvars import ContextVar
from time import perf_counter
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)
current_trace_id: ContextVar[str] = ContextVar("trace_id", default="unscoped")
P = ParamSpec("P")
R = TypeVar("R")


def traced_call(operation: str, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    started = perf_counter()
    outcome = "error"
    try:
        result = function(*args, **kwargs)
        outcome = "success"
        usage = getattr(result, "usage", None)
        if usage is not None:
            logger.info(
                "model_usage trace_id=%s operation=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                current_trace_id.get(), operation, getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None), getattr(usage, "total_tokens", None),
            )
        else:
            usage_metadata = getattr(result, "usage_metadata", None)
            if usage_metadata is not None:
                logger.info(
                    "model_usage trace_id=%s operation=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    current_trace_id.get(), operation, usage_metadata.get("input_tokens"),
                    usage_metadata.get("output_tokens"), usage_metadata.get("total_tokens"), )

        return result
    finally:

        logger.info("operation_end trace_id=%s operation=%s outcome=%s elapsed_ms=%.2f",
                    current_trace_id.get(), operation, outcome, (perf_counter() - started) * 1000)
