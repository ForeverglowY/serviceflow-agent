import json
import logging
from collections.abc import Iterator
from time import perf_counter
from uuid import uuid4

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from serviceflow.agent.runtime import AGENT_SYSTEM_MESSAGE
from serviceflow.llm.service import CLIENT
from serviceflow.observability import current_trace_id
from serviceflow.settings import settings

logger = logging.getLogger(__name__)


def stream_customer_reply(user_text: str, *, trace_id: str | None = None) -> Iterator[str]:
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text, }

    messages: list[ChatCompletionMessageParam] = [AGENT_SYSTEM_MESSAGE, user_message, ]

    with CLIENT.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
    ) as response:
        for chunk in response:
            # 用量可能在无文本、甚至 choices 为空的最后一块中返回。
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                logger.info(
                    "model_usage trace_id=%s operation=llm.stream_reply "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    trace_id or current_trace_id.get(), usage.prompt_tokens,
                    usage.completion_tokens, usage.total_tokens,
                )
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content


def encode_sse_event(event: str, data: dict) -> str:
    json_text = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {json_text}\n\n"


def stream_customer_events(user_text: str, trace_id: str | None = None) -> Iterator[str]:
    trace_id = trace_id or uuid4().hex
    started_at = perf_counter()
    first_chunk_ms = None
    chunk_count = 0
    outcome = "interrupted"
    logger.info("stream_start trace_id=%s", trace_id)
    try:
        for text in stream_customer_reply(user_text, trace_id=trace_id):
            if first_chunk_ms is None:
                first_chunk_ms = (perf_counter() - started_at) * 1000
                logger.info("stream_first_text trace_id=%s elapsed_ms=%.2f", trace_id, first_chunk_ms)
            chunk_count += 1
            yield encode_sse_event("delta", {"text": text})
    except OpenAIError:
        outcome = "error"
        logger.exception("模型流式响应失败 trace_id=%s", trace_id)
        yield encode_sse_event(
            "error",
            {"message": "模型服务暂时不可用，请稍后重试", "trace_id": trace_id},
        )
    else:
        outcome = "success"
        yield encode_sse_event("done", {"trace_id": trace_id})
    finally:
        logger.info(
            "stream_end trace_id=%s outcome=%s elapsed_ms=%.2f first_chunk_ms=%s chunk_count=%d",
            trace_id, outcome, (perf_counter() - started_at) * 1000, first_chunk_ms, chunk_count,
        )
