from langchain_deepseek import ChatDeepSeek

from serviceflow.settings import settings


def create_chat_model() -> ChatDeepSeek:
    """创建模型"""
    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.deepseek_timeout_seconds,
        max_retries=settings.deepseek_max_retries,
        temperature=0,
    )
