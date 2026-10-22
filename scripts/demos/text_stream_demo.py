from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from serviceflow.agent.runtime import AGENT_SYSTEM_MESSAGE
from serviceflow.llm.service import CLIENT
from serviceflow.settings import settings


def main() -> None:
    user_text = "请用三句话介绍你能提供哪些电商客服帮助。"

    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text, }

    messages: list[ChatCompletionMessageParam] = [AGENT_SYSTEM_MESSAGE, user_message, ]

    response = CLIENT.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        temperature=0,
        stream=True,
    )

    for chunk in response:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
