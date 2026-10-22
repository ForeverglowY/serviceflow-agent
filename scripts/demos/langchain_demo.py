from serviceflow.agent.langchain_service import run_langchain_agent


def main() -> None:
    response = run_langchain_agent("查询订单EVAL-NOT-EXIST-001。")
    print(response)


if __name__ == "__main__":
    main()
