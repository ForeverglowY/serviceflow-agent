from serviceflow.agent.service import run_customer_agent


def main() -> None:
    answer = run_customer_agent(
        "帮我查询订单20260721001"
    )
    print(answer)


if __name__ == "__main__":
    main()
