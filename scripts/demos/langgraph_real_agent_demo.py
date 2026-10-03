from serviceflow.agent.graph import run_customer_agent_graph


def main() -> None:
    answer = run_customer_agent_graph(
        "帮我查询订单20260721001"
    )

    print(answer)


if __name__ == "__main__":
    main()
