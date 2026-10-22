from serviceflow.llm.langchain_service import classify_intent


def main() -> None:
    while True:
        user_text = input("问题：")

        if user_text == "exit":
            break
        print(classify_intent(user_text))


if __name__ == "__main__":
    main()
