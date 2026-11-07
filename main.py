# import os

# from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "http://localhost:11434/v1"
API_KEY="ollama"          # значение не важно, но поле обязательно

MODEL = "gemma4:e4b"
# MODEL = "qwen3.5:9b"


def main() -> None:

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Кратко перескажи: клиент дважды оплатил один заказ.",
            }
        ],
    )

    choice = response.choices[0]
    print(len(response.choices))
    answer = choice.message.content

    print("Результат:")
    print(answer or "Модель не вернула текст")

    print("\nДиагностика:")
    print(f"ID ответа: {response.id}")
    print(f"Модель: {response.model}")
    print(f"Причина завершения: {choice.finish_reason}")

    if response.usage is not None:
        print(f"Входные токены: {response.usage.prompt_tokens}")
        print(f"Выходные токены: {response.usage.completion_tokens}")
        print(f"Всего токенов: {response.usage.total_tokens}")
    else:
        print("Провайдер не вернул статистику токенов")


if __name__ == "__main__":
    main()
