# import os
from time import perf_counter

# from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "http://localhost:11434/v1"
API_KEY="ollama"          # значение не важно, но поле обязательно

MODEL = "gemma4:e4b"
# MODEL = "qwen3.5:9b"


def summarize_request(client: OpenAI, user_text: str) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Сформулируй краткое содержание обращения в одном "
                    "предложении. Не добавляй факты, которых нет в тексте.\n\n"
                    f"Обращение: {text}"
                ),
            }
        ],
    )
        
    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]
    answer = choice.message.content

    print("\nРезультат:")
    print(answer or "Модель не вернула текстовый ответ")

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    if response.usage is not None:
        print(f"Входные токены: {response.usage.prompt_tokens}")
        print(f"Выходные токены: {response.usage.completion_tokens}")
        print(f"Всего токенов: {response.usage.total_tokens}")

    print(f"ID ответа: {response.id}")



def main() -> None:

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text)


if __name__ == "__main__":
    main()
