# import os
from time import perf_counter

# from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "http://localhost:11434/v1"
API_KEY="ollama"          # значение не важно, но поле обязательно

MODEL = "gemma4:e4b"
# MODEL = "qwen3.5:9b"

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

def print_usage(usage) -> None:
    if usage is None:
        print("Провайдер не вернул статистику токенов")
        return

    print(f"Входные токены: {usage.prompt_tokens}")
    print(f"Выходные токены: {usage.completion_tokens}")
    print(f"Всего токенов: {usage.total_tokens}")


def build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Обращение:\n{user_text}",
        },
    ]

def summarize_request(client: OpenAI, user_text: str) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(text),
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

    print_usage(response.usage)
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
