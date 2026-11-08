# import os
from time import perf_counter

# from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "http://localhost:11434/v1"
API_KEY="ollama"          # значение не важно, но поле обязательно

MODEL = "gemma4:e4b"
# MODEL = "qwen3.5:9b"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 80

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


def validate_summary(summary: str | None) -> str:
    if summary is None:
        raise ValueError("Модель не вернула текст")

    cleaned = summary.strip()
    if not cleaned:
        raise ValueError("Модель вернула пустой ответ")
    if len(cleaned) > 300:
        raise ValueError("Резюме получилось слишком длинным")

    return cleaned


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
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
        
    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]

    if choice.finish_reason == "length":
        print(
            "Ответ модели остановлен из-за ограничения длины. "
            "Увеличьте MAX_OUTPUT_TOKENS или сократите задачу."
        )
        return

    if choice.finish_reason != "stop":
        print(
            "Модель не вернула готовое резюме. "
            f"Причина завершения: {choice.finish_reason}."
        )
        return
    
    try:
        answer = validate_summary(choice.message.content)
    except ValueError as error:
        print(f"Некорректный ответ модели: {error}")
        return

    print("\nРезультат:")
    print(answer)

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
