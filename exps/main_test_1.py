"""
Программа для сокращения обращений пользователей до одного предложения
использует локальный сервер (Ollama-compatible) по адресу http://localhost:1337/v1

Не требует torch/transformers — только официальная библиотека openai.
"""

import os
from typing import Optional
import openai


class LocalSummarizer:
    """Класс для сокращения текстовых обращений до одного предложения через локальный API."""

    # def __init__(self, base_url: str = "http://localhost:1337/v1", api_key: Optional[str] = None):
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: Optional[str] = None):
        """
        Инициализация клиента.

        Args:
            base_url: Адрес локального сервера (по умолчанию http://localhost:1337/v1).
            api_key: API-ключ (если требуется, например для Ollama с auth). Оставьте None для публичных моделей.
        """
        # import openai

        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.api_key = "not-needed"

        # Если ключ не указан — используем пустую строку (подходит для большинства локальных серверов)
        if not self.api_key:
            self.api_key = ""

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def summarize_to_one_sentence(self, text: str, model: Optional[str] = None) -> str:
        """
        Сокращает текст обращения пользователя до одного предложения.

        Args:
            text: Исходный текст обращения (стринг).
            model: Модель для inference (по умолчанию берётся из env или 'qwen2.5').

        Returns:
            Строка — одно предложение с сутью обращения.

        Raises:
            ValueError: Если текст пустой или слишком короткий.
            Exception: При ошибке API локального сервера.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Текст не может быть пустым")

        if len(text.strip()) < 10:
            return text.strip()

        prompt = (
            "Сократи этот текст обращения пользователя до одного предложения. "
            "Сохраняй суть проблемы и ключевую информацию, но упрости формулировку. "
            "Ответь ТОЛЬКО одним предложением на русском языке.\n\n"
            f"{text}"
        )

        try:
            response = self.client.chat.completions.create(
                # model= "Qwen3_5-4B-Q4_K_M",  # или другая локальная модель
                model= "gemma4:e4b",  # или другая локальная модель
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,          # небольшой шум для разнообразия
                max_tokens=150            # ограничиваем длину ответа
            )

            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            print("⚠️  Ошибка API локального сервера: лимит запросов исчерпан.")
            raise
        except openai.APIConnectionError as e:
            print(f"❌ Не удалось подключиться к локальному серверу ({self.base_url}): {e}")
            raise
        except openai.AuthenticationError as e:
            print(f"❌ Ошибка аутентификации: проверьте API-ключ или модель.")
            raise

    def batch_summarize(self, texts: list[str], model: Optional[str] = None) -> list[tuple[str, str]]:
        """
        Обработка списка обращений.

        Args:
            texts: Список строк с обращениями пользователей.
            model: Модель для inference (опционально).

        Returns:
            Список кортежей (исходный_текст, сокращённый_текст).
        """
        results = []
        for i, text in enumerate(texts):
            try:
                summarized = self.summarize_to_one_sentence(text, model=model)
                original_preview = text[:60] + "..." if len(text) > 60 else text
                results.append((original_preview, summarized))
            except Exception as e:
                print(f"⚠️  Ошибка при обработке обращения {i+1}: {e}")
                results.append((text[:80], f"Ошибка обработки"))

        return results


def main():
    """Основная функция — демонстрация работы программы."""

    # ── Настройка адреса сервера ─────────────────────────────────────
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b")  # или llama3, mistral и т.д.

    print("=" * 70)
    print(f"📝 ПРОГРАММА: Сокращение обращений (локальный сервер)")
    print(f"   Адрес сервера: {base_url}")
    print(f"   Модель:        {model_name}")
    print("=" * 70)
    print()

    # ── Запуск клиента ───────────────────────────────────────────────
    summarizer = LocalSummarizer(base_url=base_url, api_key="not-needed")

    # ── Примеры обращений ────────────────────────────────────────────
    examples = [
        "Я хочу вернуть товар, который купил 3 дня назад в вашем интернет-магазине. Товар не подошёл по размеру и имеет царапины на экране. Прошу оформить возврат в течение 14 дней согласно закону о защите прав потребителей.",

        "У меня возникла проблема с подключением к вашему онлайн-банку — приложение выдает ошибку 'соединение прервано', хотя интернет работает стабильно. Пробовала обновлять браузер, перезагружать страницу, ничего не помогает. Пожалуйста, помогите разобраться!",

        "Меня интересует возможность оформления рассрочки на покупку техники в вашем магазине. Подскажите, какие условия и процентная ставка действуют сейчас для постоянных клиентов? Также хочу узнать, нужно ли оформлять кредитный договор или это просто беспроцентная рассрочка.",
    ]

    # ── Вывод результатов ────────────────────────────────────────────
    results = summarizer.batch_summarize(examples)

    for original, summarized in results:
        print(f"┌─ Оригинальное обращение ({len(original)} символов) ─────────────┐")
        print(f"│ {original}")
        print(f"├──────────────────────────────────────────────────────────────────┤")
        print(f"│ 📌 Сокращено до одного предложения:                              │")
        print(f"│   \"{summarized}\"                                                │")
        print("=" * 70)

    # ── Интерактивный режим (опционально) ───────────────────────────
    print()
    want_interactive = input("Хотите протестировать со своим текстом? (y/n): ")
    if want_interactive.lower() == "y":
        user_text = input("Введите текст обращения: ")
        if user_text.strip():
            result = summarizer.summarize_to_one_sentence(user_text)
            print(f"\n📌 Сокращено: \"{result}\"")


if __name__ == "__main__":
    main()
