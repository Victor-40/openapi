from openai import OpenAI 

# Инициализация клиента для Ollama
client = OpenAI(
    api_key="ollama",          # значение не важно, но поле обязательно
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="gemma4:e4b",          # имя модели, как в ollama list
    messages=[
        {"role": "system", "content": "Ты полезный помощник."},
        {"role": "user", "content": "Объясни, что такое асинхронность в Python простыми словами."}
    ],
    # temperature=0.7,
    # max_tokens=1500
)

print(response.choices[0])
print(response.choices[0].message.content)
