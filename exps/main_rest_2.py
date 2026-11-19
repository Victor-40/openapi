import openai

# Настройка локального сервера
client = openai.Client(
    # base_url="http://localhost:8088/v1",  # Порт зависит от конфигурации
    base_url="http://localhost:11434/v1",  # Порт зависит от конфигурации
    api_key="ollama",  # Для llama-server API key часто не нужен или можно указать "ollama"
)

# Запрос к модели
chat = client.chat.completions.create(
    model="hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M",  # Или любую модель, загруженную в llama-server
    messages=[{"role": "user", "content": "расскажти о модуле threading python"}],
    temperature=0.3
)
print(chat.choices[0].message.content)