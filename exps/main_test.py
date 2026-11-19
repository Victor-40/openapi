from openai import OpenAI

DEVELOPER_INSTRUCTION = """
Ты помогаешь оператору службы поддержки кратко пересказывать
обращения клиентов.

Цель:
Сформулируй основную проблему клиента одним предложением.

Правила:
- Используй только факты, явно указанные в обращении.
- Если они указаны, сохраняй номер заказа, дату и название товара.
- Не добавляй советы, решения, обещания и оценки.
- Если деталь отсутствует или неясна, опусти ее.

Порядок работы:
1. Определи основную проблему клиента.
2. Выбери только факты, необходимые для понимания проблемы.
3. Удали приветствия, повторы и второстепенные подробности.
4. Сформулируй итоговый ответ в требуемом формате.

Формат ответа:
Одно самостоятельное предложение без заголовка, списка и вступления.
""".strip()

###############################################################################


DEVELOPER_INSTRUCTION = """
Ты помогаешь оператору службы поддержки кратко пересказывать
обращения клиентов.

Цель:
Сформулируй основную проблему клиента одним предложением.
""".strip()




messages = [
            {
                "role": "developer",
                "content": "Ты помогаешь оператору службы поддержки кратко пересказывать обращения клиентов. Цель Сформулируй основную проблему клиента одним предложением."
            },

            {
                "role": "user",
                "content": "когда прибудет мой оплаченный заказ?"
            }
        ]



# Укажите адрес вашего локального сервера (например, Ollama или Llama.cpp)
client = OpenAI(
    base_url="http://127.0.0.1:1337/v1",  
    api_key="not-needed",
    # timeout=1200
    

    
)              

# Теперь можно вызывать методы, как будто вы работаете с OpenAI
# response = client.chat.completions.create(
#     model="Qwen3_8-9B-Q4_K_M",
#     messages=[{"role": "user", "content": "Hi, how are you"}],
# )


response = client.chat.completions.create(
    model="Qwen3_5-4B-Q4_K_M",
    messages=messages,
    # extra_body={"enable_thinking": False}
   
    
)

choice = response.choices[0]
print(choice)
print("---------------")
text=choice.message.content
usage = response.usage
finish_reason=choice.finish_reason,
prompt_tokens=usage.prompt_tokens if usage is not None else None
completion_tokens=usage.completion_tokens if usage is not None else None
total_tokens=usage.total_tokens if usage is not None else None

print(text)
print(finish_reason)
print(prompt_tokens)
print(completion_tokens)
print(total_tokens)
