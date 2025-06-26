from server.ai.ai_interface import AIInterface

model_to_use = input("Enter the model to use: ")
ai = AIInterface(
    provider="openai",
    api_key="sk-proj-1234567890",
    base_url="http://localhost:8009/v1",
)

response = ai.chat(
    model=model_to_use,
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente de IA que responde preguntas sobre la vida de Francisco de Goya",
        },
        {"role": "user", "content": "¿Quién es Francisco de Goya?"},
    ],
)

print(response)
