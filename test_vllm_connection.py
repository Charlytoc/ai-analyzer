from server.ai.ai_interface import AIInterface

ai = AIInterface(
    provider="openai",
    api_key="sk-proj-1234567890",
    base_url="http://localhost:8009/v1",
)

response = ai.chat(
    messages=[
        {"role": "user", "content": "Hola, ¿cómo estás?"},
    ],
)

print(response)
