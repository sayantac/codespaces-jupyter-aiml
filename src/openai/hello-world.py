from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Respond in French"},
        {"role": "user", "content": "Say 'Hello world'"}
    ],
    temperature=0.7,
    max_tokens=150,
)

response_message = response.choices[0].message.content
print(response_message)