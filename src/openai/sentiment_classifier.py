from openai import OpenAI

client = OpenAI()

while True:
    user_input = input("Enter a phrase and we'll tell you if its happy or sad.\n")
    if user_input == "exit" or user_input == "quit":
        break
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment classification bot, print out if the user is happy or sad"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
    )

    response_message = response.choices[0].message.content
    print(response_message)