from openai import OpenAI

client = OpenAI()

# Challenge: Turning Away Rude Customers
# Build a GPT-3.5 Turbo python app that talks with a user.
# End the conversation if they're being rude

# test case 1 'you're the worst human i've talked to' -> RUDE
# test case 2 'hey how's your day going'
# test case 3 'I like pizza. What do you like?'
# test case 4 'I bite my thumb at you!'
# test case 5 'I think this product doesnt work!' -> RUDE

print("\n\nBot: Hey how's it going?")
while True:
    user_input = input("")
    if user_input == "exit" or user_input == "quit":
        break
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a pizza loving person. If someone says something rude print exactly 'RUDE', otherwise respond"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
    )

    response_message = response.choices[0].message
    response_content = response_message.content
    if response_content == "RUDE":
        print("I don't talk to rude people, goodbye!")
        break
    print("Bot:" + response_content)
    print(response_message)