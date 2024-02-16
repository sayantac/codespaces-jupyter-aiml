from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def generate_review(review):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment classification bot, print out if the user is happy or sad. Only print out happy or sad."},
            {"role": "user", "content": review}
        ],
        temperature=0.7,
        max_tokens=150,
    )

    response_message = response.choices[0].message.content
    if response_message == "happy":
        return "Thanks for shopping with us, come back soon!"
    return "Sorry to hear about your experience, here's a coupon for 20% off, type GPT20 to use it!"