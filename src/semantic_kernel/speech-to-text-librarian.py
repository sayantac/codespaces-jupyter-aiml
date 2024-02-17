from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import semantic_kernel as sk
from pathlib import Path
from openai import OpenAI

client = OpenAI()

with open(Path.cwd() / "data" / "03_06_book_rec2.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="text")
    print(transcript)

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_text_completion_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

base_prompt = "You are a librarian." +\
    "Provide a recommendation to a book based on the following information. {{$input}}." +\
"Explain your thinking step by step including a list of top books you selected and how you got to your final choice."

recommendation = kernel.create_semantic_function(base_prompt,max_tokens=512)

print(recommendation(transcript))