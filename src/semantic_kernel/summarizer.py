import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
openai_chat_service = OpenAIChatCompletion(ai_model_id="gpt-3.5-turbo", api_key=api_key, org_id=org_id)

kernel.add_text_completion_service("chat-gpt", openai_chat_service)

prompt = kernel.create_semantic_function("""1) A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2) A robot must obey orders given it by human beings except where such orders would conflict with the First Law.
3) A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.
                                         
Give me the TLDR in exactly 5 words.""")

print(prompt())

sk_prompt = """{{$input}}

Give me one line TLDR with the fewest words.
"""

prompt_input = """1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""

# Run the prompt
summarize = kernel.create_semantic_function(
    prompt_template=sk_prompt, max_tokens=200, temperature=0, top_p=0.5)

summary = summarize(prompt_input)

print(f"Output: {summary}")