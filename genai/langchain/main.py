from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

chat_model = ChatOpenAI()
print(chat_model.predict("hi"))

system_template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
Only return the comma separated list, and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of a LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of a LLM call."""
        return text.strip().split(", ")
    
chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = chat_prompt,
    output_parser = CommaSeparatedListOutputParser()
)

# blue, green, yellow, red, purple
print(chain.run("colours"))