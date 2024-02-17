from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Furniture(BaseModel):
    type: str = Field(description="type of furniture")
    style: str = Field(description="style of furniture")
    colour: str = Field(description="colour of furniture")

furniture_request = "I'd like a blue mid century chair"

parser = PydanticOutputParser(pydantic_object=Furniture)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()})

_input = prompt.format_prompt(query=furniture_request)
model = ChatOpenAI()

output = model.invoke(_input.to_string())
parsed = parser.parse(output.content)

print(parsed.colour)