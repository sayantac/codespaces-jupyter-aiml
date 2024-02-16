from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Person(BaseModel):
    first_name: str = Field(description="first name")
    last_name: str = Field(description="last name")
    dob: str = Field(description="date of birth")

class PeopleList(BaseModel):
    people: list[Person] = Field(description="A list of people")

model = ChatOpenAI()

people_data = model.invoke(
    "Generate a list of 10 fake peoples information. Only return the list. Each person should have a first name, last name and date of birth.")

parser = PydanticOutputParser(pydantic_object=PeopleList)

prompt = PromptTemplate(template="Answer the user query.\n{format_instructions}\n {query}",
                        input_variables=["query"],
                        partial_variables={"format_instructions": parser.get_format_instructions()})

parser_input = prompt.format_prompt(query=people_data)
output = model.invoke(parser_input.to_string())

people_data_parsed = parser.parse(output.content)

print(people_data_parsed)