from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings('ignore')
llm = OpenAI(temperature=0)

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
tools = toolkit.get_tools()

desc_template = """
You are an AI assistant specializing in extracting specific information from Jira tickets. 
You have to read the text around the description for issue ID {jira_id} and return just the
value of the description.Exclude reading the images, new line breaks and links."""

ac_template = """
You are an AI assistant specializing in extracting specific information from Jira tickets. 
You have to read the custom field 'customfield_12655' for issue ID {jira_id} and return just the
value of the specific custom field without any added introduction."""

_prompt = PromptTemplate.from_template(desc_template)
prompt = _prompt.format(jira_id="IBOLIFE-2784")

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True
)

response = agent.invoke(prompt)["output"]

print(response)