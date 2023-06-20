from langchain.tools import DuckDuckGoSearchRun, Tool
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import AzureOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
import openai
from typing import List, Union
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
from langchain import LLMChain
from langchain.chains import LLMSummarizationCheckerChain
from langchain.callbacks import StdOutCallbackHandler
import requests
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.getenv('LANGCHAIN_API_KEY')

llm = AzureOpenAI(deployment_name="davinci",
                  model_name="text-davinci-003", temperature=0)


def call_bike_index_api(ip_address):
    url = f"https://bikeindex.org:443/api/v3/search?page=1&per_page=3&location={ip_address}&distance=10&stolenness=stolen"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_user_ip_address(a):
    try:
        response = requests.get('https://api.ipify.org?format=json')
        data = response.json()
        return data['ip']
    except requests.RequestException:
        return None


tools = [
    Tool(
        name="Search Bike Index",
        func=call_bike_index_api,
        description="Useful for searching for stolen bikes in a particular area. The input to this tool should be the user's IP address",
    ),
    Tool(
        name="Get User IP Address",
        func=get_user_ip_address,
        description="Useful for getting the user's IP address.",
    )
]


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

# Set up the base template
template_with_history = """Assistant is designed to be able to assist with finding a users lost/stolen bike
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. 
Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses.
Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions to help with finding a users lost/stolen bike.

Assistant has access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as someone trying to find a users lost/stolen bike when giving your final answer.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)


memory = ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

while True:
    user_input = input("User: ")
    agent_executor.run(user_input)
