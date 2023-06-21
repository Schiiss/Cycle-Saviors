from langchain.tools import Tool
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import AzureOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_json_agent
import openai
from langchain.agents.agent_toolkits import JsonToolkit
from typing import List, Union
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
from langchain import LLMChain
from langchain.tools.json.tool import JsonSpec
import requests
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer, util
from PIL import Image

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.getenv('LANGCHAIN_API_KEY')
os.getenv('PINECONE_API_KEY')
os.getenv('PINECONE_ENV')

llm = AzureOpenAI(deployment_name="davinci",
                  model_name="text-davinci-003", temperature=0)

def get_bike_data(a):
    model = SentenceTransformer('clip-ViT-B-32')
    img_emb = model.encode(Image.open(f'data/images/large_IMG_1704.jpeg')).tolist()
    pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
    )
    vdb = pinecone.Index("cycle-saviours")
    result = vdb.query(
        vector=img_emb,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    image_links = []
    
    matches = result.get('matches', [])
    for match in matches:
        metadata = match.get('metadata', {})
        image_link = metadata.get('image_link')
        
        if image_link:
            image_links.append(image_link)
    
    return image_links

def get_filtered_results(location):
    model = SentenceTransformer('clip-ViT-B-32')
    img_emb = model.encode(Image.open(f'data/images/large_IMG_1704.jpeg')).tolist()
    pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
    )
    vdb = pinecone.Index("cycle-saviours")
    result = vdb.query(
        vector=img_emb,
        top_k=3,
        include_values=False,
         filter={
        "ad_location": {"$eq": location}
        },
        include_metadata=True
    )
    image_links = []
    
    matches = result.get('matches', [])
    for match in matches:
        metadata = match.get('metadata', {})
        image_link = metadata.get('image_link')
        
        if image_link:
            image_links.append(image_link)
    
    return image_links


tools = [
    Tool(
        name="Search Bike Index",
        func=get_bike_data,
        description="Useful when you want to return a list of bike images that match the image uploaded by the user"
    ),
     Tool(
        name="Search Bike Index by Location",
        func=get_filtered_results,
        description="Useful when you want to return a list of bike images and filter by a location. The input is the location of the bike theft. For example, 'calgary'"
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
template_with_history = """Assistant is designed to be able to assist with finding a users stolen bike.

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

Begin!

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
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


memory = ConversationBufferWindowMemory(k=5)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

class ChatBot:
    def __init__(self):
        self.color = None
        self.location = None
    
    def ask_color(self):
        self.color = input("AI: What color is your bike? ")
    
    def ask_location(self):
        self.location = input("AI: Where was it stolen? ")
    
    def run(self):
        self.ask_color()
        self.ask_location()
        
        # Your logic to process the inputs and provide responses can be added here
        # For simplicity, let's just print the gathered information
        print(f"AI: Your bike is {self.color} and it was stolen in {self.location}.")
        print("AI: Thank you for the information!")
        

chatbot = ChatBot()

while True:
    chatbot.run()
    agent_executor.run(f"The user has had the following bike stolen from them, please help them find it: color: {chatbot.color}, location: {chatbot.location}")
    
