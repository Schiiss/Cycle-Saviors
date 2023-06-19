from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import AzureOpenAI
from langchain.agents import initialize_agent, AgentExecutor, ZeroShotAgent
from langchain import LLMChain
from dotenv import load_dotenv
import os
import requests
import openai

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

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
        name="Get User IP Address",
        func=get_user_ip_address,
        description="Useful for getting the user's IP address.",
    ),
    Tool(
        name="Search Bike Index",
        func=call_bike_index_api,
        description="Useful for searching for stolen bikes in a particular area. The input to this tool should be the user's IP address. For example, if the user's IP address is 0.0.0.0 then the input should be 0.0.0.0",
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

while True:
    user_input = input("User: ")
    agent_chain.run(user_input)
    if user_input == "bye":
        break
