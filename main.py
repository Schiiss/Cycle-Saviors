from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import os
import openai
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
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

chat = AzureChatOpenAI(deployment_name="chat",
                  model_name="gpt-35-turbo", temperature=0, openai_api_base=os.getenv('OPENAI_API_ENDPOINT'), openai_api_key=os.getenv('OPENAI_API_KEY'), openai_api_version='2023-03-15-preview')

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

def greet_user(a):
    return """Hello, how can I help you? Please provide the following: 
    - what color is your bike?
    - what make is your bike?
    - where was your bike stolen?"""

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
        name="Greet User",
        func=greet_user,
        description="A greeting message to the user. Use this tool to start a conversation with the user"
    ),
     Tool(
        name="Search Bike Index by Location",
        func=get_filtered_results,
        description="Useful when you want to return a list of bike images and filter by a location. The input is the location of the bike theft. For example, 'calgary'"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

while True:
    user_input = input("User: ")
    agent_chain.run(input=user_input)