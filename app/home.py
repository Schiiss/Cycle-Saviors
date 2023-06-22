"""Cycle Saviour Home App Page"""
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
import pinecone
import os
from langchain.tools import Tool
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.llms import AzureOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent, AgentType
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
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from azure.storage.blob import BlobServiceClient, BlobClient
from langchain.tools import StructuredTool
import io
from transformers import AutoProcessor, BlipForQuestionAnswering
from PIL import Image
import datetime

st.set_page_config(layout='wide')

@st.cache_resource
def load_models():
    image_text_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image_text_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    image_embedding_model = SentenceTransformer("clip-ViT-B-32")
    st.success('All models initialized and cached!!!')
    return image_text_processor, image_text_model, image_embedding_model

image_text_processor, image_text_model, image_embedding_model = load_models()

# initialize pinecone index once with cache
@st.cache_resource
def load_pinecone(index_name):
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENV = os.getenv('PINECONE_ENV')
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    return pinecone.Index(index_name)

index = load_pinecone('cycle-saviours')

# set the title
st.markdown("<h1>Find My Stolen Bike!</h1>", unsafe_allow_html=True)

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

if 'ai_response_history' not in st.session_state:
    st.session_state['ai_response_history'] = []

# split into columns for chat and responses
col1, col2 = st.columns([10, 10])

# this column holds the results once the assistant finds them
with col1:
    if pinecone_output:
        html_placeholder = '<html><h6>Ads found online that match your image and description</h6>'
        for data in pinecone_output:
            listed_date = str(data['metadata']['list_date']).split('.')[0]
            html_placeholder += '<div style="background-color: rgba(213, 238, 247, 0.5);' \
                'border-radius: 10px; padding: 10px; margin-bottom: 20px;><table style="border: none;' \
                'background-color: lightblue; border-radius: 10px;">' \
            html_placeholder += '<tr style="border: none;">' \
                f'<h3><a href={data["metadata"]["ad_link"]} style="text-decoration:none;" target="_blank">{data["metadata"]["ad_title"]}</a></h3>' \
                f'</tr><tr style="border: none;"><strong>Ad Source: </strong>{data["metadata"]["source"]} <br>' \
                f'<strong>Ad Location: </strong>{data["metadata"]["ad_location"]} <br>' \
                f'<strong>Ad List Date: </strong>{listed_date[0:4]+"-"+listed_date[4:6]+"-"+listed_date[6:]}<br>' \
                f'<br></tr><tr style="border: none;"><img src={data["metadata"]["image_link"]} alt=Ad Id: {data["metadata"]["source_id"]}></tr><br></table></div>'

        html_placeholder += '</html>'

        st.markdown(html_placeholder, unsafe_allow_html=True)

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
os.getenv("AZURE_COGS_KEY")
os.getenv("AZURE_COGS_ENDPOINT")
os.getenv("AZURE_COGS_REGION")


llm = AzureOpenAI(deployment_name="davinci",
                  model_name="text-davinci-003", temperature=0)


def get_filtered_results(location: str, color: str, bike_model: str):
    """The input is the location of the bike theft and the color of the bike. example: 'calgary', 'Red', 'Trek'"""
    metadata_list = []
    account_name = "pineconehackathon"
    account_key = os.getenv("STORAGE_ACCOUNT_KEY")
    container_name = "container"

    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
    container_client = blob_service_client.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if blob.name.endswith(('.jpg', '.png', '.jpeg')):
            blob_client = container_client.get_blob_client(blob.name)
            blob_data = blob_client.download_blob()

            with io.BytesIO() as f:
                blob_data.readinto(f)
                f.seek(0)
                img_emb = image_embedding_model.encode(Image.open(f)).tolist()

            result = index.query(
                vector=img_emb,
                top_k=3,
                include_values=False,
                filter={
                    "ad_location": {"$eq": location}
                },
                include_metadata=True
            )

            matches = result.get('matches', [])
            for match in matches:
                metadata = match.get('metadata', {})
                metadata_list.append(metadata)

    return metadata_list


structured_tool = StructuredTool.from_function(get_filtered_results)

tools = [
    Tool(
        name="Search Bike Index by Location",
        func=get_filtered_results,
        description="Useful when you want to return a list of bike images and filter by location and color. The input is the location of the bike theft and the color of the bike. example: 'calgary', 'red'",
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

toolkit = AzureCognitiveServicesToolkit()

agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

memory = ConversationBufferMemory(memory_key="chat_history")

conversation_agent = initialize_agent(
    [structured_tool], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)


class ChatBot:
    def __init__(self):
        self.color = None
        self.location = None

    def ask_color(self):
        self.color = st.text_input("AI: What color is your bike?")
        return self.color

    def ask_location(self):
        self.location = st.text_input("AI: Where was it stolen?")
        return self.location

    def save_uploaded_file(self, uploaded_file):
        UPLOAD_DIR = "data/images"
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Upload the file to Azure Blob Storage
        account_name = "pineconehackathon"
        account_key = os.getenv("STORAGE_ACCOUNT_KEY")
        container_name = "container"
        blob_name = uploaded_file.name

        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net/", credential=account_key)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        return file_path

    def ask_file(self):
        uploaded_file = st.file_uploader("Please upload an image of your bike (REQUIRED)",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_path = self.save_uploaded_file(uploaded_file)
            st.success("File saved successfully.")
            return file_path
        else:
            return None


with col2:
    chatbot = ChatBot()
    file = chatbot.ask_file()
    city = st.selectbox('Please select a city (REQUIRED)', ['SELECT CITY', 'calgary', 'edmonton', 'vancouver'])
    if file and city:
        bike_image_analysis = agent.run(
            "what color is this bike? what model is this bike?"
            "https://pineconehackathon.blob.core.windows.net/container/large_IMG_1704.jpeg"
        )
        st.text(f"AI: Does this sound like your bike? {bike_image_analysis}")
        user_response = st.text_input('User Response:')
        if user_response.lower() == 'yes':
            output = conversation_agent.run(
                f"The user has had their bike stolen from them, please help them find it. DESCRIPTION: {bike_image_analysis} LOCATION: {city}. Once you have the links for the ad STOP & provide them the links to the ads.")
            pinecone_output = get_filtered_results(city, "black", "Trek")
            st.session_state.conversation_history.append(("AI:", output))
            st.session_state.ai_response_history.append(output)
    else:
        st.text("AI: Please provide an image of your bike and the city it was last seen in.")

    for i, (query, response) in enumerate(st.session_state.conversation_history):
        if query.startswith("AI:"):
            st.text("AI: " + response)
        else:
            st.text("User: " + query)
