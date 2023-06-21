"""Cycle Saviour Home App Page"""
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
import pinecone
import os
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

st.set_page_config(layout='wide')

# initialize hugging face models once with cache


@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)


model = load_model('clip-ViT-B-32')

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

# state variables to hold chat history
if 'user_query_history' not in st.session_state:
    st.session_state['user_query_history'] = []

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

if 'ai_response_history' not in st.session_state:
    st.session_state['ai_response_history'] = []

# split into columns for chat and responses
col1, col2 = st.columns([0.7, 0.3])

# this column holds the results once the assistant finds them
with col1:
    # this will popup only on second and onward user input but that is okay in our case
    if st.session_state['user_query_history']:

        # this will ideally be the image, links, title description of add, etc.
        html_placeholder = '<html>'

        for i in range(0, 5):
            html_placeholder += '<div style="background-color: rgba(213, 238, 247, 0.5); border-radius: 10px; padding: 10px; margin-bottom: 20px;><table style="border: none;background-color: lightblue; border-radius: 10px;"><tr style="border: none;"><h3><a href="https://google.com" style="text-decoration:none;" target="_blank">"Placeholder"</a></h3></tr><tr style="border: none;">"Placeholder for ad text"<br><br></tr><tr style="border: none;"><strong>"More Placeholder space"</strong><br></tr><br></table></div>'

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

llm = AzureOpenAI(deployment_name="davinci",
                  model_name="text-davinci-003", temperature=0)


def get_filtered_results(location):
    model = SentenceTransformer('clip-ViT-B-32')
    image_links = []

    for file_name in os.listdir('data/images'):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join('data/images', file_name)
            img_emb = model.encode(Image.open(file_path)).tolist()

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

            matches = result.get('matches', [])
            for match in matches:
                metadata = match.get('metadata', {})
                image_link = metadata.get('image_link')

                if image_link:
                    image_links.append(image_link)

    return image_links


tools = [
    Tool(
        name="Search Bike Index by Location",
        func=get_filtered_results,
        description="Useful when you want to return a list of bike images and filter by a location. The input is the location of the bike theft. For example, 'calgary'. It should always be provide in lower case"
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

        return file_path

    def ask_file(self):
        uploaded_file = st.file_uploader("AI: Please upload an image of your bike (optional)",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_path = self.save_uploaded_file(uploaded_file)
            st.success("File saved successfully.")
            return file_path
        else:
            return None


with col2:
    chatbot = ChatBot()
    color = chatbot.ask_color()
    location = chatbot.ask_location()
    file = chatbot.ask_file()
    user_input = st.text_input('Start a chat with our assistant!',
                               key='input',
                               label_visibility='visible',
                               placeholder='My bike was stolen, help me find it!')

    if user_input:
        result = "User is looking for a bike with color: " + \
            color + " and location: " + location
        st.session_state.user_query_history.append(user_input)
        st.session_state.conversation_history.append((user_input, result))
        st.session_state.ai_response_history.append(result)
        if color and location:
            output = agent_executor.run(
                f"The user has had the following bike stolen from them, please help them find it: color: {color}, location: {location}. Ensure to provide them the links to the ads.")
            st.session_state.conversation_history.append(("AI:", output))
            st.session_state.ai_response_history.append(output)

    for i, (query, response) in enumerate(st.session_state.conversation_history):
        if query.startswith("AI:"):
            st.text("AI: " + response)
        else:
            st.text("User: " + query)
