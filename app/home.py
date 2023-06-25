"""Cycle Saviour Home App Page"""
import streamlit as st
import pinecone
import os
import boto3
import yaml
from sentence_transformers import SentenceTransformer
from streamlit_chat import message
from langchain.tools import StructuredTool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain import OpenAI
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
from langchain.chains import ConversationChain

# read in user configurations
with open("config.yaml", "r") as config_file:
    configs = yaml.safe_load(config_file)
####################################################
# BACKGROUND CONFIGURATIONS AND CACHE
####################################################
load_dotenv('../configs/.env')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    model_name=configs['openai_models']['chat'],
    temperature=0)

st.set_page_config(layout='wide', page_title="Find My Bike!", page_icon=":robot:")

@st.cache_resource
def load_models():
    image_text_model = BlipForQuestionAnswering.from_pretrained(configs['huggingface_models']['image_to_text'])
    image_text_processor = AutoProcessor.from_pretrained(configs['huggingface_models']['image_to_text'])
    image_embedding_model = SentenceTransformer(configs['huggingface_models']['image_embedding'])
    st.success('All models initialized and cached!!!')
    return image_text_processor, image_text_model, image_embedding_model

image_text_processor, image_text_model, image_embedding_model = load_models()

@st.cache_resource
def load_pinecone(index_name):
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENV = os.getenv('PINECONE_ENV')
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    return pinecone.Index(index_name)

index = load_pinecone(configs['pinecone']['index_name'])

# initialize with ai speaking
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hi! I am the Cycle Saviors stolen bike finding assistant. " \
                  "I search online marketplaces for your stolen bike as thieves often list bikes on these spaces. "\
                  "Start by uploading an image and selecting the location where the bike was stolen from. " \
                  "If you would like to learn more about our project visit https://github.com/Schiiss/Cycle-Saviors"]

if "past" not in st.session_state:
    st.session_state["past"] = []

if 'pinecone_output' not in st.session_state:
    st.session_state['pinecone_output'] = []

if 'image_description' not in st.session_state:
    st.session_state['image_description'] = ""

if 'search_button' not in st.session_state:
    st.session_state['search_button'] = False
####################################################
# FUNCTIONS
####################################################
def get_filtered_results(location: str):
    """Search pinecone index for similar images
    
    Args:
        location: name of city
        
    Returns:
        List of metadata for mathces
    """
    metadata_list = []

    img_emb = image_embedding_model.encode(Image.open(file_path)).tolist()

    result = index.query(
        vector=img_emb,
        top_k=10,
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

def get_image_text(filepath: str):
    """Use pinecone image to text model to create description
    
    Args:
        filepath: path to image
        
    Returns:
        A description of the bike
    """    
    image = Image.open(filepath)

    text = "what color is this bike?"
    inputs = image_text_processor(images=image, text=text, return_tensors="pt")
    outputs = image_text_model.generate(**inputs)
    text_output1 = image_text_processor.decode(outputs[0], skip_special_tokens=True)

    text = "what type of bike is this?"
    inputs = image_text_processor(images=image, text=text, return_tensors="pt")
    outputs = image_text_model.generate(**inputs)
    text_output2 = image_text_processor.decode(outputs[0], skip_special_tokens=True)

    return f'A {text_output1} {text_output2} bike'

def save_uploaded_file(uploaded_file: dict):
    """Save file local and to AWS for future reference
    
    Args:
        uploaded_file: uploaded file attributes
        
    Returns:
        The file path
    """ 
    UPLOAD_DIR = "data/images"
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # save to AWS for future reference
    session = boto3.Session()
    s3 = session.client("s3")
    s3.upload_file(file_path, 'cycle-saviours', uploaded_file.name, 
                ExtraArgs=dict(ContentType='image/png'))

    return file_path
####################################################
# INITIALIZE AGENT
####################################################
structured_tool = StructuredTool.from_function(get_filtered_results)

memory = ConversationBufferMemory(memory_key="chat_history")

conversation_agent = initialize_agent(
    [structured_tool], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)
        
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
####################################################
# STREAMLIT CODE
####################################################
st.title('Find My Bike!')

# add file upload area and dropdown for city
with st.sidebar:
    uploaded_file = st.file_uploader("Please upload an image of your bike (REQUIRED)",
                                    type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.success("File saved successfully.")
    else:
        file_path = None

    city = st.selectbox('Please select a city (REQUIRED)', ['SELECT CITY', 'calgary', 'edmonton', 'vancouver'])

# one for chat and one for search results
col1, col2 = st.columns([0.6, 0.4])

with col2:

    # we are going to direct user input with buttons
    user_input = ""
    if st.session_state['search_button'] == True:
        user_input = "Yes"

    if file_path and city != 'SELECT CITY' and user_input != "Yes":

        # return a description back to the user of the image to verify
        bike_image_analysis = get_image_text(file_path)
        st.session_state['image_description'] = bike_image_analysis
        st.session_state.generated.append(f"You uploaded an image of {bike_image_analysis} does this sound correct")

    if file_path and city != 'SELECT CITY' and user_input == 'Yes':
        st.session_state.past.append('Yes')

        # use langchain agent to run tool
        output = conversation_agent.run(
            f"The user has had their bike stolen from them, please help them find it. DESCRIPTION: {st.session_state['image_description']} LOCATION: {city}. Once you have a final anwser, say 'I have found a list of potential matches for your stolen bike.'"
        )
        st.session_state.generated.append(output)
        st.session_state['pinecone_output'] = get_filtered_results(city)

        # output cards
        with col1:
            if st.session_state['pinecone_output']:
                html_placeholder = '<html><h6>Ads found online that match your image and description</h6>'
                for data in st.session_state['pinecone_output']:

                    listed_date = str(data['list_date']).split('.')[0]

                    html_placeholder += '<div style="background-color: rgba(213, 238, 247, 0.5);' \
                        'border-radius: 10px; padding: 10px; margin-bottom: 20px;><table style="border: none;' \
                        'background-color: lightblue; border-radius: 10px;">'
                    
                    html_placeholder += '<tr style="border: none;">' \
                        f'<h3><a href={data["ad_link"]} style="text-decoration:none;" target="_blank">{data["ad_title"]}</a></h3>' \
                        f'</tr><tr style="border: none;"><strong>Ad Source: </strong>{data["source"]} <br>' \
                        f'<strong>Ad Location: </strong>{data["ad_location"]} <br>' \
                        f'<strong>Ad List Date: </strong>{listed_date[0:4]+"-"+listed_date[4:6]+"-"+listed_date[6:]}<br>' \
                        f'<br></tr><tr style="border: none;"><img src={data["image_link"]} alt=Ad Id: {data["source_id"]}></tr><br></table></div>'

                html_placeholder += '</html>'

                st.markdown(html_placeholder, unsafe_allow_html=True)

    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

    # chat message render
    if st.session_state.get("generated"):
        generated_length = len(st.session_state.generated)
        for i in range(generated_length -1, -1, -1):
            if i == 1 and generated_length < 3:
                with button_col2:
                    search_button = st.button('Yes')
                    st.session_state['search_button'] = True
            message(st.session_state.generated[i], key=str(i))
            if i == 2:
                message(st.session_state.past[0], is_user=True, key=str(i) + "_user")
