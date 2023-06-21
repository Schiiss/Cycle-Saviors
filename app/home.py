"""Cycle Saviour Home App Page"""
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
import pinecone
import os

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

# this contains the chat interface
with col2:
    user_input = st.text_input('Start a chat with our assistant!',
                               key='input',
                               label_visibility='visible',
                               placeholder='My bike was stolen, help me find it!')

    if user_input:
        if st.session_state['conversation_history']:
            # placeholder
            result = None
        else:
            # placeholder
            result = None

        st.session_state.user_query_history.append(user_input)
        st.session_state.conversation_history.append([(user_input, result)])
        st.session_state.ai_response_history.append(result)

    # if ai responded then print out dialouge
    if st.session_state['ai_response_history']:

        for i in range(len(st.session_state['ai_response_history'])-1, -1, -1):
            message('AI Response Placholder', key=str(i))
            message(st.session_state['user_query_history'][i], is_user=True, key=str(i) + '_user')
