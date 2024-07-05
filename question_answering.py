import logging

import streamlit as st
import openai
import os
import re
import bcrypt
from openai import OpenAI

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

logging.basicConfig(level=ANALYSIS_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(ANALYSIS_LEVEL)

from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters
from regulations_rag.corpus_chat import CorpusChat

from gdpr_rag.corpus_index import GDPRCorpusIndex

# App title - Must be first Streamlit command
st.set_page_config(page_title="💬 GDPR Question Answering", layout="wide")


if 'user_id' not in st.session_state:
    st.session_state['user_id'] = "test_user"


### Password
# if "password_correct" not in st.session_state.keys():
#     st.session_state["password_correct"] = True

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        pwd_raw = st.session_state['password']
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and bcrypt.checkpw(
            pwd_raw.encode(),
            st.secrets.passwords[st.session_state["username"]].encode(),
        ):
            st.session_state["password_correct"] = True
            logger.log(ANALYSIS_LEVEL, f"New questions From: {st.session_state['username']}")
            del st.session_state["password"]  # Don't store the username or password.
            del pwd_raw
            st.session_state["user_id"] = st.session_state["username"] 
            del st.session_state["username"]
            
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()


def load_data():
    logger.log(ANALYSIS_LEVEL, f"*** Loading data for {st.session_state['user_id']}. Should only happen once")
    logger.debug(f'--> cache_resource called again to reload data')
    with st.spinner(text="Loading the gdpr documents and index - hang tight! This should take 5 seconds."):

        key = st.secrets["index"]["decryption_key"]
        corpus_index = GDPRCorpusIndex(key)

        rerank_algo = RerankAlgos.LLM
        rerank_algo.params["openai_client"] = st.session_state['openai_client']
        rerank_algo.params["model_to_use"] = st.session_state['selected_model']
        rerank_algo.params["user_type"] = corpus_index.user_type
        rerank_algo.params["corpus_description"] = corpus_index.corpus_description
        rerank_algo.params["final_token_cap"] = 5000 # can go large with the new models

        embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
        chat_parameters = ChatParameters(chat_model = "gpt-4o", temperature = 0, max_tokens = 500)
        
        chat = CorpusChat(openai_client = st.session_state['openai_client'],
                          embedding_parameters = embedding_parameters, 
                          chat_parameters = chat_parameters, 
                          corpus_index = corpus_index,
                          rerank_algo = rerank_algo,   
                          user_name_for_logging=st.session_state["user_id"])



        return chat


if 'openai_api' not in st.session_state:
    st.session_state['openai_client'] = OpenAI(api_key = st.secrets['openai']['OPENAI_API_KEY'])

if 'selected_model' not in st.session_state.keys():
    #st.session_state['model_options'] = ['gpt-4-0125-preview', 'gpt-4', 'gpt-3.5-turbo']
    st.session_state['model_options'] = ['gpt-4o']
    st.session_state['selected_model'] = 'gpt-4o'
    st.session_state['selected_model_previous'] = 'gpt-4o'

if 'chat' not in st.session_state:
    st.session_state['chat'] = load_data()
    st.session_state['chat'].chat_parameters.model = st.session_state['selected_model']



st.title('GDPR: Question Answering')


st.write(f"I am a bot designed to answer questions based on {st.session_state['chat'].index.corpus_description}. How can I assist today?")

temperature = 0.0
max_length = 1000 # Note. If you increase this, you need to amend the two instances of the following lines of code in chat_bot.py

# Credentials
# with st.sidebar:

    #st.subheader('Models and parameters')
        
    # st.session_state['selected_model'] = st.sidebar.selectbox('Choose a model', st.session_state['model_options'], key='user_selected_model')
    # if st.session_state['selected_model'] != st.session_state['selected_model_previous']:
    #     st.session_state['selected_model_previous'] = st.session_state['selected_model']
    #     st.session_state['chat'].chat_parameters.model = st.session_state['selected_model']
        # logger.log(ANALYSIS_LEVEL, f"{st.session_state['user_id']} changed model and is now using {st.session_state['selected_model']}")


        #   if (model_to_use == "gpt-3.5-turbo" or model_to_use == "gpt-4") and total_tokens > 3500 and model_to_use!="gpt-3.5-turbo-16k":
        #                     logger.warning("!!! NOTE !!! You have a very long prompt. Switching to the gpt-3.5-turbo-16k model")
        #                     model_to_use = "gpt-3.5-turbo-16k"    
        # Because the 'hard coded' number 3500 plus this max_lenght gets very close to the default model's token limit
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=2048, value=512, step=8)
    # temperature = st.sidebar.slider('temperature', min_value=0.00, max_value=2.0, value=0.0, step=0.01)
    # st.divider()
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    logger.debug("Adding \'messages\' to keys")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 

# Display or clear chat messages
# https://discuss.streamlit.io/t/chat-message-assistant-component-getting-pushed-into-user-message/57231
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            ############################################################################
            response = message["content"]
            # Split the answer into two parts: before "Reference:" and the references part
            parts = re.split(r'Reference:\s*', response, maxsplit=1)
            # Extract the text before "Reference:"
            answer_without_references = parts[0].strip()
            st.markdown(answer_without_references)

            if response.strip() in st.session_state['chat'].references:
                all_references = st.session_state['chat'].references[response.strip()]

                # Extract the references part and split into lines
                if len(parts) > 1:
                    references = parts[1].strip().split('\n')
                else:
                    references = []
                counter = 0
                for reference in references:
                    with st.expander(reference.strip()):
                        st.markdown(all_references.iloc[counter]['text'])
                    counter = counter + 1

            ############################################################################


        else:
            st.write(message["content"])

def clear_chat_history():
    logger.debug("Clearing \'messages\'")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    logger.debug(f"st.chat_input() called. Value returned is: {prompt}")        
    if prompt is not None and prompt != "":
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            #placeholder = st.empty()

            with st.spinner("Thinking..."):
                logger.debug(f"Making call to GDPR with prompt: {prompt}")
                
                st.session_state['chat'].user_provides_input(prompt)

                ############################################################################
                response = st.session_state['chat'].messages[-1]["content"]
                # logger.log(ANALYSIS_LEVEL, f"Response to {st.session_state['user_id']}: {response}")

                # Split the answer into two parts: before "Reference:" and the references part
                parts = re.split(r'Reference:\s*', response, maxsplit=1)
                # Extract the text before "Reference:"
                answer_without_references = parts[0].strip()
                st.markdown(answer_without_references)

                if response.strip() in st.session_state['chat'].references:
                    all_references = st.session_state['chat'].references[response.strip()]

                    # Extract the references part and split into lines
                    if len(parts) > 1:
                        references = parts[1].strip().split('\n')
                    else:
                        references = []
                    counter = 0
                    for reference in references:
                        with st.expander(reference.strip()):
                            st.markdown(all_references.iloc[counter]['text'])
                        counter = counter + 1

                ############################################################################


                logger.debug(f"Response received")
                logger.debug(f"Text Returned from GDPR chat: {response}")
            st.session_state['messages'].append({"role": "assistant", "content": response})
            logger.debug("Response added the the queue")
    
