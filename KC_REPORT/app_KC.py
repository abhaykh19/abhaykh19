import openai
import streamlit as st
import openai
import streamlit as st
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, GPTListIndex, LLMPredictor, PromptHelper, ServiceContext
import json
from gpt_index.response.schema import Response
import json


# pip install streamlit-chat
from streamlit_chat import message

openai.api_key = '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'

# Create a Vector index
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, GPTListIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from langchain import OpenAI
import sys
import os

os.environ['OPENAI_API_KEY'] = "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"


path=''
max_input = 4096
tokens = 200
chunk_size = 600  # for LLM, we need to define chunk size
max_chunk_overlap = 20

# define prompt
promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

# define LLM
llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=tokens))

# load data
docs = SimpleDirectoryReader(path).load_data()

service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=promptHelper)
vectorIndex = GPTSimpleVectorIndex.from_documents(documents=docs, service_context=service_context)

vectorIndex.save_to_disk('vectorIndex_KC.json')



with st.expander("Disclaimer"):
    st.write("This is a disclaimer. The content of this application is a part of experimentation. The model is trained on Kimberly-Clark Year-End 2022 Results And 2023 Outlook online release.")
    st.write("check out this for the annual report [link](https://investor.kimberly-clark.com/news-releases/news-release-details/kimberly-clark-announces-year-end-2022-results-and-2023-outlook)")


#Creating the chatbot interface
st.title("Your Customized Chatty Bot")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

user_input = get_text()

def ask_anything_1(vectorIndex,input_var):
  vIndex = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
#  input_var = input("Ask Question Related to the data trained: ")
  response = vIndex.query(input_var,response_mode="compact")
  return response


if user_input:
    vectorIndex = 'vectorIndex_KC.json'
    output = str(ask_anything_1(vectorIndex,user_input))
    print(type(output))
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
