import os
import requests
import time

from dotenv import load_dotenv
import streamlit as st

os.environ["NO_PROXY"] = "172.16.87.91"

load_dotenv(".env")
RAG_CORE_URL = os.getenv("RAG_CORE_URL")

USER_ID = "user1"

def response_generator(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def call_api(user_id: str, query: str):
    response = requests.post(
        url=f"{RAG_CORE_URL}/query?query={query}&user_id={user_id}",
        stream=True,
        timeout=60,
    )
    if response.headers.get('Content-Type') == 'application/json':
        text = response.json()["text"]
        for token in response_generator(text):
            yield token
    else:
        stream = response.iter_content(chunk_size=1, decode_unicode=True)
        for token in stream: 
            time.sleep(0.008)
            yield token.decode("utf-8")

def store_chat(user_id: str, query: str, response: str):
    response = requests.post(
        url=f"{RAG_CORE_URL}/add_message_pair?query={query}&response={response}&user_id={user_id}",
        timeout=60,
    )
    if response.status_code != 200:
        raise ValueError(f"Error storing chat: {response.text}")

# Get chat history from the server
chat_history = requests.get(f"{RAG_CORE_URL}/get_chat_history?user_id={USER_ID}").json()

# Initialize the chat history in the session state
if "chat_history" not in st.session_state and not chat_history:
    st.session_state.chat_history = []
else:
    st.session_state.chat_history = chat_history

with st.chat_message("assistant"):
    initial_message = "Hi dad, do you need any help from me?"
    if st.session_state.chat_history:
        st.markdown(initial_message)
    else:
        st.write_stream(response_generator(initial_message))

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# For accepting user input
if prompt := st.chat_input("You wanna say something to me dad?"):
    # Add the new message to the chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(call_api(USER_ID, prompt))
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Store the chat history
    store_chat(USER_ID, prompt, response)
