import streamlit as st
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Load Arnav's chat history
def load_arnav_chats(file_path, limit=30):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    messages = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4},.*? - (Arnav): (.+)', raw)
    arnav_lines = [msg for _, msg in messages if msg.lower() != "null"]
    last_lines = arnav_lines[-limit:]
    return "\n".join(f"Arnav: {line.strip()}" for line in last_lines if line.strip())

# Load chat examples
chat_examples = load_arnav_chats("chat.txt")

system_prompt = f"""
You're Arnav. Reply like him in a sweet, cute, slightly dramatic style.
Use this history to guide your tone and vocabulary:
{chat_examples}
"""

# Streamlit UI
st.set_page_config(page_title="Chat Like Arnav", page_icon="🤖")
st.title("🤖 ChatLikeArnav")

# Chat input
user_input = st.chat_input("Type a message...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Arnav is typing..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages
            ]
        )

        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)
