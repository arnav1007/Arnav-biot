import streamlit as st
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest

# Load OpenAI key
load_dotenv()
client = OpenAI()

# Init Qdrant client
qdrant = QdrantClient(":memory:")  # Use same memory or persistent connection
collection_name = "arnav-chats"

# UI Setup
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
        # Step 1: Embed user query
        query_vector = client.embeddings.create(
            input=user_input, model="text-embedding-3-small"
        ).data[0].embedding

        # Step 2: Search Qdrant
        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )
        retrieved_texts = "\n".join([hit.payload["text"] for hit in search_results])

        # Step 3: Build prompt
        system_prompt = f"""
You are Arnav. Speak like him — sweet, cute, slightly dramatic, emotional.
Use this chat memory to mimic his voice:
{retrieved_texts}
"""

        # Step 4: Generate response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages,
            ]
        )

        reply = response.choices[0].message.content.strip()
        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)
