import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="sk-abcdef1234567890abcdef1234567890abcdef12")

# Store conversation in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("💬 Chatbot with OpenAI")

# User input via chat box
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call OpenAI model
    completion = client.chat.completions.create(
        model="gpt-4o-mini",   # You can also use "gpt-4" or "gpt-3.5-turbo"
        messages=st.session_state.messages,
        max_tokens=500
    )

    # Extract assistant reply
    response = completion.choices[0].message.content

    # Add to conversation
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
