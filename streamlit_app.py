import streamlit as st
import os
import boto3
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import current_time
import create_booking
import delete_booking

st.set_page_config(page_title="Restaurant Assistant", page_icon="🍽️")

# Sidebar for AWS credentials
with st.sidebar:
    st.header("⚙️ Configuration")
    aws_access_key = st.text_input("AWS Access Key", type="password")
    aws_secret_key = st.text_input("AWS Secret Key", type="password")
    
    if aws_access_key and aws_secret_key:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        st.success("✅ AWS Credentials Set")

st.title("🍽️ Restaurant Assistant")
st.markdown("Ask me about restaurants or make a reservation!")

# Initialize agent (same code as before)
# ... [copy your agent initialization code]

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent(prompt)
            st.markdown(str(response))
    
    st.session_state.messages.append({"role": "assistant", "content": str(response)})