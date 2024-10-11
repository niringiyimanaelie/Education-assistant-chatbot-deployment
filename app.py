import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import textwrap
import streamlit as st
import pandas as pd

### Keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

### Establish Pinecone Client and Connection
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('chatbot1')

### Establish OpenAi Client
openai.api_key = openai_api_key
client = openai.OpenAI()

### Get embeddings
def get_embeddings(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

## Get context
def get_contexts(query, embed_model='text-embedding-ada-002', k=5):
    query_embeddings = get_embeddings(query, model=embed_model)
    pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
    context = [item['metadata']['text'] for item in pinecone_response['matches']]
    #source= pd.DataFrame(context)
    #source.to_csv('source.csv')
    return context, query

### Augmented Prompt
def augmented_query(user_query, embed_model='text-embedding-ada-002', k=5):
    context, query = get_contexts(user_query, embed_model=embed_model, k=k)
    return "\n\n---\n\n".join(context) + "\n\n---\n\n" + query

### Ask GPT
def ask_gpt(system_prompt, user_prompt, model="gpt-3.5-turbo", temp=0.7):
    temperature_ = temp
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature_,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)

### Education ChatBot
def Education_ChatBot(query):
    embed_model = 'text-embedding-ada-002'
    primer = """
    You are an Education Assistant AI. Your role is to facilitate real-time Q&A, polls, quizzes, and class discussions, 
    making lessons more interactive and engaging for students. You help teachers gauge student understanding by summarizing 
    discussions, create quizzes, and highlighting areas where students may struggle.

    You offer:
    - Clear and concise answers to real-time questions.
    - Summarization of discussions to help teachers review key points.
    - Create Quizes

    If the information needed to answer a question is not available, respond with: 
    'I do not know based on the information provided.'

    Your tone is helpful, supportive, and focused on creating an interactive and effective learning environment.
    """

    llm_model = 'gpt-3.5-turbo'
    user_prompt = augmented_query(query, embed_model)
    return ask_gpt(primer, user_prompt, model=llm_model)


### Streamlit Interface ###
st.title("Education ChatBot")

# Get user input
user_input = st.text_input("Ask a question:", "")

if st.button("Submit"):
    if user_input:
        response = Education_ChatBot(user_input)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a question or request.")
