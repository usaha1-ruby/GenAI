#IMPORTS
import boto3
from langchain_aws import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# Configure Streamlit app
st.set_page_config(page_title="Social Media Training Q&A Assistant", page_icon=":books:")
st.title(":books: Social Media Q&A Assistant")

@st.cache_resource
def configure_anthropic_model(temp=1, top_p=1, max_token=1000):
    """This function to configure anthropic model"""
    llm = ChatBedrockConverse(
        model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=temp, #0.6
        top_p=top_p,
        max_tokens=max_token
    )
    return llm

@st.cache_resource
def embed_data(filename):
    """
    This function to embed the data using bedrock embeddings and in-memory store in FAISS vectorstore
    """
    bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="eu-west-1"
        )
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss  = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

def vector_search(vectorstore, query):
    """method for semantic search"""
    docs = vectorstore.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info += doc[0].page_content + "\n"
    return info

#configure LLM
expected_temp=0.4
tokens=512
llm = configure_anthropic_model(temp=expected_temp, max_token=tokens)

#create vector store from the document
filename = "./first_practice/practice-basic-rag/knwoledge_doc/social-media-training.pdf"
vectorstore = embed_data(filename)

# Setup memory for chat history
chat_history = StreamlitChatMessageHistory(key="langchain_messages")
if len(chat_history.messages) == 0:
    chat_history.add_ai_message("Hello! Ask me anything about the Social Media Training document. Type 'exit' to quit.")

#Create prompt template
my_promprt = """
        system: You are a conversational assistant designed to help answer questions from an employee. 
        You should reply to the human's question using the information provided below. Include all 
        relevant information but keep your answers short. Only answer the question. Do not say things 
        like "according to the training or handbook or according to the information provided..."
        
        human:
        <Information>
        {info}
        </Information>
    
        {input}
        
        assistant:
"""

#Configure prompt template
prompt = PromptTemplate(
    input_variables=["info", "input"],
    template=my_promprt
)

#Create llm chain
question_chain = prompt | llm

# Render chat messages from StreamlitChatMessageHistory
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

#Get user question, peform similarity search, invoke model and return result
if prompt := st.chat_input("Please ask your question about social media training"):
    if prompt.lower() == "exit":
        st.write("Exiting the program.")
        st.stop
    else:
        #perform vector search
        info = vector_search(vectorstore, prompt)
        
        #invoke model
        response = question_chain.invoke({
            "info": info,
            "input": prompt
        })

        # updating chat hostory and Display user message
        chat_history.add_user_message(prompt)
        st.chat_message("user").write(prompt)

        # Display assistant response
        chat_history.add_ai_message(response.content)
        st.chat_message("ai").write(response.content) #assistant
