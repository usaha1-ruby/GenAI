#IMPORTS
import boto3
from langchain_aws import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_aws.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader


def configure_anthropic_model(temp=1, top_p=1, max_token=1000):
    """This function to configure anthropic model"""
    llm = ChatBedrockConverse(
        model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=temp, #0.6
        top_p=top_p,
        max_tokens=max_token
    )
    return llm

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

#Get question, peform similarity search, invoke model and return result
while True:
    question = input("Please ask your question about social media training:\n")
    if question.lower() == "exit":
        print("Exiting the program.")
        break
    
    #perform vector search
    info = vector_search(vectorstore, question)
    
    #invoke model
    response = question_chain.invoke({
        "info": info,
        "input": question
    })

    print(f"Answer: {response.content}\n")