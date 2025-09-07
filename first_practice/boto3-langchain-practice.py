#Imports
import boto3
#from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate

#Create the bedrock client
#Not required....

#Create the llm
llm = ChatBedrockConverse(
    model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.6,
    top_p=1.0,
    max_tokens=1000
)

#Generate the response
prompt = [
    (
        "system",
        "You are a helpful assistant that can write email. Assist with user request.",
    ),
    ("human", """Write an email from Mark the hiring manager to Sarah the candidate to schedule an interview for 
    the Software Engineer position. The interview is scheduled for next Monday at 10 AM. 
    Please include a polite greeting and closing."""),
]
response = llm.invoke(prompt)

#Display the result
print(response.content)

#prompt = ChatPromptTemplate.from_template("Write me a haiku about {topic}")
#print(prompt)


##Key take asway
#1. learn langchain