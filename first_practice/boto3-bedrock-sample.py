#Imports
import boto3
import json

#Create the bedrock run-time client
bedrock_client = boto3.client("bedrock-runtime")

#Setting the prompt
prompt_data = """command: You are a traveler. Write a blog about travelling Amsterdamin in summer.

Blog:
"""

#Model specification
model_id = "amazon.titan-text-express-v1"
content_type = 'application/json'
accept ='application/json'

#Configuring parameters to invoke the model
body = json.dumps({
        "inputText": prompt_data,
        "textGenerationConfig": {
            "temperature": 0.8,  
            "topP": 1.0,
            "maxTokenCount": 1000,
            "stopSequences": []
        }
    }
)

#Invoke the model
response = bedrock_client.invoke_model(
    body=body,
    contentType=content_type,
    accept=accept,
    modelId=model_id
)

#Parsing and displaying the output
response_body = json.loads(response.get('body').read().decode('utf-8'))
output = response_body.get('results')[0].get('outputText', '')
print(output)

## two major areas which user/developer should learn
# prompt engineering
# How to select correct FM