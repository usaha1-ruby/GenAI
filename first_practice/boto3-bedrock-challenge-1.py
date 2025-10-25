##write a code using LLM to translate english to french using boto3 and bedrock

import boto3
import json

##Create the bedrock run-time client
bedrock_client = boto3.client("bedrock-runtime")

##Model specification
#model_id = "eu.meta.llama3-2-1b-instruct-v1:0"
model_id = "eu.meta.llama3-2-3b-instruct-v1:0"
temparature = 0.5
top_p = 0.9
max_gen_len = 1000

##Setting the prompt
prompt = "Translate to french: 'Learning about Generative AI is fun and exciting using Amazon Bedrock'"
formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for translating language<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

##Format the request payload using the model's native structure.
native_request = {
    "prompt": formatted_prompt,
    "temperature": temparature,
    "top_p": top_p,
    "max_gen_len": max_gen_len
}

##Convert the native request to JSON to invoke the model
request = json.dumps(native_request)

##Invoke the model
try:
    response = bedrock_client.invoke_model(
        body=request,
        modelId=model_id
    )
except bedrock_client.exceptions.ModelErrorException as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

##Decode the response body, Parse and display the output
response_body = json.loads(response.get('body').read().decode('utf-8'))
output = response_body["generation"]
print(output)