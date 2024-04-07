import openai
import os
import re

# Function to generate a response from OpenAI's API
def responseprompt(prompt):
    # Set the API key from an environment variable for security
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    if openai.api_key is None:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Generate a response using the OpenAI chat completion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Specify the model you are using
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.5,
    )
    # Return the message content from the response, if available
    return response['choices'][0]['message']['content'] if 'content' in response['choices'][0]['message'] else ""

# Function to preprocess and extract instructions from the response
def preprocessedresponse(prompt):
    # Get the response from the API
    res = responseprompt(prompt)
    # Initialize lists to hold the instructions and the total response
    inslist = []
    totlist = []
    
    # Split the response by newlines and remove empty lines
    totlist = list(filter(lambda x: x.strip(), res.split("\n")))
    
    # Find the starting index for the instructions and extract them
    start_index = totlist.index("Instructions:") + 1 if "Instructions:" in totlist else 0
    inslist = totlist[start_index:]
    
    # Clean up the instruction strings
    inslist = [re.sub(r'^\d+', '', string) for string in inslist]  # Remove leading numbers
    inslist = [string.lstrip(" .") for string in inslist]          # Strip leading spaces and dots
    inslist = [string.rstrip(". ") for string in inslist]          # Strip trailing dots and spaces
    
    return inslist

# Make sure to set your OPENAI_API_KEY in your environment before running this script.
# You can test the function by uncommenting the lines below:
# prompt_text = "Tell me a joke."
# print(responseprompt(prompt_text))
