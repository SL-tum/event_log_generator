from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
# Example of creating the credentials using an API key:

class WatsonxLLM:

    def __init__(self, model):
        self.model = model

    def chat(self, messages, temperature=0.1, max_tokens=4000):

        response = self.model.chat(messages=messages)

        return response["choices"][0]["message"]["content"]
