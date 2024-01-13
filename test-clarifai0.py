import os

from clarifai.client.model import Model

prompt = "What is tebentafusp?"

openai_api_key = os.getenv('OPENAI_API_KEY')

model_prediction = Model("https://clarifai.com/openai/chat-completion/models/GPT-4").predict_by_bytes(prompt.encode(), input_type="text")

print(model_prediction.outputs[0].data.text.raw)
