from clarifai.client.model import Model
import os

openai_api_key = os.getenv('OPENAI_API_KEY')

prompt = "I am reading a paper review about uveal melanoma.  What would be the purpose of the authors in writing the following sentence?"

inference_params = dict(temperature=0.2, max_tokens=100, api_key = openai_api_key)

model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

print(model_prediction.outputs[0].data.text.raw)

