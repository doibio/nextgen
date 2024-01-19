import sys
from clarifai.client.model import Model
import os

openai_api_key = os.getenv('OPENAI_API_KEY')

pre_prompt = "How would I write a sentence that would highlight the seriousness and complexity of uveal melanoma following these criteria : "

inference_params = dict(temperature=0.2, max_tokens=100, api_key = openai_api_key)

def print_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sentences = content.split('\n\n')

    for sentence in sentences:
        prompt = pre_prompt + sentence.strip()
        model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
        
        print(model_prediction.outputs[0].data.text.raw, flush=True)
        
        print("------------------------------")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print_sentences_from_file(file_path)
    else:
        print("Please provide a file path as an argument.")
