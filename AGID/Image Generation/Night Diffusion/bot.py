import requests
import json
import os

# Your API endpoint and API key
API_ENDPOINT = 'https://stablediffusionapi.com/api/v4/dreambooth'
API_KEY = 'sH36b0MEzMnUCQmI6gMqL5Qbbgl32uB2gVWUSIjk4ZsmnqHudwHMj9pxlsBQ' 

# Path to the text file containing prompts, using a relative path to the parent directory
PROMPT_FILE_PATH = '../prompts.txt'  # Assumes prompts.txt is in the parent directory

# Folder to save output images, use a relative path to the current directory
OUTPUT_FOLDER = 'Output'

# Get the current directory where the script is located
current_dir = os.path.dirname(__file__)

# Combine the current directory with the relative paths
PROMPT_FILE_PATH = os.path.normpath(os.path.join(current_dir, PROMPT_FILE_PATH))
OUTPUT_FOLDER = os.path.normpath(os.path.join(current_dir, OUTPUT_FOLDER))

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Read prompts from the text file
def read_prompts_from_file(file_path):
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    return prompts

# Function to make an API request and save the resulting image
def generate_image_from_prompt(prompt, output_file_path):
    payload = {
        "key": API_KEY,
        "model_id": "sdxl",
        "prompt": prompt,
        "negative_prompt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "30",
        "safety_checker": "no",
        "enhance_prompt": "yes",
        "guidance_scale": 7.5,
        "tomesd": "yes",
        "use_karras_sigmas": "yes",
        "scheduler": "UniPCMultistepScheduler"
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = json.loads(response.text)
        if 'output' in response_data:
            image_url = response_data['output'][0]
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                with open(output_file_path, 'wb') as file:
                    file.write(image_response.content)
                print(f'Saved image: {output_file_path}')
            else:
                print(f'Error downloading image: {image_url}')
        else:
            print(f'No "output" field in API response for prompt: {prompt}')
    else:
        print(f'Error generating image for prompt: {prompt}')

# Read prompts from the file
prompts = read_prompts_from_file(PROMPT_FILE_PATH)

# Generate images for each prompt
for idx, prompt in enumerate(prompts, start=1):
    output_file_path = os.path.join(OUTPUT_FOLDER, f'image_{idx}.png')  # Add .png extension
    generate_image_from_prompt(prompt, output_file_path)
