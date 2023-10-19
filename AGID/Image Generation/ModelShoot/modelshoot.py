# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import requests
import io
import time

# Image Processing
from PIL import Image
import clip

# Image Generation
from transformers import CLIPTokenizer
import torch

# Image similarity
from Simililarity_Score import ImageSimilarity


# Arguments
parser=argparse.ArgumentParser(description='Python Script to Generate Stable Diffusion Images')

parser.add_argument("--CSV_PATH",type=str,help='Pass the path of CSV file (e.g /content/data.csv)')
parser.add_argument('--ORIGINAL_DIR',type=str,help='Directory path of Images (e.g /content/images/)')
parser.add_argument('--API_TOKEN',type=str,help='API token for Modelshoot from HuggingFace')
parser.add_argument('--THRESHOLD_CLIP',type=float,help='Threshold for similarity using CLIP Embeddings')
parser.add_argument('--THRESHOLD_VGG',type=float,help='Threshold for similarity using VGG Embeddings')
parser.add_argument('--NUM_IMAGES_PER_PROMPT',type=int,default=2,help='Number of Images per prompt')

args = parser.parse_args()

# Access the parsed arguments
csv_path = args.CSV_PATH
ORIGINAL_DIR = args.ORIGINAL_DIR
NUM_IMAGES_PER_PROMPT = args.NUM_IMAGES_PER_PROMPT
THRESHOLD_CLIP = args.THRESHOLD_CLIP
THRESHOLD_VGG = args.THRESHOLD_VGG
API_TOKEN=args.API_TOKEN

print('Reading the Data!!')
df=pd.read_csv(csv_path)
prompts=list(df.tweetContentProcessed)

# Directory to store the Images
OUTPUT_DIR='Data-Generated/'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu' # device
STRENGTH = 0.3 # The noise to add to original image
NUM_INFERENCE_STEPS = 30 # Number of inference steps to the Diffusion Model

print(f'Loading the Models to {DEVICE}')

API_URL = "https://api-inference.huggingface.co/models/wavymulder/modelshoot"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

CLIP,preprocess = clip.load("ViT-B/32", device=DEVICE)


similarity=ImageSimilarity()

print('Start Generating images')
time.sleep(15)
try:
  for i,prompt in enumerate(tqdm(prompts)):
    time.sleep(2)
    max_score = 0
    idx = str(df.id[i])
    image_path =  ORIGINAL_DIR + idx + '.jpg'
    original_image = Image.open(image_path)

    for j in range(NUM_IMAGES_PER_PROMPT):
        
        try:
            image_bytes = query({"inputs": f"{prompt}"})
            output_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            print(image_bytes)
            time.sleep(3)  # Wait for 3 seconds
            break
        clip_similarity = similarity.get_similarity_score(output_image, original_image, model_name="CLIP")
        vgg_similarity = similarity.get_similarity_score(output_image, original_image, model_name="VGG")

        if clip_similarity > THRESHOLD_CLIP and vgg_similarity > THRESHOLD_VGG:
            output_image.save(f'{OUTPUT_DIR}/{idx}.jpg')
            break

except Exception as e:
  print(f'Error:{e}')



