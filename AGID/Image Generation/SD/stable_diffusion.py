# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# Image Processing
from PIL import Image
import clip

# Image Generation
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Image similarity
from similarityScore import ImageSimilarity


# Arguments 
parser=argparse.ArgumentParser(description='Python Script to Generate Stable Diffusion Images')

parser.add_argument("--csv_path", type=str, help='Pass the path of CSV file (e.g /content/data.csv)')
parser.add_argument('--original_dir', type=str, help='Directory path of images (e.g /content/images/)')
parser.add_argument('--threshold_clip', type=float, help='Threshold for similarity using CLIP embeddings')
parser.add_argument('--threshold_vgg', type=float, help='Threshold for similarity using VGG embeddings')
parser.add_argument('--num_images_per_prompt', type=int, default=3, help='Number of images per prompt')
parser.add_argument('--last_tweet_id',type=int,default=0,help='Last processed tweet incase of Failue')

args = parser.parse_args()

# Access the parsed arguments
ORIGINAL_DIR = args.original_dir
NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
THRESHOLD_CLIP = args.threshold_clip
THRESHOLD_VGG = args.threshold_vgg


print('Reading the Data!!')
df=pd.read_csv(args.csv_path)
prompts=list(df.tweetContentProcessed[args.last_tweet_id])

# Directory to store the Images
OUTPUT_DIR='Data-Generated/'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu' # device
STRENGTH = 0.3 # The noise to add to original image
NUM_INFERENCE_STEPS = 30 # Number of inference steps to the Diffusion Model

print(f'Loading the Models to {DEVICE}')
DIFFUSION_MODEL_PATH = "stabilityai/stable-diffusion-2-1" # Set the model path to load the diffusion model from
CLIP,preprocess = clip.load("ViT-B/32", device=DEVICE)

scheduler = EulerDiscreteScheduler.from_pretrained(DIFFUSION_MODEL_PATH, subfolder="scheduler")

model = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16)
model = model.to(DEVICE)

model.set_progress_bar_config(disable=True) #To disable progress bar


similarity=ImageSimilarity()

print('Start Generating images')

try:
  for i,prompt in enumerate(tqdm(prompts)):
    max_score = 0
    idx = str(df.id[i])
    image_path =  ORIGINAL_DIR + idx + '.jpg'

    if os.path.exists(image_path): original_image = Image.open(image_path)
    else: print(f'Error: Image file not found for ID: {idx}')

    for j in range(NUM_IMAGES_PER_PROMPT):
      output_image = model(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
      clip_similarity = similarity.get_similarity_score(output_image, original_image, model_name="CLIP")
      vgg_similarity = similarity.get_similarity_score(output_image, original_image, model_name="VGG")

      if clip_similarity > THRESHOLD_CLIP and vgg_similarity > THRESHOLD_VGG:
        output_image.save(f'{OUTPUT_DIR}/{idx}.jpg')
        break

except Exception as e:
  print(f'Error: {e}')
