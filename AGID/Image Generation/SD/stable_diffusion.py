# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# Image Processing
from PIL import Image

# Image Generation
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Image similarity
from similarityScore import ImageSimilarity
import warnings
warnings.filterwarnings('ignore')

""""
 Generate image using T2I model for all the prompts and store the best image in a Data-Generated folder,  
 Images with greater similarity  than thresholds are stored in Useful-Data folder and similarity for all the images is recorded in a CSV File.
"""

# Arguments 
parser=argparse.ArgumentParser(description='Python Script to Generate Stable Diffusion Images')

parser.add_argument("--excel_path", type=str, help='Pass the path of CSV file (e.g /content/data.csv)')
parser.add_argument('--original_img_dir', type=str, help='Directory path of images (e.g /content/images/)')
parser.add_argument('--output_dir', type=str, help='Directory to store the images')
parser.add_argument('--threshold_clip', type=float, help='Threshold for similarity using CLIP embeddings')
parser.add_argument('--threshold_vgg', type=float, help='Threshold for similarity using VGG embeddings')
parser.add_argument('--gen_model',help='Name of the generation model')
parser.add_argument('--num_images_per_prompt', type=int, default=3, help='Number of images per prompt')
parser.add_argument('--last_tweet_id',type=int,default=0,help='Last processed tweet incase of Failue')

args = parser.parse_args()

# Access the parsed arguments
ORIGINAL_DIR = args.original_img_dir
NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
THRESHOLD_CLIP = args.threshold_clip
THRESHOLD_VGG = args.threshold_vgg


print('Reading the Data!!')
data=pd.read_excel(args.excel_path)
df=data[args.last_tweet_id+1 :]

# Directory to store the Generated Images
OUTPUT_DIR=os.path.join(args.output_dir,f'{args.gen_model}')
all_images= os.path.join(OUTPUT_DIR,'Data-Generated') # Folder to store all the generated images
thresholded_images=os.path.join(OUTPUT_DIR,'Useful-Data') # Folder to generated images post threshold on similarity
score_df = pd.DataFrame(columns=['Image_ID', 'CLIP_Score', 'VGG_Score'])

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(all_images):
    os.makedirs(all_images)

if not os.path.exists(thresholded_images):
    os.makedirs(thresholded_images)


DEVICE = 'cuda' if torch.cuda.is_available else 'cpu' # device
STRENGTH = 0.3 # The noise to add to original image
NUM_INFERENCE_STEPS = 30 # Number of inference steps to the Diffusion Model

print(f'Loading the Models to {DEVICE}')
DIFFUSION_MODEL_PATH = "stabilityai/stable-diffusion-2-1" # Set the model path to load the diffusion model from

scheduler = EulerDiscreteScheduler.from_pretrained(DIFFUSION_MODEL_PATH, subfolder="scheduler")

model = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16)
model = model.to(DEVICE)

model.set_progress_bar_config(disable=True) #To disable progress bar


similarity=ImageSimilarity()

print('Start Generating images')

# model.eval()
try:
  for _,row in tqdm(df.iterrows(),total=len(df), desc="Processing prompts"):
    
    prompt=row['tweetContentProcessed']
    idx = str(row['id'])
    image_path =  ORIGINAL_DIR + idx + '.jpg'

    # Error checking for original image (Checking Path,Image Format)
    if os.path.exists(image_path):
      img=Image.open(image_path)
      if img.mode != 'RGB':
          print(f'Grayscale image found for ID: {idx}')
          continue  
      else: original_image = img
    else:
        print(f'Error: Image file not found for ID: {idx}')

    best_clip_similarity = -1
    best_vgg_similarity = -1

    for j in range(NUM_IMAGES_PER_PROMPT):

      output_image = model(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
      clip_similarity = similarity.get_similarity_score(output_image, original_image, model_name="CLIP")
      vgg_similarity = similarity.get_similarity_score(output_image, original_image, model_name="VGG")

      if clip_similarity > best_clip_similarity and vgg_similarity > best_vgg_similarity:
            best_clip_similarity = clip_similarity
            best_vgg_similarity = vgg_similarity
            best_output_image = output_image


      if clip_similarity > THRESHOLD_CLIP and vgg_similarity > THRESHOLD_VGG:
        output_image.save(f'{thresholded_images}/{idx}.jpg')
        break

    best_output_image.save(f'{all_images}/{idx}.jpg')
    score_df = score_df.append({'Image_ID': idx, 'CLIP_Score': best_clip_similarity, 'VGG_Score': best_vgg_similarity}, ignore_index=True)

except Exception as e:
  print(f'Error: {e}')

finally: 
   score_df.to_csv(OUTPUT_DIR+'/score.csv',index=False) # Saving the score to Output directory 
