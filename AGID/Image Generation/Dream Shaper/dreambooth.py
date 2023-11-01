
# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import requests
import io
import json

# Image Processing
from PIL import Image

# Image similarity
from similarityScore import ImageSimilarity


# Arguments
parser=argparse.ArgumentParser(description='Python Script to Generate Stable Diffusion Images')


parser.add_argument("--excel_path", type=str, help='Pass the path of CSV file (e.g /content/data.csv)')
parser.add_argument('--original_img_dir', type=str, help='Directory path of images (e.g /content/images/)')
parser.add_argument('--API_TOKEN',help='API token for Modelshoot from HuggingFace')
parser.add_argument('--output_dir', type=str, help='Directory to store the images')
parser.add_argument('--threshold_clip', type=float, help='Threshold for similarity using CLIP embeddings')
parser.add_argument('--threshold_vgg', type=float, help='Threshold for similarity using VGG embeddings')
parser.add_argument('--gen_model',help='Name of the generation model')
parser.add_argument('--num_images_per_prompt', type=int, default=2, help='Number of images per prompt')
parser.add_argument('--last_tweet_id',type=int,default=0,help='Last processed tweet incase of Failue')


args = parser.parse_args()

# Access the parsed arguments
ORIGINAL_DIR = args.original_img_dir
NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
THRESHOLD_CLIP = args.threshold_clip
THRESHOLD_VGG = args.threshold_vgg
API_TOKEN= str(args.API_TOKEN)


print('Reading the Data!!')
data=pd.read_csv(args.excel_path)
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

API_URL = "https://stablediffusionapi.com/api/v4/dreambooth"

def query(prompt):
    payload = {
        "key": API_TOKEN,
        "model_id": "dream-shaper-8797",
        "prompt": f"{prompt}",
        "negative_prompt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "width": "512",
        "height": "512",
        "samples": f"{NUM_IMAGES_PER_PROMPT}",
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

    response = requests.post(API_URL, headers=headers, json=payload)
    return response


similarity=ImageSimilarity()

print('Start Generating images')

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
        continue

    response=query(prompt)
    if response.status_code == 200:
        response_data = json.loads(response.text)
        print(response_data)
    else: print('Image is not generated!')

    best_clip_similarity = -1
    best_vgg_similarity = -1

    for i in range(NUM_IMAGES_PER_PROMPT):
       
       image_resp = requests.get(response_data['output'][i])
       output_image = Image.open(io.BytesIO(image_resp.content))

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


finally: score_df.to_csv(OUTPUT_DIR+'/score.csv',index=False) # Saving the score to Output directory 

    
