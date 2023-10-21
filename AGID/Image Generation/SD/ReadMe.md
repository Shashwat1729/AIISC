
# Stable Diffusion

 ## Step1.) First install the requirements in an virtual environment   

 Install the requirements using the command
 ```pip install -r requirements.txt```

## Step 2: Organize Your Files
Place the stable_diffusion.py and similarity_score.py scripts in the same directory for easy access.

## Step 3: Run the Stable Diffusion Script
To generate images from text prompts, run the SD.py script in your terminal or command prompt. Use the following command with the specified arguments:

--CSV_PATH: The path to the CSV file containing your data, where the CSV should have an 'id' column and tweet content in the 'tweetContentProcessed' column.

--ORIGINAL_DIR: The path to the directory where original images are stored.

--NUM_IMAGES_PER_PROMPT: The number of images to generate per text prompt.

--THRESHOLD_CLIP: The Clip threshold, which sets a similarity threshold between images using Clip embeddings. Images with a similarity below this threshold will be discarded.

--THRESHOLD_VGG: The VGG threshold, which sets a similarity threshold between images using VGG embeddings. Images with a similarity below this threshold will be discarded.

-- last_tweet_id: Last processed tweet before the model stopped running

```
python stable_diffusion.py -csv_path /NYT.csv --original_dir tweetImages/ --threshold_clip 0.40 --threshold_vgg 0.4 number_images_per_prompt 3 --last_tweet_id 520
```
