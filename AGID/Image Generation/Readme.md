
# Stable Diffusion

 ## Step1.) First install the requirements in an virtual environment   

 Install the requirements using the command
 ```pip install -r requirements.txt```

## Step 2: Organize Your Files
Place the stable_diffusion.py and similarityScore.py scripts in the same directory for easy access.

## Step 3: Run the Model Script
To generate images from text prompts, run the SD.py script in your terminal or command prompt. Use the following command with the specified arguments:

--excel_path: The path to the CSV file containing your data, where the CSV should have an 'id' column and tweet content in the 'tweetContentProcessed' column.

--original_img_dir: The path to the directory where original images are stored.  

--output_dir: The path to the output directory

--API_TOKEN : API Token of the model (Not Required for Stable diffusion)

--num_images_per_prompt: The number of images to generate per text prompt.

--threshold_clip: The Clip threshold, which sets a similarity threshold between images using Clip embeddings. Images with a similarity below this threshold will be discarded.

--threshold_vgg: The VGG threshold, which sets a similarity threshold between images using VGG embeddings. Images with a similarity below this threshold will be discarded.

--gen_model: Name of the generation model 

--last_tweet_id: Last processed tweet before the model stopped running

```
python model.py --excel_path AIISC/NYT.csv --original_img_dir AIISC/tweetImages/ --API_TOKEN " " --output_dir /  --threshold_clip 0.40 --threshold_vgg 0.4 --gen_model " " --num_images_per_prompt 2
```
