
### Step1.) First install the requirements in an virtual environment   
### Step2.)Have the SD.py, similairyt_score.py scripts in the same directory   
### Step3.) Run the SD.py to generate images in terminal

Code expects the CSV file to have an id column, and tweets in tweetContentProcessed column

##### Use the Below command to run the stable diffusion script.The arguments are CSV PATH, IMAGE PATH, NUMBER OF IMAGES PER PROMPT, CLIP THRESHOLD, VGG THRESHOLD
VGG threshold: The Threshold value for similarity between images using VGG embeddings below which we  discard the generated images.  
Clip threshold: The Threshold value for similarity between images using Clip embeddings below which we  discard the generated images
```
python SD.py --CSV_PATH AIISC/NYT.csv --ORIGINAL_DIR AIISC/tweetImages/ --NUM_IMAGES_PER_PROMPT 3 --THRESHOLD_CLIP 0.60 --THRESHOLD_VGG 0.65
```
