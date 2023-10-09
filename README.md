# AIISC
 # Model Image Generation

This repository contains Python scripts for generating images using various models provided by Stable Diffusion, including **Dream Shaper**, **F22**, and **Night Diffusion**. These scripts read prompts from a text file and send requests to the respective model APIs to generate images based on these prompts. The generated images are saved in the `Output` folder.

## Prerequisites

Before running the scripts, ensure you have the following:

- Python installed on your system.
- The required Python libraries (`requests` and `json`) installed. You can install them using pip:

    ```bash
    pip install requests
    ```

- API keys for the specific models you want to use. Replace `'YOUR_API_KEY'` in the scripts with your actual API keys.

## Usage

1. Clone this repository or download the script for the specific model you want to use.

2. Create a text file (`prompts.txt`) in the parent directory of the script, and add your desired prompts, one per line.

3. Modify the script's configuration if needed:

    - `API_ENDPOINT`: The API endpoint for the specific model.
    - `API_KEY`: Your API key.
    - `PROMPT_FILE_PATH`: Relative path to the prompts file.
    - `OUTPUT_FOLDER`: Relative path to the folder where generated images will be saved.
    - Adjust other payload parameters as necessary for the chosen model.

4. Run the script:

    ```bash
    python bot.py
    ```

5. The script will process each prompt from the `prompts.txt` file, send requests to the API, and save the generated images in the `Output` folder.

## Example

Here's an example of the `prompts.txt` file:



Running the script with these prompts will generate images corresponding to each prompt and save them in the `Output` folder.

## Important Notes

- Make sure you have proper permissions to access the specific model's API and that your API key is correctly set in the script.

- Depending on the number of prompts and the complexity of the model, the script may take some time to complete.

- Ensure that the required Python libraries are installed and that your Python environment is set up correctly.

- For more information on the specific model and its API, refer to the official documentation.



