import discord
from discord.ext import commands
import asyncio
import os

# Your Discord bot token
TOKEN = 'MTE1NTMyMzEwMjE5NzE5ODk3OA.Gg1kx6.6mFzMtCdoduBjpB7mPXrgwoI9Wp4DHdewr070E'

# Path to the text file containing prompts
PROMPT_FILE_PATH = r'D:\Downloads\AIISC\AGID\prompts.txt'

# Folder to save output images
OUTPUT_FOLDER = r'D:\Downloads\AIISC\AGID\MidJourney\Output'

# Initialize the Discord bot with a command prefix
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
bot = commands.Bot(command_prefix='/imagine ', intents=intents)

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

# Event for when the bot is ready
@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

    # Access the text channel using the provided channel ID
    channel = bot.get_channel(1155315618116022322)

    if channel is not None:
        print(f'Found channel: {channel.name} (ID: {channel.id})')

        # Read prompts from the text file
        prompts = read_prompts_from_file(PROMPT_FILE_PATH)

        # Send prompts as messages to the channel with the command prefix
        for idx, prompt in enumerate(prompts, start=1):
            await channel.send(f"/imagine {prompt}")
            await asyncio.sleep(3)  # Sleep for 3 seconds between prompts

        print(f'Sent {len(prompts)} prompts.')
    else:
        print(f'Channel with ID {1155315618116022322} not found.')

# Event for when a message is received
@bot.event
async def on_message(message):
    # Check if the message contains an attachment (an image)
    if message.attachments:
        attachment_url = message.attachments[0].url
        file_extension = attachment_url.split('.')[-1]
        
        # Save the image with a numbered filename in the output folder
        output_file_path = os.path.join(OUTPUT_FOLDER, f'{message.id}.{file_extension}')
        await message.attachments[0].save(output_file_path)
        print(f'Saved image: {output_file_path}')

# Start the bot
bot.run(TOKEN)
