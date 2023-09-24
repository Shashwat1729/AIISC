import subprocess

file_paths = [
    r'D:\Downloads\AIISC\AGID\Dream Shaper\bot.py',
    r'D:\Downloads\AIISC\AGID\F22_Dif\bot.py',
    r'D:\Downloads\AIISC\AGID\Night Diffusion\bot.py',
    r'D:\Downloads\AIISC\AGID\SD\bot.py',
    r'D:\Downloads\AIISC\AGID\MidJourney\bot.py'
]

for file_path in file_paths:
    try:
        subprocess.run(['python', file_path])
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error running '{file_path}': {str(e)}")
