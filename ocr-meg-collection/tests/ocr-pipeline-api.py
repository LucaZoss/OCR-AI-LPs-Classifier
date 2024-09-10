from openai import OpenAI
import sys
import os


import base64
import requests

# setting up Dynamic path
current_path = os.path.dirname(os.path.realpath(__file__))
print(current_path)

# api_key = os.environ.get('OPENAI_KEY')
# OpenAI API Key
api_key = "sk-7mFl0kHX0-EheoipR2JzGLEnPTFsiACZ_t1O_fd3TpT3BlbkFJk0uEfZNLfvwTPYq6NkLet7Q_vvz43SB9FRsZer7vYA"

PROMPT = '''
Given the following image of a vinyl record, your task is to extract specific information from the cover and label of the record and provide the output in a structured JSON format.

### Information to Extract:

1. **Title**: The main title of the album, as it appears on the front side of the cover.
2. **Subtitle**: Any subtitle of the album, also found on the front side of the cover.
3. **Interpreter/Artist**: The main artists, authors, or interpreters of the album.
4. **Edition**: The name of the editor, usually a corporate entity.
5. **Edition Number**: The specific edition number of the album.
6. **Edition Year**: The year of the album's edition.
7. **Production**: The name of the producer, which could be an organization and/or an individual.
8. **Production Year**: The year the album was produced.
9. **Track Names**: Names of all the tracks listed on both Face A and Face B.
10. **Track Length**: Length of each track.

### Other Information:
- Any additional relevant information found on the album cover or label.

### Output Format:
- Provide the extracted information in the following JSON format:
  ```json
  {
    "Title": "",
    "Subtitle": "",
    "Interpreter_Artist": "",
    "Edition": "",
    "Edition_Number": "",
    "Edition_Year": "",
    "Production": "",
    "Production_Year": "",
    "Tracks": {
      "Face_A": [
        {
          "Track_Name": "",
          "Track_Length": ""
        },
        ...
      ],
      "Face_B": [
        {
          "Track_Name": "",
          "Track_Length": ""
        },
        ...
      ]
    },
    "Other_Information": ""
  }

### Guidelines:
- **Do Not Make Up Any Information**: Extract only the information that is clearly present in the image.
- **Do Not Translate Information**: Keep all text exactly as it appears in the image.
- If specific information is not available or visible, mark the field as "NaN".

### Important Note:
Ensure accuracy and completeness based on the visible text in the image. Provide the extracted data in the requested CSV format without any alterations or assumptions.
'''

# Function to encode the image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ds/test/20240906_131209.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())

print(response.json()['choices'][0]['message']['content'])
