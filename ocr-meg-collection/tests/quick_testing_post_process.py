import pandas as pd
import re
import json


response_json = {'id': 'chatcmpl-A4OCxSSAQZldr7MELhUzN4GNg9q1d',
                 'object': 'chat.completion',
                 'created': 1725609731,
                 'model': 'gpt-4o-mini-2024-07-18',
                 'choices': [
                     {'index': 0,
                      'message':
                          {'role': 'assistant',
                           'content': '```json\n{\n  "artist": "François Jouffa",\n  "album_title": "Démons et merveilles à Bali",\n  "songs": [\n    "Être du nom de Brabo",\n    "Kézako",\n    "La chasse au dodo",\n    "Sur la table d\'hôte",\n    "Les parades de la journée",\n    "La galerie de la promenade",\n    "Géométrie de la procession",\n    "Géométrie poétique d\'insertion",\n    "Chant des espoirs",\n    "Principe d\'andromède",\n    "Jipes de galilée",\n    "Circles de la famille",\n    "Comble de cope à Neuvrangan"\n  ],\n  "editor": "Éditions A",\n  "other_info": {\n    "format": "33 T",\n    "distribution": "Discodis"\n  }\n}\n```',
                           'refusal': None
                           },
                          'logprobs': None,
                          'finish_reason': 'stop'
                      }
                 ],
                 'usage': {'prompt_tokens': 25548, 'completion_tokens': 194, 'total_tokens': 25742},
                 'system_fingerprint': 'fp_54e2f484be'}

# access only the content key of the message:

content_str = response_json['choices'][0]['message']['content']
content = re.findall(r"```json(.*?)```", content_str, re.DOTALL)

if content:
    extracted_text = content[0].strip()
    print(extracted_text)
else:
    print("No content found between triple backticks.")

# Convert json format to pandas dataframe
try:
    content_dict = json.loads(extracted_text)
    df = pd.DataFrame(content_dict)
    print(df)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
