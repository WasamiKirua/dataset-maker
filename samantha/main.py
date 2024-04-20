from tqdm import tqdm
from colorama import Fore, Style
from datasets import Dataset
from time import sleep
from openai import OpenAI
import json
import os

os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def split_json():
    with open('./samantha-1.1.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    split_size = 50
    
    if not os.path.exists('./chunks'):
        os.makedirs('./chunks')
    print(f"{Fore.YELLOW}Samantha chunks created{Style.RESET_ALL}\n")
    
    for i in range(0, len(data), split_size):
        filename = f"./chunks/samantha-sharegpt-{i//split_size}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(data[i:i+split_size], file, indent=2)
    
    print(f"{Fore.YELLOW}Samantha dataset splitted in {split_size} Elements.{Style.RESET_ALL}\n")

def translate_text_to_italian(text):
    sleep(1)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use an appropriate model
        messages=[
            {"role": "user", "content": f"Traduci il seguente testo in italiano: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content
    return translated_text

def translate_samantha_to_italian(input_json):
    translated_json = []

    for item in tqdm(input_json, desc='Translating instructions'):
        translated_conversations = []
        for conversation in item['conversations']:
            if conversation['from'] == 'human' or conversation['from'] == 'gpt':
                translated_text = translate_text_to_italian(conversation['value'])
                # Replace literal "\n" characters with actual line breaks
                translated_text = translated_text.replace("\\n", "\n")
                translated_conversations.append({'from': conversation['from'], 'value': translated_text})
            else:
                translated_conversations.append({'from': conversation['from'], 'value': conversation['value']})
        translated_item = {'conversations': translated_conversations}
        translated_json.append(translated_item)
    
if __name__ == "__main__":
    #split_json()

    #NOTE No Robots Dataset
    out_dir = 'samantha-ita'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # for f in tqdm(os.listdir('norobots-chunks'), desc='Processing files'):
    #     f_path = os.path.join('norobots-chunks', f)

    #     # Load the instructions from the input JSON file
    #     with open(f_path, 'r', encoding='utf-8') as json_file:
    #         input_json = json.load(json_file)

    #     # Translate the instructions to Italian
    #     translated_json = translate_samantha_to_italian(input_json)

    #     # Save the translated instructions to a new JSON file
    #     with open(f"{out_dir}/{f}", "w", encoding='utf-8') as output_json_file:
    #         json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)

    with open('test.json', 'r', encoding='utf-8') as json_file:
        input_json = json.load(json_file)
    
    translated_json = translate_samantha_to_italian(input_json)

    # Save the translated instructions to a new JSON file
    with open(f"{out_dir}/test-ita.json", "w", encoding='utf-8') as output_json_file:
        json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)
