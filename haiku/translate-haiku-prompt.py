from datasets import load_dataset
from openai import OpenAI
from colorama import Fore, Style
from tqdm import tqdm
from time import sleep
import json
import os

os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def translate_text_to_italian(text):
    sleep(3)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use an appropriate model
        messages=[
            {"role": "user", "content": f"Traduci il seguente testo in italiano: {text}"}
        ]
    )
    print(response)
    # Extract the translated text from the response
    translated_text = response.choices[0].message.content
    return translated_text

def translate_instructions_to_italian(input_json):
    translated_instructions = []

    for item in tqdm(input_json, desc='Translating instructions'):
        english_text = item['instructions']
        italian_text = translate_text_to_italian(english_text)
        translated_instructions.append({"instructions": italian_text})

    return translated_instructions

def split_json():
    with open('./haiku-prompts.json', 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    split_size = 200
    total_ids = len(data)
    destination_folder = './chunks'

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    print(f"{Fore.YELLOW}{destination_folder} Created{Style.RESET_ALL}\n")

    # Calculate the number of splits needed
    num_splits = (total_ids + split_size - 1) // split_size  # Ceiling division

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, total_ids)
        split_part = data[start_idx:end_idx]

        filename = f"{destination_folder}/haiku-prompt-{i}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(split_part, file, indent=4)

    print(f"Capybara Data has been split into {num_splits} parts, each containing {split_size} Elements.")

def main():
    ## Export Dataset into JSON
    haiku = load_dataset('davanstrien/haiku_prompts', split='train')
    haiku_records = []

    for instruct in haiku:
        haiku_records.append(instruct)

    with open('haiku-prompts.json', 'w', encoding='utf-8') as json_file:
        json.dump(haiku_records, json_file, indent=2)

    ## Split the JSON into chunks
    split_json()
    
    ## Translate JSON instructions
    out_dir = 'chunks-ita'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in tqdm(os.listdir('./chunks'), desc='Processing files'):
        f_path = os.path.join('./chunks', f)

        # Load the instructions from the input JSON file
        with open(f_path, 'r', encoding='utf-8') as json_file:
            input_json = json.load(json_file)

        # Translate the instructions to Italian
        translated_json = translate_instructions_to_italian(input_json)

        # Save the translated instructions to a new JSON file
        with open(f"{out_dir}/{f}", "w", encoding='utf-8') as output_json_file:
            json.dump(translated_json, output_json_file, ensure_ascii=False)

if __name__ == '__main__':
    main()
    
    


