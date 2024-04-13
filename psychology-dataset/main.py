from datasets import load_dataset
from colorama import Fore, Style
from openai import OpenAI
from tqdm import tqdm
from time import sleep
import os
import json

os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def prep_dataset():
    psycho = load_dataset('jkhedri/psychology-dataset', split='train')
    psycho = psycho.remove_columns(['response_k'])

    all_records = []

    for record in psycho:
        all_records.append(record)
    
    with open('psychology-dataset.json', 'w', encoding='utf-8') as json_file:
        json.dump(all_records, json_file, indent=4)

def split_json():
    with open('./psychology-dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    split_size = 200
    
    if not os.path.exists('./Chunks'):
        os.makedirs('./Chunks')
    print(f"{Fore.YELLOW}./Psychology dataset chunks created{Style.RESET_ALL}\n")
    
    for i in range(0, len(data), split_size):
        filename = f"./Chunks/Psychology-dataset-{i//split_size}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(data[i:i+split_size], file, indent=2)
    
    print(f"{Fore.YELLOW}Psychology dataset has been split into parts, each containing {split_size} Elements.{Style.RESET_ALL}\n")

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

def translate_instructions_to_italian(input_json):
    translated_json = []

    for item in tqdm(input_json, desc='Translating instructions'):
        translated_conversations = []
        for conversation in item['conversations']:
            translated_text = translate_text_to_italian(conversation['value'])
            translated_conversations.append({'from': conversation['from'], 'value': translated_text})
        translated_item = {'id': item['id'], 'conversations': translated_conversations}
        translated_json.append(translated_item)

    return translated_json

def merge_json_files(folder_path, output_file):
    # Open the output file in append mode
    with open(output_file, 'a') as outfile:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Write each JSON object to a new line in the output file
                    for item in data:
                        json.dump(item, outfile, ensure_ascii=False)
                        outfile.write('\n')

if __name__ == '__main__':
    # Function calling
    prep_dataset()

    split_json()

    destination_folder = 'Chunks-Gpt'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for f in os.listdir('Chunks'):
        f_path = os.path.join('Chunks', f)
        processed_data = []

        with open(f_path, 'r', encoding='utf-8') as json_file:
            input_json = json.load(json_file)

            for item in input_json:
                new_item = {
                    "id": str(len(processed_data)),  # Assigning a new ID to each item
                    "conversations": [
                        {
                            "from": "human",
                            "value": item["question"]
                        },
                        {
                            "from": "gpt",
                            "value": item["response_j"]
                        }
                    ]
                }
                processed_data.append(new_item)

        new_file_path = os.path.join(destination_folder, f"Gpt_{f}")
        # Save the processed data to the new file
        with open(new_file_path, 'w', encoding='utf8') as json_file:
            json.dump(processed_data, json_file, indent=2)
    
    # Translate JSON instructions
    out_dir = 'Chunks-Gpt-ita'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in tqdm(os.listdir('Chunks-Gpt'), desc='Processing files'):
        f_path = os.path.join('Chunks-Gpt', f)

        # Load the instructions from the input JSON file
        with open(f_path, 'r', encoding='utf-8') as json_file:
            input_json = json.load(json_file)

        # Translate the instructions to Italian
        translated_json = translate_instructions_to_italian(input_json)

        # Save the translated instructions to a new JSON file
        with open(f"{out_dir}/{f}", "w", encoding='utf-8') as output_json_file:
            json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)

    # Merge JSON into JSONL
    folder_path = "Chunks-Gpt-ita"  # Update this with the path to your folder containing JSON files
    output_file = "psycology-dataset-gpt-ita.jsonl"  # Output file name
    merge_json_files(folder_path, output_file)