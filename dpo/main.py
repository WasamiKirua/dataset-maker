from datasets import load_dataset
from colorama import Fore, Style
from tqdm import tqdm
from time import sleep
from openai import OpenAI
import json
import os

os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def prep_dataset():
    truthy = load_dataset('jondurbin/truthy-dpo-v0.1', split='train')
    truthy = truthy.remove_columns(['id', 'source'])

    all_records = []

    for record in truthy:
        all_records.append(record)

    with open('truthy-dpov0.1.json', 'w', encoding='utf-8') as thruthy_file:
        json.dump(all_records, thruthy_file, indent=4)

    orca = load_dataset('Intel/orca_dpo_pairs', split='train')
    
    all_records = []

    for record in orca:
        all_records.append(record)
    
    with open('orca_dpo_pairs.json', 'w', encoding='utf-8') as orca_file:
        json.dump(all_records, orca_file, indent=4)

def split_json():
    with open('./truthy-dpov0.1.json', 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    split_size = 200
    total_ids = len(data)
    destination_truthy = './Truthy-Chunks'

    if not os.path.exists(destination_truthy):
        os.makedirs(destination_truthy)

    print(f"{Fore.YELLOW}{destination_truthy} Created{Style.RESET_ALL}\n")

    # Calculate the number of splits needed
    num_splits = (total_ids + split_size - 1) // split_size  # Ceiling division

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, total_ids)
        split_part = data[start_idx:end_idx]

        filename = f"{destination_truthy}/truthy-dpov0.1-{i}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(split_part, file, indent=4)

    print(f"Truthy-dpo-v0.1 Data has been split into {num_splits} parts, each containing {split_size} Elements.")

    with open('./orca_dpo_pairs.json', 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    split_size = 200
    total_ids = len(data)
    destination_orca = './Orca-Chunks'

    if not os.path.exists(destination_orca):
        os.makedirs(destination_orca)

    print(f"{Fore.YELLOW}{destination_orca} Created{Style.RESET_ALL}\n")

    # Calculate the number of splits needed
    num_splits = (total_ids + split_size - 1) // split_size  # Ceiling division

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, total_ids)
        split_part = data[start_idx:end_idx]

        filename = f"{destination_orca}/orca_dpo_pairs-{i}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(split_part, file, indent=4)

    print(f"Orca_dpo_pairs Data has been split into {num_splits} parts, each containing {split_size} Elements.")

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

def translate_instructions_to_italian(input_json_path):
    out_json_path = input_json_path.replace('.json', '_translated.json')

    with open(input_json_path, 'r', encoding='utf-8') as input_file , open(out_json_path, 'w', encoding='utf-8') as output_file:
        json_data = json.load(input_file)
        translated_data = []

        for entry in tqdm(json_data, desc='Translating entries'):
            translated_entry = {}

            if 'system' in entry and entry['system']:
                translated_entry['system'] = translate_text_to_italian(entry['system'])
            else:
                translated_entry['system'] = entry['system']
            
            if 'question' in entry:
                translated_entry['question'] = translate_text_to_italian(entry['question'])

            if 'prompt' in entry:
                translated_entry['prompt'] = translate_text_to_italian(entry['prompt'])

            if 'chosen' in entry:
                translated_entry['chosen'] = translate_text_to_italian(entry['chosen'])

            if 'rejected' in entry:
                translated_entry['rejected'] = translate_text_to_italian(entry['rejected'])

            translated_data.append(translated_entry)

        json.dump(translated_data, output_file, ensure_ascii=False, indent=4)

    return out_json_path

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
    #NOTE Function calling
    prep_dataset()
    split_json()

    #NOTE Translating
    for f in os.listdir('Truthy-Chunks'):
        f_path = os.path.join(f'Truthy-Chunks/{f}')

        input_json_path = f_path
        translated_json_path = translate_instructions_to_italian(input_json_path)
        print(f"Translated JSONL file saved to: {translated_json_path}")

    #NOTE merge jsons into JSONL
    folder_path = "Truthy-Chunks"  # Update this with the path to your folder containing JSON files
    output_file = "truthy-dpo-ita.jsonl"  # Output file name
    merge_json_files(folder_path, output_file)

    #NOTE Translating
    for f in os.listdir('Orca-Chunks'):
        f_path = os.path.join(f'Orca-Chunks/{f}')

        input_json_path = f_path
        translated_json_path = translate_instructions_to_italian(input_json_path)
        print(f"Translated JSONL file saved to: {translated_json_path}")