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

def transform_conversation(jsonl_line):
    try:
        conversation_data = json.loads(jsonl_line)
        new_messages = []
        for message in conversation_data['conversation'].split('\n\n'):
            # Split message into speaker and text, handling cases where there's no colon
            colon_index = message.find(': ')
            if colon_index != -1:
                speaker, text = message[:colon_index], message[colon_index + 2:]
            else:
                # If no colon is found, assume the speaker is "Theodore" (human)
                speaker, text = 'Theodore', message
            if speaker == 'Theodore':
                speaker = 'human'
            elif speaker == 'Samantha':
                speaker = 'gpt'
            new_messages.append({'from': speaker, 'value': text})
        return {'id': str(conversation_data['elapsed']), 'conversations': new_messages}
    except Exception as e:
        print("Error:", e)
        print("Line causing the error:", jsonl_line)
        return None
    
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

# Translate and replace "human" and "gpt" messages
def translate_instructions_to_italian(input_jsonl_path):
    out_jsonl_path = input_jsonl_path.replace('.jsonl', '_translated.jsonl')

    with open(input_jsonl_path, 'r', encoding='utf-8') as input_file, open(out_jsonl_path, 'w', encoding='utf-8') as output_file:

        for line in tqdm(input_file, desc='Translating instructions'):
            json_data = json.loads(line.strip())
            translated_conversations = []

            for conversation in tqdm(json_data['conversations'], desc='Translating conversations'):
                translated_text = translate_text_to_italian(conversation['value'])
                translated_conversations.append({'from': conversation['from'], 'value': translated_text})

            json_data['conversations'] = translated_conversations
            output_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

    return out_jsonl_path

def merge_jsonl_from_directory(input_directory, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_directory):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(input_directory, filename)
                with open(filepath, 'r') as infile:
                    for line in infile:
                        outfile.write(line)

def main():
    #NOTE Transform JSONL into GPT format

    for f in os.listdir('samantha-data'):
        f_path = os.path.join(f'samantha-data/{f}')

        transformed_data = []
        # Read input JSONL file and transform each line
        with open(f_path, 'r') as file:
            for line in file:
                transformed = transform_conversation(line.strip())
                if transformed:
                    transformed_data.append(json.dumps(transformed))

        # Write transformed data to output JSONL file
        with open(f'samantha-data-transformed/{f}', 'w') as file:
            file.write('\n'.join(transformed_data))

    #NOTE Translate transformed JSONL into ITA
    
    for f in os.listdir('samantha-data-transformed'):
        f_path = os.path.join('samantha-data-transformed', f)

        input_jsonl_path = f_path
        translated_jsonl_path = translate_instructions_to_italian(input_jsonl_path)
        print(f"Translated JSONL file saved to: {translated_jsonl_path}")

    #NOTE Merge JSONL files
    input_directory = "samantha-data-transformed"  # Update this with the path to your input directory
    output_file = "samantha-data-gpt-ita.jsonl"  # Output file name
    merge_jsonl_from_directory(input_directory, output_file)
    

if __name__ == "__main__":
    main()
