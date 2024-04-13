from datasets import load_dataset
from tqdm import tqdm
from colorama import Fore, Style
from time import sleep
from openai import OpenAI
import json
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def dataset_prep():
    #NOTE No-Robots
    no_robots = load_dataset('Doctor-Shotgun/no-robots-sharegpt')['train']
    no_robots = no_robots.filter(lambda r: r["category"] not in ["Chat", "Brainstorm", "Classify", "Coding", "Extract", "Closed QA"])
    no_robots = no_robots.remove_columns(['category', 'id'])
    print(f"{Fore.GREEN}{no_robots}{Style.RESET_ALL}\n")
    all_records = [record for record in no_robots]
    with open("no-robots-shargpt.json", 'w', encoding='utf-8') as file:
        json.dump(all_records, file, indent=4)
    print(f"{Fore.YELLOW}Dataset converted into JSON!{Style.RESET_ALL}\n")

    #NOTE Neural-Story
    neural_story = load_dataset('NeuralNovel/Neural-Story-v1')['train']
    print(f"{Fore.GREEN}{neural_story}{Style.RESET_ALL}\n")
    all_records = [record for record in neural_story]
    with open("neural-story.json", 'w', encoding='utf-8') as file:
        json.dump(all_records, file, indent=4)
    print(f"{Fore.YELLOW}Dataset converted into JSON!{Style.RESET_ALL}\n")

    #NOTE OpenHermes
    openhermes = load_dataset('teknium/OpenHermes-2.5')['train']
    openhermes = openhermes.filter(lambda r: r["category"] not in ['orca', 'multiple_choice', 'general', 'coding', 'wordgame', 'joke', 'theory_of_mind', 'trivia', 'plan', 'agent', 'summarization', 'counterfactual_contextual',
                                                                    'misconception', 'riddle', 'cot', 'detailed_writing', 'gtkm', 'stylized_response', 'rp'])
    openhermes = openhermes.filter(lambda r: r['source'] not in ['CamelAI', 'EvolInstruct_70k', 'cot_alpaca_gpt4', 'glaive-code-assist', 'metamath', 'platypus'])
    # Filter out null values in the "category" column
    openhermes = openhermes.filter(lambda r: r["category"] is not None)
    # Remove unecessary columns
    openhermes = openhermes.remove_columns(['views', 'skip_prompt_formatting', 'language', 'custom_instruction', 'source', 'system_prompt', 'hash', 'title', 'id', 'model_name', 'avatarUrl', 'topic', 'model', 'category', 'idx'])
    # Backing JSON
    all_records = [record for record in openhermes]

    with open("openhermes-sharegpt.json", 'w', encoding='utf-8') as file:
        json.dump(all_records, file, indent=4)
    print(f"{Fore.YELLOW}Dataset converted into JSON!{Style.RESET_ALL}\n")

def split_json():
    with open('./no-robots-shargpt.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    split_size = 200
    if not os.path.exists('./norobots-chunks'):
        os.makedirs('./norobots-chunks')
    print(f"{Fore.YELLOW}./norobots-chunks Created{Style.RESET_ALL}\n")
    for i in range(0, len(data), split_size):
        filename = f"./norobots-chunks/no-robots-sharegpt-{i//split_size}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(data[i:i+split_size], file, indent=2)
    print(f"{Fore.YELLOW}norobots dataset has been split into parts, each containing {split_size} Elements.{Style.RESET_ALL}\n")

    with open('openhermes-sharegpt.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    split_size = 1000
    if not os.path.exists('./openhermes-chunks'):
        os.makedirs('./openhermes-chunks')
    print(f"{Fore.YELLOW}./openhermes-chunks Created{Style.RESET_ALL}\n")
    for i in range(0, len(data), split_size):
        filename = f"./openhermes-chunks/no-robots-sharegpt-{i//split_size}.json"
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(data[i:i+split_size], file, indent=2)
    print(f"{Fore.YELLOW}openhermes dataset has been split into parts, each containing {split_size} Elements.{Style.RESET_ALL}\n")

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

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    with open(output_file, 'w') as outfile:
        for item in data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def translate_text_to_italian(text):
    sleep(1)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # Use an appropriate model
        messages=[
            {"role": "user", "content": f"Traduci in italiano: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content
    return translated_text

def translate_robots_to_italian(input_json):
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

    return translated_json

def translate_neural_to_italian(input_json):
    translated_json = []

    for item in tqdm(input_json, desc='Translating instructions'):
        translated_text = translate_text_to_italian(item['text'])
        translated_json.append({'text': translated_text})

    return translated_json

def create_prompt_neural_story(text):
    sleep(1)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # Use an appropriate model
        messages=[
            {"role": "user", "content": f"genera un prompt il più breve possibile per scrivere la seguente storia: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content
    return translated_text

def neural_story_sharegpt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []
    for item in tqdm(data, desc='Generating prompt'):
        text = item['text']
        translated_text = create_prompt_neural_story(text)
        
        conversation = [
            {"from": "human", "value": translated_text},
            {"from": "gpt", "value": text}
        ]
        
        output_data.append({"conversations": conversation})
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def openhermes_sharegpt(input_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    translated_data = []
    for item in tqdm(data, desc='Translating process'):
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
        translated_data.append(translated_item)

    return translated_data

if __name__ == '__main__':
    dataset_prep()
    split_json()

    #NOTE No Robots Dataset
    out_dir = 'norobots-ita'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in tqdm(os.listdir('norobots-chunks'), desc='Processing files'):
        f_path = os.path.join('norobots-chunks', f)

        # Load the instructions from the input JSON file
        with open(f_path, 'r', encoding='utf-8') as json_file:
            input_json = json.load(json_file)

        # Translate the instructions to Italian
        translated_json = translate_robots_to_italian(input_json)

        # Save the translated instructions to a new JSON file
        with open(f"{out_dir}/{f}", "w", encoding='utf-8') as output_json_file:
            json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)

    #Merge No-Robots-ita into JSONL
    folder_path = "norobots-ita"
    output_file = "no-robots-sharegpt-ita.jsonl"
    merge_json_files(folder_path, output_file)
    
    #NOTE: Neural Story
    with open('neural-story.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    translated_json = translate_neural_to_italian(data)

    # Save the translated instructions to a new JSON file
    with open('neural-story-ita.json', "w", encoding='utf-8') as output_json_file:
        json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)

    #Create Prompt and convert Neural Story to Sharegpt
    input_file = 'neural-story-ita.json'
    output_file = 'neural-story-sharegpt-ita.json'
    neural_story_sharegpt(input_file, output_file)

    #Convert Neural Story ShareGpt to JSONL
    input_file = 'neural-story-sharegpt-ita.json'
    output_file = 'neural-story-sharegpt-ita.jsonl'
    json_to_jsonl(input_file, output_file)

    #NOTE Openhermes
    out_dir = 'openhermes-ita'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for f in tqdm(os.listdir('openhermes-chunks'), desc='Processing files'):
        f_path = os.path.join('openhermes-chunks', f)

        # Translate the instructions to Italian
        translated_json = openhermes_sharegpt(f_path)

        # Save the translated instructions to a new JSON file
        with open(f"{out_dir}/{f}", "w", encoding='utf-8') as output_json_file:
            json.dump(translated_json, output_json_file, ensure_ascii=False, indent=2)

    #NOTE OpenHermes merge and convert to JSONL
    folder_path = 'openhermes-ita'
    out_dir = 'openhermes-ita-cleaned'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Remove system if present
    for f in os.listdir(folder_path):
        f_path = os.path.join(folder_path, f)
        with open(f_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        translated_data = []
        for item in data:
            translated_conversations = []
            for conversation in item['conversations']:
                if conversation['from'] == 'human' or conversation['from'] == 'gpt':
                    translated_conversations.append({'from': conversation['from'], 'value': conversation['value']})
            translated_item = {'conversations': translated_conversations}
            translated_data.append(translated_item)
        
        # Save the translated instructions to a new JSON file
        with open(f'{out_dir}/{f}', "w", encoding='utf-8') as output_json_file:
            json.dump(translated_data, output_json_file, ensure_ascii=False, indent=2)
    
    folder_path = 'openhermes-ita-cleaned'
    output_file = 'openhermes-sharegpt-ita.jsonl'
    merge_json_files(folder_path, output_file)
