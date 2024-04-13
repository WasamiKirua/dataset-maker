from time import sleep
from openai import OpenAI
from tqdm import tqdm
import json
import os

os.environ["OPENAI_API_KEY"] = ''

# Assuming OPENAI_API_KEY is already set in your environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def process_instruction_with_openai(text):
    # Simulating API call with a placeholder function
    sleep(1)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Crea un haiku in italiano che segua queste linee guida specifiche: Struttura e Metrica: Componi un poema breve di tre versi. Cerca di avvicinarti a una struttura metrica di 5-7-5 sillabe per verso, ma sentiti libero di adattare leggermente il conteggio delle sillabe per mantenere l'armonia e la naturalità del linguaggio italiano. Elemento Stagionale (Kigo): Includi nel tuo haiku un riferimento chiaro a una delle quattro stagioni (primavera, estate, autunno, inverno). Questo può essere fatto attraverso l'uso di immagini naturali, parole o concetti che evocano specificamente quel periodo dell'anno. Taglio (Kireji): Usa una forma di pausa, come la punteggiatura (virgola, punto e virgola, punto) o un cambio di immagine o tono tra i versi, per creare un momento di riflessione o sorpresa. Questa pausa dovrebbe servire a dividere il poema in due parti che, pur essendo distinte, rimangono connesse in significato o emozione. Semplicità ed Essenzialità: Concentrati su immagini e concetti semplici, preferibilmente legati alla natura o a momenti quotidiani, che rivelino qualcosa di più profondo sulla condizione umana, sulla natura o sulla spiritualità. Ogni parola deve essere scelta con cura per la sua capacità di evocare immagini vivide e significati ricchi. Evita Rime e Metafore Complesse: Mantieni il linguaggio diretto e privo di rime forzate o di complesse figure retoriche. L'haiku dovrebbe preferire la chiarezza e l'immediatezza, con un focus sulla potenza evocativa delle immagini naturali e quotidiane. Momento Istantaneo: Cerca di catturare l'essenza di un attimo fugace, offrendo una visione o un'osservazione che, pur nella sua brevità, apre a riflessioni più ampie o universali. Originalità e Personalità: Lascia che la tua voce unica traspaia nell'haiku, esplorando temi, immagini e emozioni che ti sono personali o che ti colpiscono particolarmente. Ricorda che, nonostante le regole, l'haiku è un'espressione artistica soggettiva e personale."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def process_instructions(input_json):
    new_json_structure = []
    for idx, item in enumerate(tqdm(input_json, desc='Processing instructions')):
        instruction = item['instructions']
        api_response = process_instruction_with_openai(instruction)
        
        new_item = {
            "id": str(idx + 1),
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": api_response}
            ]
        }
        new_json_structure.append(new_item)
    return new_json_structure

def main(input_file_path, output_file_path):  
    # Load the instructions from the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as json_file:
        input_json = json.load(json_file)
    
    # Process each instruction and generate the new JSON structure
    new_json = process_instructions(input_json)
    
    # Save the new JSON structure to a file
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    f_name = input_file_path.split('/')[1]
    print(f'{output_file_path}/{f_name}')
    with open(f'{output_file_path}/{f_name}', "w", encoding='utf-8') as output_json_file:
        json.dump(new_json, output_json_file, ensure_ascii=False, indent=4)

def merge_json_files(folder_path, output_file):
    # Open the output file in append mode
    with open(output_file, 'a') as outfile:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r',) as file:
                    data = json.load(file)
                    # Write each JSON object to a new line in the output file
                    for item in data:
                        json.dump(item, outfile, ensure_ascii=False)
                        outfile.write('\n')

if __name__ == '__main__':
    out_dir = './haiku-dataset-ita'
    for f in os.listdir('./chunks-ita'):
        f_path = os.path.join('chunks-ita', f)
        main(f_path, out_dir)

    # Merge JSON files into a JSONL
    folder_path = "./haiku-dataset-ita"
    output_file = "haiku-dataset-ita.jsonl"
    merge_json_files(folder_path, output_file)
    
