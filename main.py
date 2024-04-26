import os
import shutil

# Convert epubs to plain text
def convert_epubs():
    input_folder = 'epub'
    output_folder_raw = 'training-data/0_raw'

    if not os.path.exists(output_folder_raw):
        os.makedirs(output_folder_raw)

    list_epub = os.listdir(input_folder)
    for f in list_epub:
        f_path = os.path.join(input_folder, f)
        f_name = f'{f.split(".")[0]}.txt'
        os.system(f'python epub2txt-all.py {f_path} {output_folder_raw}/{f_name} -n -f -p')

def add_gutenberg_markers(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+', encoding='utf-8') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("*** START OF THE PROJECT GUTENBERG EBOOK ***\n" + content)
                file.write("\n*** END OF THE PROJECT GUTENBERG EBOOK ***")

def replace_shit(input_file, output_dir):
    # Open input file
    with open(input_file, 'r', encoding='utf-8') as file:
        # Read lines
        lines = file.readlines()

        # Construct output filename
        output_filename = os.path.basename(input_file).replace('.txt', '_cleaned.txt')
        output_file = os.path.join(output_dir, output_filename)

        # Open output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                # Remove unwanted lines and log them
                if line.startswith('#') or line.startswith('##') or (line.startswith('(') and line.endswith(')\n')):
                    continue

                # Remove unwanted text and log them
                line = line.replace('— ', '').replace('« ', '').replace(' »', '').replace('– ', '').replace('_', '').replace('«', '').replace('»', '').replace('>  ', '').replace('>', '').replace(' —', '').replace('—', '').replace('- ', '')

                outfile.write(line) 
        print(f'{input_file} 💩 Cleaned!')

def preprocess():
    # Run Preprocessing script
    os.system('python step2-preprocess.py --output_dir "./training-data/1_preprocessed/" --input_dir "./training-data/0_raw/"')

def chunking(source_dir, output_file):
    os.system(f'python step3-chunking.py --source_dir {source_dir} --output_file {output_file}')

if __name__ == '__main__':
    convert_epubs()
    add_gutenberg_markers('training-data/0_raw')
    preprocess()

    input_dir = './training-data/1_preprocessed/'
    output_dir = './training-data/2_cleaned'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_list = os.listdir(input_dir)
    for f in f_list:
        input_file = os.path.join(input_dir, f)
        replace_shit(input_file, output_dir)
    
    #NOTE: Check txt files manually for a better cleaning (shit in shit out)
    # for f in os.listdir('./training-data/2_cleaned'):
    #     f_name = f.split('.')[0]

    #     if not os.path.exists(f'./training-data/3_parquet/{f_name}'):
    #         os.makedirs(f'./training-data/3_parquet/{f_name}')
    
    # # Move files pre parquet creation
    # for f in os.listdir('./training-data/2_cleaned'):
    #     f_name = f.split('.')[0]

    #     source = f'./training-data/2_cleaned/{f}'
    #     destination = f'./training-data/3_parquet/{f_name}/{f}'
    #     shutil.copyfile(source, destination)

    # # making parquet
    # for folder in os.listdir('./training-data/3_parquet'):
    #     for f in os.listdir(f'./training-data/3_parquet/{folder}'):
    #         parquet_name = f.split('.')[0]
    #         source_dir = f'./training-data/3_parquet/{folder}/'
    #         output_file = f'./training-data/3_parquet/{folder}/{parquet_name}.parquet'
    #         chunking(source_dir, output_file)
