import os
import glob

def process_files(directory):
    # Get all text files in the directory
    files = glob.glob(os.path.join(directory, '*.tess'))

    # Create a new file to store the concatenated lines
    with open('concatenated.txt', 'w') as outfile:
        for file_name in files:
            with open(file_name, 'r') as infile:
                for line in infile:
                    # Remove the bracketed part at the beginning of each line
                    sentence = line.split('>', 1)[-1].strip()
                    # Write the sentence to the new file if it's not empty
                    if len(sentence) > 10:
                        outfile.write(sentence + ' ')
            outfile.write('\n')

# Call the function with the path to your directory
process_files('/home/tt/Documents/lat_text_tesserae/texts')
