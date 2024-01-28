from cltk.sentence.lat import LatinPunktSentenceTokenizer
import os

def split_text(input_file, output_file):
    # Initialize Latin sentence tokenizer
    tokenizer = LatinPunktSentenceTokenizer()

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into sentences
    sentences = tokenizer.tokenize(text)

    # Write the sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + os.linesep)

# Call the function with your input and output file paths
split_text('concatenated.txt', 'demo.txt')
