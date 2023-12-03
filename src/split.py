def split_corpus(input_file, latin_output_file, english_output_file):
    with open(input_file, 'r') as f, \
         open(latin_output_file, 'w') as latin_f, \
         open(english_output_file, 'w') as english_f:
        for line in f:
            latin, english = line.split('\t')
            latin_f.write(latin + '\n')
            english_f.write(english.strip() + '\n')

split_corpus('/home/tt/Documents/DeclEngine/data/lt-en.txt', 'lt.txt', 'en.txt')
