def split_corpus(input_file, latin_output_file, english_output_file):
    with open(input_file, 'r') as f, \
         open(latin_output_file, 'w') as latin_f, \
         open(english_output_file, 'w') as english_f:
        for line in f:
            split = line.split('\t')
            if len(split) < 2:
                continue
            latin_f.write(split[0] + '\n')
            english_f.write(split[1].strip() + '\n')

split_corpus('output5.txt', 'lt.txt', 'en.txt')
