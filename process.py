with open('sentences.txt', 'r') as infile:
    original, paraphrased = [], []
    for line in infile:
        sents = line.split('\t')
        sents[1] = sents[1].replace('\n', '')
        original.append(sents[0])
        paraphrased.append(sents[1])

with open('sentences.src', 'w') as infile:
    for sent in original:
        infile.writelines(sent+'\n')

with open('sentences.tgt', 'w') as infile:
    for sent in paraphrased:
        infile.writelines(sent+'\n')
