import re
import json
import spacy
import argparse

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("annotation_folder", type=str, help="")

nlp = None

CHOICES_DICT = {
    'Character-Subject Misalignment': 0,
    'Action-Verb Misalignment': 1,
    'Long Abstract Subject': 2,
    'Long Introductory Phrases and Clauses': 3,
    'Subject-Verb Interruption': 4,
}


def choices_to_clearmetric(choices=[]):
    score = [0, 0, 0, 0, 0]
    for choice in choices:
        score[CHOICES_DICT[choice]] = 1
    return score


def read_annotation(path_to_json_file):

    annotations = json.load(open(path_to_json_file, encoding='utf-8'))

    for annotation in annotations:
        sentence = annotation['data']['text']
        items = annotation['annotations'][0]['result']
        
        labels = {}
        relations = []
        score = choices_to_clearmetric()
        
        for a in items:
            if a['type']=='labels':
                labels[a['id']] = {
                    'text': a['value']['text'],
                    'start': a['value']['start'],
                    'end': a['value']['end'],
                    'labels': a['value']['labels'],
                }
            
            elif a['type']=='choices':
                score = choices_to_clearmetric(a['value']['choices'])
            
            elif a['type']=='relation':
                src, tgt = a['from_id'], a['to_id']

                if 'Character' in labels[src]['labels'] and 'Action' in labels[tgt]['labels']:
                    relations.append(
                        {'character': dict(labels[src]), 'action': dict(labels[tgt])}
                    )
                    # relations[-1]['character'].pop('labels')
                    # relations[-1]['action'].pop('labels')
                
                elif 'Subject' in labels[src]['labels'] and 'Verb' in labels[tgt]['labels']:
                    relations.append(
                        {'subject': labels[src], 'verb': labels[tgt]}
                    )
                    # relations[-1]['subject'].pop('labels')
                    # relations[-1]['verb'].pop('labels')
                
        yield {'sent': sentence, 'score': score, 'labels': relations}



def char_to_token(annotation):
    
    if 'labels' not in annotation:
        return annotation
    
    if 'tokens' not in annotation:
        global nlp, spacy_model_name
        if not nlp:
            nlp = spacy.load(spacy_model_name)
        doc = nlp(annotation['sent'])
        annotation['tokens'] = [t.text for t in doc]
    
    for i, label in enumerate(annotation['labels']):

        for role in label:
            found = False
            text = label[role]['text']
            start, end = label[role]['start'], label[role]['end']

            matchs = re.finditer(text, doc.text)
                
            for match in matchs:
                if match is None:
                    raise Exception(f'No matching span found: "{text}" in ({doc.text})')
                
                span = doc.char_span(*match.span())

                if span is None:
                    continue
                
                span = list(span)
                match_start = span[0].idx
                match_end = span[-1].idx + len(span[-1].text)
                
                if (start==match_start) and (end==match_end):
                    annotation['labels'][i][role]['i_start'] = span[0].i
                    annotation['labels'][i][role]['i_end'] = span[-1].i + 1
                    found = True
                    break
            
            if not found:
                raise Exception(f'No matching span found: "{text}" in ({doc.text})')
    return annotation


if __name__=='__main__':
    args = parser.parse_args()
    spacy_model_name = 'en_core_web_md'

    annotation_path = 'data/' + args.annotation_folder +'/annotation-original.json'
    annotation = read_annotation(annotation_path)

    i = 0
    for item in annotation:
        label = char_to_token(item)
        print(len(label['labels']))
        print(item)
        i += 1
        if i>0:
            break
