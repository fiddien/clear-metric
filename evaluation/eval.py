import json
import re
import spacy
from spacy.lang.en import English
# import evaluate

nlp = English()
tokenizer = nlp.tokenizer
spacy_model_name = 'en_core_web_md'

choices_dict = {
    'Character-Subject Misalignment': 0,
    'Action-Verb Misalignment': 1,
    'Long Abstract Subject': 2,
    'Long Introductory Phrases and Clauses': 3,
    'Subject-Verb Interruption': 4,
}

def choices_to_clearmetric(choices=[]):
    score = [0, 0, 0, 0, 0]
    for choice in choices:
        score[choices_dict[choice]] = 1
    return score


def calculate_metrics(true_labels, predicted_labels):
    # Ensure that the length of both vectors is the same
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Input vectors must have the same length.")

    # Initialize variables to count True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    tp, tn, fp, fn = 0, 0, 0, 0

    # Calculate the metrics
    for true, predicted in zip(true_labels, predicted_labels):
        if true == 1 and predicted == 1:
            tp += 1
        elif true == 0 and predicted == 0:
            tn += 1
        elif true == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': f1_score, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


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
                        {'character': labels[src].copy(), 'action': labels[tgt].copy()}
                    )
                    relations[-1]['character'].pop('labels')
                    relations[-1]['action'].pop('labels')
                if 'Subject' in labels[src]['labels'] and 'Verb' in labels[tgt]['labels']:
                    relations.append(
                        {'subject': labels[src].copy(), 'verb': labels[tgt].copy()}
                    )
                    relations[-1]['subject'].pop('labels')
                    relations[-1]['verb'].pop('labels')
                
        yield {'sent': sentence, 'score': score, 'labels': relations}


def read_prediction(path_to_json_file):

    items = json.load(open(path_to_json_file, encoding='utf-8'))

    for item in items:
        yield item


def char_to_token(annotation, tokens=None):
    
    if 'labels' not in annotation:
        return annotation
    
    if 'tokens' not in annotation:
        doc = tokenizer(annotation['sent'])
        annotation['tokens'] = [t.text for t in doc]
        if tokens:
            assert tokens == annotation['tokens']
    
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
                    annotation['labels'][i][role]['i_end'] = span[-1].i
                    found = True
                    break
            if not found:
                raise Exception(f'No matching span found: "{text}" in ({doc.text})')
            
    return annotation



def to_bio_labels(annotation):
    
    if 'labels' not in annotation:
        return annotation
    
    if 'tokens' not in annotation:
        annotation['tokens'] = [t.text for t in tokenizer(annotation['sent'])]

    bio_labels = [] # one list for each pair of subject-verb and character-action
    for i, label in enumerate(annotation['labels']):
        bio = ['O' for t in annotation['tokens']]
        for role in label:
            start, end = label[role]['i_start'], label[role]['i_end']
            bio[start] = f'B-{role.upper()}'
            for i in range(start+1, end+1):
                bio[i] = f'I-{role.upper()}'
        
        bio_labels.append(bio)
    annotation['bio_labels'] = bio_labels

    return annotation



if __name__=='__main__':
    annotation_path = 'data/Style-Examples/annotation-src.json'
    result_path = 'evaluation/Style-Examples/prediction-src.json'
    
    annotations = read_annotation(annotation_path)
    predictions = read_prediction(result_path)
    i = 0
    scores = []
    for anno, pred in zip(annotations, predictions):
        anno = to_bio_labels(char_to_token(anno, tokens=pred['tokens']))
        pred = to_bio_labels(pred)
        scores.append(calculate_metrics(anno['score'], pred['score']))

    n = len(scores)
    print('Accuracy  :', sum([x['accuracy'] for x in scores])/n)
    print('Precision :', sum([x['precision'] for x in scores])/n)
    print('Recall    :', sum([x['recall'] for x in scores])/n)
    print('F1-score  :', sum([x['f1-score'] for x in scores])/n)