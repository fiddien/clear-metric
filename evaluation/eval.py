import json
import re
import spacy
from spacy.lang.en import English
import argparse

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--mode", type=str, default=None,
                    help="Mode of evaluation: clear-metric, span-label. If none given, then it's both")
parser.add_argument("-sm", "--span-mode", type=str, default='subset',
                    help="Strictness of the span-labeling evaluation: 'subset' (default) or 'exact'.")

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


def sort_by_start(dic):
    # dic = sorted(dic, key=lambda x: x['start'])
    if 'character' in dic:
        return dic['character']['start'], dic['action']['start']
    else:
        return dic['subject']['start'], dic['verb']['start']


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
        relations = sorted(relations, key=sort_by_start)
        yield {'sent': sentence, 'score': score, 'labels': relations}


def read_prediction(path_to_json_file):

    items = json.load(open(path_to_json_file, encoding='utf-8'))

    for item in items:
        item['labels'] = sorted(item['labels'], key=sort_by_start)
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


def evaluate_clear_metrics(true_labels, predicted_labels):
    # Ensure that the length of both vectors is the same
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Input vectors must have the same length.")

    num_categories = len(true_labels[0])
    class_metrics = {}

    for category in range(num_categories):
        tp, tn, fp, fn = 0, 0, 0, 0

        for true, predicted in zip(true_labels, predicted_labels):
            if true[category] == 1:
                if predicted[category] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted[category] == 1:
                    fp += 1
                else:
                    tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[f"Class_{category}"] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1_score,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        }

    return class_metrics


def evaluate_spans(predicted_labels, ground_truth_labels, mode='subset'):
    """
    Evluate the spans. Perform alignment for spans using token number information.

    Args:
        predicted_labels (list): List of predicted spans with token number information.
        ground_truth_labels (list): List of ground truth spans with token number information.
        sentence_length (int): The length of the sentence in tokens.

    Returns:
        true_positives (int): Count of true positive spans.
        false_positives (int): Count of false positive spans.
        false_negatives (int): Count of false negative spans.
    """
    def get_token_positions(span):
        return set(range(span['i_start'], span['i_end'] + 1))

    roles = ['character', 'action', 'subject', 'verb']

    # Convert the spans to sets of token positions
    predicted_spans = []
    ground_truth_spans = []
    for item in predicted_labels:
        for role in item:
            predicted_spans.append((role, get_token_positions(item[role])))
    for item in ground_truth_labels:
        for role in item:
            ground_truth_spans.append((role, get_token_positions(item[role])))

    support = {role: sum([1 if r==role else 0 for r, _ in predicted_spans]) for role in roles}

    # Calculate true positives, false positives, and false negatives
    true_positives = {role:0 for role in roles}
    false_positives = {role:0 for role in roles}
    false_negatives = {role:0 for role in roles}
    
    if mode=='subset':
        for pred_role, pred_span in predicted_spans:
            if any((pred_span.issubset(gt_span) and pred_role==gt_role) \
                for gt_role, gt_span in ground_truth_spans):
                true_positives[pred_role] += 1
            else:
                false_positives[pred_role] += 1

        for gt_role, gt_span in ground_truth_spans:
            if not any((gt_span.issubset(pred_span) and gt_role==pred_role) \
                    for pred_role, pred_span in predicted_spans):
                false_negatives[gt_role] += 1

    elif mode=='exact':
        for pred_role, pred_span in predicted_spans:
            if any((pred_span==gt_span and pred_role==gt_role) \
                for gt_role, gt_span in ground_truth_spans):
                true_positives[pred_role] += 1
            else:
                false_positives[pred_role] += 1

        for gt_role, gt_span in ground_truth_spans:
            if not any((gt_span==pred_span and gt_role==pred_role) \
                    for pred_role, pred_span in predicted_spans):
                false_negatives[gt_role] += 1
    
    else:
        raise ValueError("Wrong 'mode' argument. It should be either 'exact' or 'subset' (default).")
    
    precision, recall, f1_score = {}, {}, {}
    for role, tp in true_positives.items():
        precision[role] = tp / (tp+false_positives[role]) if tp+false_positives[role]!=0 else 0
        recall[role] = tp / (tp+false_negatives[role]) if tp+false_negatives[role]!=0 else 0
        f1_score[role] = 2*precision[role]*recall[role]/(precision[role]+recall[role]) \
                         if precision[role]+recall[role]!=0 else 0
        
    precision['macro_avg'] = sum(precision.values())/4
    recall['macro_avg'] = sum(recall.values())/4
    f1_score['macro_avg'] = sum(f1_score.values())/4
    # precision['micro_avg'] = sum(true_positives.values())/len(predicted_spans)
    # recall['micro_avg'] = sum(true_positives.values())/len(ground_truth_spans)

    true_positives['total'] = sum(true_positives.values())
    false_positives['total'] = sum(false_positives.values())
    false_negatives['total'] = sum(false_negatives.values())
    

    return {
        'TP': true_positives, 
        'FP': false_positives, 
        'FN': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'support': support,
    }

if __name__=='__main__':

    args = parser.parse_args()

    annotation_path = 'data/Style-Examples/annotation-src.json'
    result_path = 'evaluation/Style-Examples/prediction-src.json'
    
    annotations = read_annotation(annotation_path)
    predictions = read_prediction(result_path)
    i = 0
    true_scores = []
    pred_scores = []
    span_scores = []
    for anno, pred in zip(annotations, predictions):
        anno = char_to_token(anno, tokens=pred['tokens'])
        pred = pred

        if args.mode is None or args.mode=='clear-metric':
            true_scores.append(anno['score'])
            pred_scores.append(pred['score'])


        if args.mode is None or args.mode=='span-label':
            span_scores.append(evaluate_spans(pred['labels'], anno['labels'], args.span_mode))

        # i += 1
        # if i>10: break

    if args.mode is None or args.mode=='clear-metric':
        scores = evaluate_clear_metrics(true_scores, pred_scores)
        n = len(scores)
        classes = list(choices_dict.keys())
        print('=== ClearMetric Evaluation ===')
        for i, cls in enumerate(scores):
            print(f'{cls}: {classes[i]}')
            for metric in scores[cls]:
                if len(metric)==2:
                    print(f'  {metric}: {scores[cls][metric]}')
                else:
                    print(f'  {metric}: {scores[cls][metric]*100:.1f}%')

    if args.mode is None or args.mode=='span_label':
        n = len(span_scores)
        metrics = list(span_scores[0].keys())
        roles = list(span_scores[0][metrics[0]].keys())[:4]
        print('=== Span Labels Evaluation ===')
        for role in roles:
            print(f'[{role.upper()}]')
            for metric in metrics:
                if len(metric)==2:
                    s = sum([score[metric][role] for score in span_scores])
                    print(f'  {metric}: {s}')
                elif 'support' in metric:
                    s = sum([score[metric][role] for score in span_scores])
                    print(f'  {metric}: {s}')
                else:
                    s = sum([score[metric][role] for score in span_scores])/n
                    print(f'  {metric}: {s*100:.1f}%')

        print('TOTAL')
        for metric in ['TP', 'FP', 'FN']:
            s = sum([score[metric]['total'] for score in span_scores])
            print(f'  {metric}: {s}')

        for agg in ['macro_avg']:
            print(agg.upper())
            for metric in ['precision', 'recall', 'f1_score']:
                s = sum([score[metric][agg] for score in span_scores])/n
                print(f'  {metric}: {s*100:.1f}%')