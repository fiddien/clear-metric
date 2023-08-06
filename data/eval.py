import json
import re
import spacy
from spacy.lang.en import English
import argparse

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--mode", type=str, default=None,
                    help="Mode of evaluation: clear-metric, span-label. If none given, then it's both")
parser.add_argument("-sm", "--span-mode", type=str, default='partial',
                    help="Strictness of the span-labeling evaluation: 'partial' (default) or 'exact'.")

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
        precision = tp / (tp + fp) if (tp or fp) else None
        recall = tp / (tp + fn) if (tp or fn) else None
        if precision is not None and recall is not None:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = None

        class_metrics[f"rule {category+1}"] = {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Accuracy": accuracy,
        }

    return class_metrics


def evaluate_spans(predicted_labels, ground_truth_labels, mode='partial'):
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

    # Calculate true positives, false positives, and false negatives
    tp = {role:0 for role in roles}
    fp = {role:0 for role in roles}
    fn = {role:0 for role in roles}
    
    if mode=='partial':
        for i, (pred_role, pred_span) in enumerate(predicted_spans):
            match_found = any((pred_span.issubset(gt_span) and pred_role==gt_role) \
                               for gt_role, gt_span in ground_truth_spans)
            if match_found:
                tp[pred_role] += 1
            else:
                fp[pred_role] += 1

        for gt_role, gt_span in ground_truth_spans:
            match_found = any((gt_span.issubset(pred_span) and gt_role==pred_role) \
                               for pred_role, pred_span in predicted_spans)
            if not match_found:
                fn[gt_role] += 1

    elif mode=='exact':
        for pred_role, pred_span in predicted_spans:
            match_found = any((pred_span==gt_span and pred_role==gt_role) \
                               for gt_role, gt_span in ground_truth_spans)
            if match_found:
                tp[pred_role] += 1
            else:
                fp[pred_role] += 1

        for gt_role, gt_span in ground_truth_spans:
            match_found = any((gt_span==pred_span and gt_role==pred_role) \
                               for pred_role, pred_span in predicted_spans)
            if not match_found:
                fn[gt_role] += 1
    
    else:
        raise ValueError("Wrong 'mode' argument. It should be either 'exact' or 'subset' (default).")
    
    tp['total'] = sum(tp.values())
    fp['total'] = sum(fp.values())
    fn['total'] = sum(fn.values())

    support = {role: tp[role]+fn[role] for role in roles}
    support['total'] = sum(support.values())
    
    precision, recall, f1_score = {}, {}, {}
    for role, tp_ in tp.items():
        precision[role] = tp_ / (tp_+fp[role]) if tp_ or fp[role] else None
        recall[role] = tp_ / (tp_+fn[role]) if tp_ or fn[role] else None
        if precision[role] and recall[role]:
            f1_score[role] = 2*precision[role]*recall[role]/(precision[role]+recall[role])
        else:
            f1_score[role] = None
        
    # valid_precision = [p for p in precision.values() if p is not None]
    # precision['macro_avg'] = sum(valid_precision)/len(valid_precision) if valid_precision else None

    # valid_recall = [r for r in recall.values() if r is not None]
    # recall['macro_avg'] = sum(valid_recall)/len(valid_recall) if valid_recall else None
    
    # valid_f1 = [f for f in f1_score.values() if f is not None]
    # f1_score['macro_avg'] = sum(valid_f1)/len(valid_f1) if valid_f1 else None

    # precision['micro_avg'] = 0
    # recall['micro_avg'] = 0
    # f1_score['micro_avg'] = 0
    
    # for role in roles:
    #     n = support[role]
    #     if n > 0:
    #         precision['micro_avg'] += precision[role]/n if precision[role] is not None else 0
    #         recall['micro_avg'] += recall[role]/n if recall[role] is not None else 0
    #         f1_score['micro_avg'] += f1_score[role]/n if f1_score[role] is not None else 0


    return {
        'TP': tp, 'FP': fp, 'FN': fn,
        'Support': support,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
    }


def macro_average(tp_list, fp_list, fn_list):
    """
    Calculate macro-average precision, recall, and F1-score.

    Args:
        tp_list (list): List of true positives for each category.
        fp_list (list): List of false positives for each category.
        fn_list (list): List of false negatives for each category.

    Returns:
        macro_precision (float): Macro-average precision.
        macro_recall (float): Macro-average recall.
        macro_f1 (float): Macro-average F1-score.
    """
    num_categories = len(tp_list)

    macro_precision = sum(tp_list) / sum(tp_list + fp_list) if sum(tp_list + fp_list) > 0 else 0.0
    macro_recall = sum(tp_list) / sum(tp_list + fn_list) if sum(tp_list + fn_list) > 0 else 0.0

    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0

    return macro_precision, macro_recall, macro_f1


def micro_average(total_tp, total_fp, total_fn, total_tn=0):
    """
    Calculate micro-average precision, recall, and F1-score.
    """

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    return micro_precision, micro_recall, micro_f1


if __name__=='__main__':

    args = parser.parse_args()

    annotation_path = 'data/Style-Examples/annotation-src.json'
    result_path = 'evaluation/Style-Examples/prediction-src.json'
    print('Ground truth file data path :', annotation_path)
    print('Prediction file data path   :', result_path)
    
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
        # if i>100: break
    print(f'n =', len(pred_scores))

    confusion_metrics = ['TP', 'FP', 'FN', 'TN']
    metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']

    if args.mode is None or args.mode=='clear-metric':
        print('=== ClearMetric Evaluation: Predicting Rules Adherence of a Sentence ===')
        
        scores = evaluate_clear_metrics(true_scores, pred_scores)
        n = len(scores)
        classes = list(choices_dict.keys())
        
        rows = {c: f"{c: >10}" for c in scores}
        rows['total'] = f'{"TOTAL": >10}'
        rows['macro_avg'] = f'{"MACRO AVG": >10}' + ' '*40
        rows['micro_avg'] = f'{"MICRO AVG": >10}' + ' '*40
        header = " "*10
        for metric in confusion_metrics+metrics: 
            header += f'{metric: >10}'

        # macro_avg = {metric: 0 for metric in metrics}
        # n_macro_avg = {metric: 0 for metric in metrics}
        # micro_avg = {metric: 0 for metric in metrics}
        # n_micro_avg = {metric: 0 for metric in metrics}
        pr, rc, f1, ac = [], [], [], []
        tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
        
        for i, cls in enumerate(scores):
            totals = {metric: 0 for metric in confusion_metrics}
            tp = scores[cls]['TP']
            fp = scores[cls]['FP']
            fn = scores[cls]['FN']
            tn = scores[cls]['TN']
            tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn
            for val in [tp, fp, fn, tn]:
                rows[cls] += f'{val: >10}'
                

            precision, recall = None, None
            if tp or fp:
                precision = tp/(tp+fp)
                pr.append(precision)
                rows[cls] += f'{round(precision*100, 1): >10}'
            else:
                rows[cls] += f'{"NaN": >10}'

            if tp or fn:
                recall = tp/(tp+fn)
                rc.append(recall)
                rows[cls] += f'{round(recall*100, 1): >10}'
            else:
                rows[cls] += f'{"NaN": >10}'

            
            if precision and recall:
                f1_score = 2*precision*recall/(precision+recall)
                f1.append(f1_score)
                rows[cls] += f'{round(f1_score*100, 1): >10}'
            else:
                rows[cls] += f'{"NaN": >10}'

                
            accuracy = (tp+tn)/(tp+fp+fn+tn)
            ac.append(accuracy)
            rows[cls] += f'{round(accuracy*100, 1): >10}'
            


            # for metric in scores[cls]:
            #     if len(metric)==2:
            #         rows[cls] += f'{scores[cls][metric]: >10}'
            #         total[metric] += scores[cls][metric]
            #     else:
            #         if scores[cls][metric] is not None:
            #             rows[cls] += f'{round(scores[cls][metric]*100, 1): > 10}'
            #             macro_avg[metric] += scores[cls][metric]*100
            #             n_macro_avg[metric] += 1                        
            #         else:
            #             rows[cls] += f'{"NaN": >10}'

        # Calculate totals
        totals = [tp_total, fp_total, fn_total, tn_total]
        for tot in totals: 
            rows['total'] += f'{tot: >10}'

        # Calculate macro averages
        for val in [pr, rc, f1, ac]:
            rows['macro_avg'] += f'{round(sum(val)/len(val)*100, 1): >10}'

        # Calculate micro averages
        for mic_avg in micro_average(*totals):
            rows['micro_avg'] += f'{round(mic_avg*100, 1): >10}'

        print(header)
        for c in rows: print(rows[c])


    if args.mode is None or args.mode=='span_label':
        print(f'=== Span Labelling Evaluation ({args.span_mode} match) ===')
        n = len(span_scores)
        
        confusion_metrics = confusion_metrics[:-1]
        metrics = metrics[:-1]
        roles = list(span_scores[0][metrics[0]].keys())[:4]
        
        header = " "*10
        for metric in confusion_metrics + ['Support'] + metrics:
            header += f'{metric: >10}'
        
        rows = {role: f"{role: >10}" for role in roles}
        rows['total'] = f'{"TOTAL": >10}'
        rows['macro_avg'] = f'{"MACRO AVG": >10}' + ' '*40
        rows['micro_avg'] = f'{"MICRO AVG": >10}' + ' '*40

        tp_total, fp_total, fn_total = 0, 0, 0
        precision_total, recall_total, f1_score_total = 0, 0, 0
        for role in roles:
            tp = sum([score['TP'][role] for score in span_scores])
            fp = sum([score['FP'][role] for score in span_scores])
            fn = sum([score['FN'][role] for score in span_scores])
            tp_total += tp; fp_total += fp; fn_total += fn

            sup = sum([score['Support'][role] for score in span_scores])
            
            for vals in [tp, fp, fn, sup]:
                rows[role] += f'{vals: >10}'
            
            if tp or fp:
                precision = tp/(tp+fp)
                precision_total += precision
                rows[role] += f'{round(precision*100, 1): >10}'
            else:
                rows[role] += f'{"NaN": >10}'

            if tp or fn:
                recall = tp/(tp+fn)
                recall_total += recall
                rows[role] += f'{round(recall*100, 1): >10}'
            else:
                rows[role] += f'{"NaN": >10}'
            
            if precision and recall:
                f1_score = 2*precision*recall/(precision+recall)
                f1_score_total += f1_score
                rows[role] += f'{round(f1_score*100, 1): >10}'
            else:
                rows[role] += f'{"NaN": >10}'
                

        # Calculate totals
        totals = []
        for metric in ['TP', 'FP', 'FN', 'Support']:
            s = sum([score[metric]['total'] for score in span_scores])
            totals.append(s)
            rows['total'] += f'{s: >10}'

        # Calculate macro averages
        for total in [precision_total, recall_total, f1_score_total]:
            rows['macro_avg'] += f'{round(total/len(roles)*100, 1): >10}'

        # Calculate micro averages
        for mic_avg in micro_average(tp_total, fp_total, fn_total):
            rows['micro_avg'] += f'{round(mic_avg*100, 1): >10}'
            

        print(header)
        for c in rows: print(rows[c])