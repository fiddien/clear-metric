import json


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


def read_annotation(path_to_json_file):

    annotations = json.load(open(path_to_json_file))

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
                    relations[-1]['character'].pop('labels')
                    relations[-1]['action'].pop('labels')
                if 'Subject' in labels[src]['labels'] and 'Verb' in labels[tgt]['labels']:
                    relations.append(
                        {'subject': labels[src], 'verb': labels[tgt]}
                    )
                    relations[-1]['subject'].pop('labels')
                    relations[-1]['verb'].pop('labels')
                
        yield {'sent': sentence, 'score': score, 'labels': relations}


def read_result(path_to_json_file):

    items = json.load(open(path_to_json_file))

    for item in items:
        yield item


if __name__=='__main__':
    annotation_path = 'data/Style-Examples/annotation.json'
    result_path = 'evaluation/style-inference-2023-07-19.json'
    
    ra = read_annotation(annotation_path)
    rb = read_result(result_path)
    count_same = 0
    count_all = 0
    for a, b in zip(ra, rb):
        print(a['sent'])
        print(len(a['labels']), len(b['labels']))
        if len(a['labels'])==len(b['labels']):
            count_same += 1
        count_all += 1
        for a_ in a['labels']:
            print(a_)
        print()
        for b_ in b['labels']:
            print(b_)
        print();print()
        if count_all > 5: break

    print(count_same, count_same/count_all)