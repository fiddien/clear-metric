import json

ann = json.load(open('data/Style-Examples/annotation_all.json', 'r', encoding='utf-8'))

print(len(ann))
a1 = ann[::2]
a2 = ann[1::2]
print(len(a1), len(a2))
print(a1[0]['data']['text'])
print(a2[0]['data']['text'])

json.dump(a1, open('data/Style-Examples/annotation-src.json', 'w'))
json.dump(a2, open('data/Style-Examples/annotation-tgt.json', 'w'))
