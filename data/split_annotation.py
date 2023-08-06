import json

ann = json.load(open('ParaSCI-arXiv-train/ParaSCI-arXiv.json', 'r', encoding='utf-8'))

print(len(ann))
a1 = ann[::2]
a2 = ann[1::2]
print(len(a1), len(a2))
print(a1[0]['data']['text'])
print(a2[0]['data']['text'])

json.dump(a1, open('ParaSCI-arXiv-train/annotation-src.json', 'w'))
json.dump(a2, open('ParaSCI-arXiv-train/annotation-tgt.json', 'w'))
