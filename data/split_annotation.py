import argparse
import json

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("annotation_folder", type=str, help="")

args = parser.parse_args()

ann = json.load(open('data/' + args.annotation_folder + '/annotation-all.json', 'r', encoding='utf-8'))

print(len(ann))
a1 = ann[::2]
a2 = ann[1::2]
print(len(a1), len(a2))
print(a1[0]['data']['text'])
print(a2[0]['data']['text'])

json.dump(a1, open('data/' + args.annotation_folder + '/annotation-original.json', 'w'))
json.dump(a2, open('data/' + args.annotation_folder + '/annotation-paraphrase.json', 'w'))
