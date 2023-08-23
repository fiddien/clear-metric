import argparse
import json

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("folder", type=str, help="")

args = parser.parse_args()

pred1 = json.load(open('data/' + args.folder + '/prediction-original.json', 'r', encoding='utf-8'))
pred2 = json.load(open('data/' + args.folder + '/prediction-paraphrase.json', 'r', encoding='utf-8'))

print(len(pred1), len(pred2))
pred = []
for a, b in zip(pred1, pred2):
    pred.append(a)
    pred.append(b)
print(len(pred))

json.dump(pred, open('data/' + args.folder + '/prediction-all.json', 'w'))