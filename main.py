import argparse
from parse import Sentence
from clear_metric import ClearMetric

parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--sentences", type=str,
                    help="List of sentences to score. Separate each sentences with <sep> token.")
parser.add_argument("-cs", "--clear-scores-only", action='store_true',
                    help="Only outputs the (action, character) pair")
parser.add_argument("-i", "--input-file", type=str,
                    help="A file that contains sentences to score. Separate different sentences by a new line.")
parser.add_argument("-o", "--output-file", type=str,
                    help="Store the scores into a text file.")
parser.add_argument("-b", "--batch-size", type=int, default=500,
                    help="Process the sentences in batch.")
parser.add_argument("-sf", "--start-from", type=int, default=0,
                    help="Start sentences from certain line number.")
parser.add_argument("-ea", "--end-at", type=int, default=-1,
                    help="End sentences at certain line number.")



def print_all(scores, sents, sent_num, save_result=False):
    outputs = ''
    for i, score in enumerate(scores):
        out = ''
        out += f"Sent {i:>7d} : {sents.amrs.graphs[i].metadata['snt']}\n"
        out += f"  Scores     : {score}\n"
       
        outputs += out
    
    if save_result:
        mode = 'w' if sent_num<=batch_size else 'a'
        with open(args.output_file, mode, encoding='utf-8') as f:
            f.write(outputs)
    else:
        print(outputs)


def run_parsing(sents_batch, output_file=None, count_start=0):
    
    sents = Sentence(sents_batch)
    metric = ClearMetric(sents)
    scores = metric.score_sents()
    return scores, sents


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size

    
    if args.input_file is not None:
        sents = [line.rstrip() for line in open(args.input_file, encoding='utf-8') if line!='']
    else:
        sents = args.sentences.split('<sep>')
    
    if args.end_at == -1:
        args.end_at = len(sents)
    
    for i in range(0, len(sents), batch_size):
        i += args.start_from
        j = min(len(sents), i+batch_size, args.end_at)
        sents_batch = sents[i:j]

        print("Processing sentence {} to {}".format(i, j-1))
        scores, results = run_parsing(sents_batch, args.output_file, i)
        print_all(scores, results, i)
        
        
        if i > args.end_at:
            break