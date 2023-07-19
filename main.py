import argparse
from parse import Sentence
from score import ClearMetric
from time import time
import json, os

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
parser.add_argument("-S", "--start-from", type=int, default=0,
                    help="Start sentences from certain line number.")
parser.add_argument("-E", "--end-at", type=int, default=-1,
                    help="End sentences at certain line number.")



def run_parsing(sents_batch):
    
    sents = Sentence(sents_batch)
    metric = ClearMetric(sents)
    scores = metric.score_sents()
    return scores, sents


def sents_iter(start, end, batch_size=8, input_file_path=None, sentences=None):

    if input_file_path:
        
        if end == -1:
            end = 9999999

        with open(input_file_path, encoding='utf-8') as text_file:
            
            if start > 0:
                for _ in range(start):
                    line = next(text_file, None)
            rem_count = end - start
            start_batch, batch_count = start, 0
            line = next(text_file, None)

            while line and rem_count > 0:
                start_batch += batch_count
                sents_batch = []
                batch_count = 0

                while line and batch_count < min(batch_size, rem_count):
                    sents_batch.append(line.rstrip())
                    line = next(text_file, None)
                    batch_count += 1
                rem_count -= batch_count

                yield sents_batch, start_batch, start_batch+batch_count-1
                
    else:
        assert isinstance(sentences, str)

        sents = sentences.split('<sep>')
        if len(sents)==1:
            sents = sentences.split('\n')
    
        if end == -1:
            end = len(sents)
        
        for start_batch in range(start, end, batch_size):
            end_batch = min(start_batch+batch_size, end, len(sents))
            yield sents[start_batch:end_batch], start_batch, end_batch-1


def save_results(scores, sents, output_file=None, start_sent_num=0, batch_size=0):
    outputs = []
    
    for i, out in enumerate(sents.to_json_format(scores)):
        # outputs += sent
        # outputs += f"  Scores     : {score}\n"
        out['id'] = start_sent_num + i
        outputs.append(out)
        
    if output_file:
        # mode = 'w' if start_sent_num<=batch_size else 'a'
        mode = 'a'
        with open(output_file, mode, encoding='utf-8') as f:
            outputs = json.dumps(outputs)
            f.write(outputs)
    else:
        print(outputs)



if __name__ == '__main__':
    args = parser.parse_args()
    
    sents_batch = sents_iter(
        args.start_from, args.end_at, args.batch_size, args.input_file, args.sentences
    )
    for batch, i, j in sents_batch:
        start_time = time()
        print("Processing sentence {} to {}".format(i, j))
        scores, descriptions = run_parsing(batch)
        save_results(scores, descriptions, args.output_file+'.temp', i, args.batch_size)
        print('Elapsed time:', time()-start_time)

    # Post-process the output file
    if args.output_file:
        with open(args.output_file+'.temp', 'r') as infile, \
             open(args.output_file, 'w') as outfile:
            data = infile.read()
            data = data.replace("][", ",")
            outfile.write(data)
        os.remove(args.output_file+'.temp')