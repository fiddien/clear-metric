import argparse
from parse import Sentence, load_models
from score import ClearMetric
from time import time

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
    
    sents = Sentence(sents_batch, *load_models())
    metric = ClearMetric(sents)
    scores = metric.score_sents()
    return scores, sents


def sents_iter(start, end, batch_size):

    if args.input_file:
        
        if end == -1:
            end = 9999999

        with open(args.input_file, encoding='utf-8') as text_file:
            
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

                yield sents_batch, start_batch, start_batch+batch_count
                
    else:
        sents = args.sentences.split('<sep>')
    
        if end == -1:
            end = len(sents)
        
        for start_batch in range(start, end, batch_size):
            end_batch = min(start_batch+batch_size, end, len(sents))
            yield sents[start_batch:end_batch], start_batch, end_batch


def save_results(scores, sents, output_file, start_sent_num):
    outputs = ''
    for i, score in enumerate(scores):
        out = ''
        out += f"Sent {i:>7d} : {sents.amrs.graphs[i].metadata['snt']}\n"
        out += f"  Scores     : {score}\n"

        out += f"  Structures :\n"
        if sents.structures[i]:
            for structure in sents.structures[i]:
                if structure:
                    out += "    S - V   : {} - {}\n".format(
                        ' '.join([t.text for t in structure['verb']]),
                        ' '.join([t.text for t in structure['subject']])
                    )
        else: 
            out += "    None\n"

        out += f"  Stories    :\n"
        if sents.stories[i]:
            for story in sents.stories[i]:
                if story:
                    out += "    C - A   : {} - {}\n".format(
                        ' '.join([t.text for t in story['action']]),
                        ' '.join([t.text for t in story['character']])
                    )
        else: 
            out += "    None\n"
       
        outputs += out
    
    if output_file:
        mode = 'w' if start_sent_num<=args.batch_size else 'a'
        with open(output_file, mode, encoding='utf-8') as f:
            f.write(outputs)
    else:
        print(outputs)



if __name__ == '__main__':
    args = parser.parse_args()
    
    for batch, i, j in sents_iter(args.start_from, args.end_at, args.batch_size):
        start_time = time()
        print("Processing sentence {} to {}".format(i, j-1))
        scores, descriptions = run_parsing(batch)
        save_results(scores, descriptions, args.output_file, i)
        print('Elapsed time:', time()-start_time)