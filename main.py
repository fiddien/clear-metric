import argparse
import amrlib
from amrlib.graph_processing.annotator import add_lemmas, annotate_penman
from amrlib.alignments.rbw_aligner import RBWAligner
import spacy
from tqdm import tqdm
from time import sleep


parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--sentences", type=str,
                    help="List of sentences to score. Separate each sentences with <sep> token.")
parser.add_argument("-ac", "--action-character-only", action='store_true',
                    help="Only outputs the (action, character) pair")
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


stog = amrlib.load_stog_model(batch_size=16)
nlp = spacy.load('en_core_web_sm')

# === AMR parsing happens here to determine the character and action of the sentence ===

# === AMR parsing happens here to determine the character and action of the sentence ===

class AMR:
    def __init__(self, sents, amr_parser=stog):
        self.sents = sents
        graphs = amr_parser.parse_sents(sents, disable_progress=False)
        self.graphs_str = graphs
        self.aligned = [self.align_vars(g) for g in graphs]

    def align_vars(self, graph_str):
        """ 
        Align the nodes of each AMR graphs to a token in the respective sentence string.
        Argument:
            graph_str: (str) AMR graph
        Return: 
            a (dict) that contain the Penman graph and the collection nodes in the graph
            accompanied by information about the tokens, id of the tokens (token number)
            and each corresponding variable name
        """
        assert isinstance(graph_str, str)

        penman_graph = add_lemmas(graph_str, snt_key='snt')
        penman_graph = annotate_penman(penman_graph)
        aligned_graph = RBWAligner.from_penman_w_json(penman_graph)
        nodes_alignment = {}

        # Nodes that can be aligned to the strings/tokens
        for i, (a, t) in enumerate(zip(aligned_graph.alignments, aligned_graph.tokens)):
            if a:
                nodes_alignment[a.triple[-1]] = {
                    'token': t,
                    'token_id': i,
                    'var': a.triple[0],
                }

        # Nodes that does not have an alignment
        # for node in penman_graph.instances():
        #     if node.target not in nodes_alignment:
        #         nodes_alignment[node.target] = {
        #             'token': '',
        #             'token_id': None,
        #             'var': node.source,
        #         }

        return {
            'penman': penman_graph,
            'aligned_nodes': nodes_alignment,
        }


def get_concept_map(graph):
    """
    Create a dictionary with AMR graph variables as the keys 
    and their corresponding concept as the values.
    Argument: 
        graph: (penman.graph)
    Return: 
        (dict) that contains {variable1: concept1, variable2: concept2}
        Example: {'w': 'wants-01'}
    """
    return {x.source: x.target for x in graph.instances()}


def get_role_map(graph):
    """
    Create a dictionary with tuples of two neighboring AMR graph nodes as the keys
    and their connection/role as the values.
    Argument:
        graph: (penman.graph)
    Return:
        dictionary: {(source node, target node): role}
    """
    return {(e.source, e.target): e.role for e in graph.edges()}


def get_children(graph, var):
    """
    Return the first child of a node.
    Arguments:
        graph: a Penman graph
        var: (str) the variable name of the node whose children wants to be returned
    Return:
        children: (tuple) child variable and its role
    """
    children = [(t, r) for (s, r, t) in graph.triples if s==var and r!=':instance']
    if var.isdigit() and children:
        return children[1]
    return children[0] if children else (None, None)


def get_actn_char_candidates(graph, aligned_nodes):
    """
    Argument:
        graph: (penman.graph)
    Return:
        (list) candidates of action and character pairs
    """
    concept_map = get_concept_map(graph)
    role_map = get_role_map(graph)
    candidates = []

    for (source, target), role in role_map.items():

        if concept_map[source] in aligned_nodes:
            
            if role in [':ARG0', ':ARG1']:

                # If the node is an intermediete node (aligned into an entity)
                if concept_map[target] in ['person', 'country', 'and']:
                    role_child = ''

                    # Traverse through the graph
                    while concept_map[target] not in aligned_nodes:
                        # print(target, graph.metadata['snt'])
                        target_child, role_child = get_children(graph, target)
                        
                        # if role_child=='name':
                        #     att = graph.attribute()
                        #     target_child = [x.target for x in att if x.source==target_child][0]
                        
                        if target_child:
                            target = target_child
                        else:
                            break

                    if target not in concept_map:
                        # The action-character tuple
                        pair = (concept_map[source], target)
                    else:
                        pair = (concept_map[source], concept_map[target])
                else:
                    pair = (concept_map[source], concept_map[target])
                candidates.append(pair)
        

    return candidates


def get_actn_char(aligned_amrs, return_results=True, print_results=False,
                  return_graphs=False):
    """
    Arguments:
        sents: (list) of sentences (str)
        amr_parser: one of (amrlib.models)
        print_results: (bool)
        return_results: (bool)
        return_graphs: (bool)
    Return:
        (list) of action-character pair candidates
        The candidates are in the form of tuples of token (str), token id (int), and its propbank name (str).
    """
    graphs = aligned_amrs

    for i, g in enumerate(graphs):
        print(g['penman'].metadata)
        candidates = get_actn_char_candidates(g['penman'], g['aligned_nodes'])
        if candidates:
            graphs[i]['candidates'] = []
            # Store the token, token id, and the concept
            for actn, char in candidates:
                actn_token, char_token = ('', None, None), ('', None, None)
                if actn in g['aligned_nodes']:
                    actn_token = (
                        g['aligned_nodes'][actn]['token'], 
                        g['aligned_nodes'][actn]['token_id'], 
                        actn
                    )
                if char in g['aligned_nodes']:
                    char_token = (
                        g['aligned_nodes'][char]['token'], 
                        g['aligned_nodes'][char]['token_id'], 
                        char
                    )
                graphs[i]['candidates'].append((actn_token, char_token))
        else:
            graphs[i]['candidates'] = [(('', None, None),('', None, None))]

    if print_results:
        for g in graphs:
            actn, char = g['candidates'][0] # Only print the first candidate
            print("Sentence    : {}".format(g['penman'].metadata['snt']))
            print("  Character : {} [id={}]".format(char[0], char[1]))
            print("  Action    : {} [id={}]".format(actn[0], actn[1]))
    if return_graphs:
        return [g['candidates'][0][:2] for g in graphs], graphs
    if return_results:
        return [g['candidates'][0][:2] for g in graphs]
    


# === Dependency parsing happens here to analyse the sentence syntactic structure ===

class Sentence():
    def __init__(self, sentence, model=nlp, action=None, character=None):
        self.snt = sentence
        self.model = model
        self.action = (action[1], action[0]) if action else (-1, '')
        self.character = (character[1], character[0]) if character else [(-1, '')]
        self.NP, self.VP = None, None
        self.NP_pos, self.VP_pos = (0, 0), (0, 0)
        self.subject, self.verb = [(-1, '')], [(-1, '')]
        self.doc = self.parse()
        self.root = self.get_root()

    def parse(self):
        return self.model(self.snt)


    def get_root(self):
        for token in self.doc:
            if token.dep_ == 'ROOT':
                return token
        return None


    def get_np(self):
        """ Get the noun phrase which directly depends on the root (main verb). """
        for token in self.doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'expl'] and token.head == self.root:
                NP_unsorted = self._extract_dependents(token)
                NP = [t for t in sorted(NP_unsorted, key=lambda x: x.i)]

                self.subject, self.NP = [], []
                for i, t in enumerate(NP):
                    if t.text == ',' and i > 0:
                        break
                    self.subject.append((t.i, t.text, t.pos_))
                    self.NP.append(t)
                
                self.NP_pos = (self.NP[0].i, self.NP[-1].i)
                self.NP = ' '.join([t.text for t in self.NP])
                return self.NP
        return None


    def get_vp(self):
        """ Get the verb phrase which includes the root (main verb). """
        VP = [self.root]
        for child in self._extract_dependents(self.root):
            if child.dep_ in ['aux', 'auxpass']:
                VP.append(child)

        VP_group = self._find_consecutive_tokens(VP)
        if VP_group:
            for group in VP_group:
                if self.root in group:
                    VP = group
                    break
        else:
            VP = [self.root]

        self.verb = (self.root.i, self.root.text)
        self.VP_pos = (VP[0].i, VP[-1].i)
        self.VP = ' '.join([t.text for t in VP])
        return self.VP


    def _find_consecutive_tokens(self, lst):
        lst = [t for t in sorted(lst, key=lambda x: x.i)]
        consecutive_groups = []
        current_group = [lst[0]]

        for i in range(1, len(lst)):
            if lst[i].i == lst[i-1].i + 1:
                current_group.append(lst[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [lst[i]]

        consecutive_groups.append(current_group)
        consecutive_tuples = [
            group for group in consecutive_groups if len(group) > 1
        ]

        return consecutive_tuples


    def _extract_dependents(self, token):
        return [child for child in token.subtree]


    def score_character(self):
        subject = [s[:2] for s in self.subject]
        return 0 if self.character in subject else 1


    def score_action(self):
        return 0 if self.action == self.verb else 1


    def score_long_abstract_subject(self):
        # The "abstract" part is not implemented here
        return 0 if len(self.subject) <= 8 else 1


    def score_long_intro_phrases_clauses(self):
        return 0 if self.NP_pos[0] <= 8 else 1


    def score_subj_verb_connection(self):
        space_NP_VP = self.VP_pos[0] - self.NP_pos[1] - 1
        return 0 if space_NP_VP <= 1 else 1


    def get_scores(self):
        self.get_vp()
        self.get_np()
        scores = [
            self.score_character(),
            self.score_action(),
            self.score_long_abstract_subject(),
            self.score_long_intro_phrases_clauses(),
            self.score_subj_verb_connection(),
        ]
        self.scores = scores
        return scores


def print_actn_char(sents, actn_char_pairs, sent_num, save_result=False):
    output = ''
    for sent, (actn, char) in zip(sents, actn_char_pairs):
        out = ''
        out += f"Sentence {sent_num:>4d} : {sent}\n"
        out += f"  Action      : {actn[0]} [id={actn[1]}]\n"
        out += f"  Character   : {char[0]} [id={char[1]}]\n"
        print(out)

        sent_num += 1

        if save_result:
            output += out
    
    if save_result:
        mode = 'w' if sent_num<=batch_size else 'a'
        with open(args.output_file, mode, encoding='utf-8') as f:
            f.write(output)


def print_all(sents, actn_char_pairs, sent_num, save_result=False):
    output = ''
    for sent, (actn, char) in tqdm(zip(sents, actn_char_pairs)):
        
        s = Sentence(sent, action=actn, character=char)
        s.get_scores()

        out = ''
        out += f"Sentence {sent_num:>4d} : {sent}\n"
        out += f"  Action      : {s.action[1]} [id={s.action[0]}]\n"
        out += f"  Verb        : {s.VP} [id={s.VP_pos}]\n"
        out += f"  Character   : {s.character[1]} [id={s.character[0]}]\n"
        out += f"  Subject     : {s.NP} [id={s.NP_pos}]\n"
        out += f"  Scores      : {s.scores}\n"
        out += f"  Total sc.   : {sum(s.scores)}\n"

        print(out)
    
        sent_num += 1
        sleep(0.01)

        if save_result:
            output += out
    
    if save_result:
        mode = 'w' if sent_num<=batch_size else 'a'
        with open(args.output_file, mode, encoding='utf-8') as f:
            f.write(output)

def run_parsing(sents_batch, action_character_only, output_file=None, count_start=0):
    
    print('Parsing semantics...')
    graphs = AMR(sents_batch)
    actn_char_pairs = get_actn_char(graphs.aligned)
    
    if action_character_only:
        result = print_actn_char(sents_batch, actn_char_pairs, sent_num=count_start, save_result=output_file)
    else:
        print('Parsing syntax...')
        result = print_all(sents_batch, actn_char_pairs, sent_num=count_start, save_result=output_file)


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
        run_parsing(sents_batch, args.action_character_only, args.output_file, i)
        
        if i > args.end_at:
            break