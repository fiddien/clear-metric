import spacy, benepar
import amrlib
from amrlib.graph_processing.annotator import add_lemmas, annotate_penman, load_spacy
from amrlib.alignments.rbw_aligner import RBWAligner
import json

spacy_model_name = 'en_core_web_md'
nlp, stog = None, None

def load_models(mode=['spacy', 'amr']):
    # Lazy loader
    # benepar.download('benepar_en3')
    if 'spacy' in mode:
        global nlp
        if not nlp:
            model = spacy.load(spacy_model_name)
            model.add_pipe('benepar', config={'model': 'benepar_en3'})
            load_spacy(spacy_model_name)
            nlp = model
        return model

    if 'amr' in mode:
        global stog
        if not stog:
            model = amrlib.load_stog_model()
            model = model.parse_sents
            stog = model
        return model

# === Semantic parsing happens here to determine the character and action of the sentence ===

class Node:
    def __init__(self, i, var, concept=None, role=None, text='',
                 start=None, end=None):
        self.i = i
        self.start = start
        self.end = end
        
        self.text = text
        self.var = var
        self.concept = concept
        self.role = role

    def __repr__(self):
        return self.text
    

class GraphAligner:
    def __init__(self, graphs, sents=[]):
        assert isinstance(graphs, list)
        assert isinstance(graphs[0], str)
        self.graphs = graphs
        
        if isinstance(sents, spacy.tokens.doc.Doc):
            self.align_var = self.align_nodes(sents)
        else:
            self.align_var = self.align_nodes()


    def align_nodes(self, sents=[], add_tokens=False):
        """
        Align the nodes of each AMR graphs to a token in the respective sentence string.
        Argument:
            graph_str: (str) AMR graph
        Return:
            a (dict) that contain the Penman graph and the collection nodes in the graph
            accompanied by information about the tokens, id of the tokens (token number)
            and each corresponding variable name
        """

        alignments = []
        if sents:
            iter = zip(self.graphs, sents)
        else:
            iter = zip(self.graphs, [[]]*len(self.graphs))
            
        for idx, (graph_str, sent) in enumerate(iter):
            
            tokens = [t for t in sent] if add_tokens else None
            penman_graph = add_lemmas(graph_str, snt_key='snt')
            penman_graph = annotate_penman(penman_graph, 
                                           tokens=tokens)
            aligned_graph = RBWAligner.from_penman_w_json(penman_graph)
            self.graphs[idx] = aligned_graph.get_penman_graph()
            sent = penman_graph.metadata['snt']
            
            nodes_alignment = {}
            start_idx = 0
            # Nodes that can be aligned to the strings/tokens
            for i, (align, token) in enumerate(zip(aligned_graph.alignments,
                                                   aligned_graph.tokens)):
                end_idx = start_idx + len(token)

                if align:
                    var = align.triple[0] if align.is_concept() else align.triple[1]
                    role = align.triple[1] if align.is_role() else None
                    concept = align.triple[2] if align.is_concept() else None

                    if var not in nodes_alignment:
                        nodes_alignment[var] = [
                            Node(i, var, concept, role, token, 
                                 start_idx, end_idx)
                        ]
                    else:
                        nodes_alignment[var].append(
                            Node(i, var, concept, role, token,
                                 start_idx, end_idx)
                        )
                
                start_idx += sent[end_idx:end_idx+2].count(' ')
                start_idx = end_idx + 1

            alignments.append(nodes_alignment)

        return alignments

    
    def _get_role_map(self, idx):
        """
        Create a dict of dict of relations within the AMR graph.
        The first keys are the source nodes. The second keys are the target nodes.
        The values are the relation between the nodes.
        Argument:
            idx: (int) index number of the graph
        Return:
            dictionary: {(source node, target node): role}
        """
        relations = {}
        for edge in self.graphs[idx].edges():
            if edge.source not in relations:
                relations[edge.source] = {edge.target: edge.role}
            else:
                relations[edge.source][edge.target] = edge.role
        return relations


    @staticmethod
    def _check_alignment(alignment, role, v1, v2):

        if role[v1][v2]==':ARG0':
            
            # Check if v1 and v2 are aligned
            if v1 in alignment: 
                
                if v2 in alignment: 
                    # Return both nodes' alignments
                    return alignment[v1], alignment[v2]
                else: 
                    # v2 is not aligned, check its children
                    if v2 in role:
                        # Iterate through v2's children nodes
                        for v2_child in role[v2]: 
                            if v2_child in alignment:
                                return alignment[v1], alignment[v2_child]
        
        elif role[v1][v2]==':ARG0-of':
            
            # Check if v1 and v2 are aligned
            if v1 in alignment: 

                if v2 in alignment: 
                    # Return both nodes' alignments
                    return alignment[v2], alignment[v1]
                else: 
                    # v2 is not aligned, check its children?
                    pass
        
        return None, None
                        

    def get_actn_char(self):
        """
        Return:
            (list) candidates of action and character pairs
        """
        for idx, aligned in enumerate(self.align_var):
            
            role = self._get_role_map(idx)
            candidates = []
            
            for source in role.keys():
                
                for target in role[source]:

                    act, char = self._check_alignment(aligned, role, source, target)
                    if act:
                        candidates.append({
                                'action': act, 
                                'character': char
                            })
                    
            yield candidates


# === Syntatic parsing happens here to analyse get the verb and subject of the sentence ===
class Sentence:
    def __init__(self, sentences, syntax_model=None, semantic_model=None):
        if not syntax_model:
            syntax_model = load_models('spacy')

        self.docs = self.parse_syntax(syntax_model, sentences)
        assert len(list(self.docs.sents)) == len(sentences)
        self.structures = self.get_structures()

        if not semantic_model:
            semantic_model = load_models('amr')

        self.amrs = self.parse_semantic(semantic_model, sentences)
        self.stories = list(self.amrs.get_actn_char())



    def __len__(self):
        return len(self.docs)

    
    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        i = self.i
        if i == len(self.amrs.graphs):
            raise StopIteration
        
        out = ''
        out += f"Sentence     : {self.amrs.graphs[i].metadata['snt']}\n"
        out += f"  Structures :\n"
        if self.structures[i]:
            for structure in self.structures[i]:
                if structure:
                    out += "    S - V    : {} - {}\n".format(
                        ' '.join([t.text for t in structure['subject']]),
                        ' '.join([t.text for t in structure['verb']]),
                    )
        else: 
            out += "    None\n"
        out += f"  Stories    :\n"
        if self.stories[i]:
            for story in self.stories[i]:
                if story:
                    out += "    C - A    : {} - {}\n".format(
                        ' '.join([t.text for t in story['character']]),
                        ' '.join([t.text for t in story['action']]),
                    )
        else: 
            out += "    None\n"
        self.i += 1
        return out


    def get_sentences(self):
        return self.docs.sents


    def parse_syntax(self, syntax_model, sentences):        
        if isinstance(sentences, list) and isinstance(sentences[0], str):
            sentences = " ".join(sentences)
        return syntax_model(sentences)


    def parse_semantic(self, semantic_model, sentences):
        return GraphAligner(semantic_model(sentences), self.docs)


    def to_json_format(self, scores=None):

        if not scores:
            scores = [None] * len(self)
    
        for sent, labels, score in zip(self.get_sentences(), self.get_labels(), scores):
            result = []
            for item in labels:
                if 'character' in item:
                    char = ' '.join([t.text for t in item['character']])
                    char_start = item['character'][0].start
                    char_end = item['character'][-1].end
                    
                    actn = ' '.join([t.text for t in item['action']])
                    actn_start = item['action'][0].start
                    actn_end = item['character'][-1].end

                    result.append({
                        'character': {
                            'text': char,
                            'start': char_start,
                            'end': char_end,
                        },
                        'action': {
                            'text': actn,
                            'start': actn_start,
                            'end': actn_end,
                        }
                    })
                
                elif 'subject' in item:
                    first_word = item['subject'][0].sent[0]
                    subj = ' '.join([t.text for t in item['subject']])
                    subj_start = item['subject'][0].idx - first_word.idx
                    subj_end = subj_start + len(subj)
                    
                    verb = ' '.join([t.text for t in item['verb']])
                    verb_start = item['verb'][0].idx - first_word.idx
                    verb_end = verb_start + len(verb)

                    result.append({
                        'subject': {
                            'text': subj,
                            'start': subj_start,
                            'end': subj_end,
                        },
                        'verb': {
                            'text': verb,
                            'start': verb_start,
                            'end': verb_end,
                        }
                    })
            yield {'sent': sent.text, 'score': score, 'labels': result}
        

    def get_labels(self):
        for structure, story in zip(self.structures, self.stories):
            yield structure + story


    @staticmethod
    def is_main(verb):
        for token in verb:
            if token.dep_=='ROOT':
                return True
        return False
    

    def get_structures(self):

        
        Verbs = [v for v in self.get_verbs()]
        Subjects = [s for s in self.get_subjects(Verbs)]

        structures = []
        for verbs, subjects in zip(Verbs, Subjects):
            vs = []
            if verbs:
                if len(verbs) < len(subjects):
                    verbs = verbs * len(subjects)
                elif len(verbs) > len(subjects):
                    subjects = subjects * len(verbs)

                for v, s in zip(verbs, subjects):
                    if isinstance(s[0], list) and len(s)>1:
                        for s_ in s:
                            
                            vs.append({'verb': v, 
                                       'subject': s_, 
                                       'main': self.is_main(v)})
                    elif isinstance(v[0], list) and len(v)>1:
                        for v_ in v:
                            vs.append({'verb': v_, 
                                       'subject': s, 
                                       'main': self.is_main(v_)})
                    else:
                        vs.append({'verb': v, 
                                   'subject': s, 
                                   'main': self.is_main(v)})
            
            # Move the main clause to the front of the list
            changed = False
            for item in vs:
                if item['main']:
                    main_clause = item
                    vs.remove(item)
                    changed = True
                    break
            if changed:
                vs = [main_clause] + vs

            structures.append(vs)
        return structures


    def get_verbs(self):
        """ Get the verb phrase which includes the root (main verb). """
        dependents = [
            'aux',
            'auxpass',
            # 'advmod',
            'prt',
            'neg',
            'ccomp',
        ]
        for sent in list(self.docs.sents):
            # Capture the case which theire is no verb/auxilary word
            if sent.root.pos_ not in ['VERB', 'AUX']:
                yield []

            verbs = [sent.root]
            # Go through the dependents of the root
            for child in sent.root.children:
                # Capture the surrounding words
                if child.dep_ in dependents and child.pos_!='NOUN':
                    verbs.append(child)
                # Capture conjugate verbs
                elif child.dep_=='conj' and child.pos_=='VERB':
                    verbs.append(child)

            verbs_group = self._group_consecutive_tokens(verbs)
            if verbs_group:
                yield verbs_group
            else:
                yield [[sent.root]]


    def get_subjects(self, Verbs):
        """ Get the noun phrase which directly depends on the verbs """
        for sent, verbs in zip(self.docs.sents, Verbs):
            subjects = []

            for verb in verbs:
                for token in verb:
                    if token.pos_=='VERB' or token==sent.root:
                        nouns = self._get_noun(token)
                        if nouns:
                            subjects.append(nouns)
            yield subjects


    @staticmethod
    def _get_noun(verb):
        dependents = [
            'nsubj',     # active sentence
            'nsubjpass', # passive sentence
            'expl',      # captures an existential *there* or *it* in extraposition constructions
        ]
        subject = []
        for token in verb.children:
            if token.dep_ in dependents:
                subject.append(token)
                for child in token.children:

                    if child.dep_ in ['acl', 'relcl', 'appos', 'prep']:
                        break
                    subject.append(child)

                    for gchild in child.children:
                        if gchild.dep_ in ['det', 'nummod']:
                            subject.append(gchild)

                subject = list(sorted(subject, key=lambda x: x.i))

                # Delete trailing punctuation(s)
                if subject[0].pos_ == 'PUNCT':
                    subject = subject[1:]

                if subject[-1].pos_ == 'PUNCT':
                    subject = subject[:-1]

        split_pos = [i for i, token in enumerate(subject) \
                    if token.text.lower()=='and']
        if split_pos:
            subject_ = []
            for pos in split_pos:
                subject_ += [subject[:pos]]
                subject = subject[pos+1:]
            subject_ += [subject]
            subject = subject_

        return subject


    @staticmethod
    def _group_consecutive_tokens(seq):
        """seq: (list) tokens"""
        if not seq:
            return seq

        if len(seq)==1:
            return [seq]
        
        seq = [t for t in sorted(seq, key=lambda x: x.i)]
        grouped = [[seq[0]]]
        
        for x in seq[1:]:
            if x.i == grouped[-1][-1].i + 1:
                grouped[-1].append(x)
            else:
                grouped.append([x])
        
        return grouped
