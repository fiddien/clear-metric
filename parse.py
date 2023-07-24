import spacy, benepar
import amrlib
from amrlib.graph_processing.annotator import add_lemmas, annotate_penman, load_spacy
from amrlib.alignments.rbw_aligner import RBWAligner

spacy_model_name = 'en_core_web_md'
nlp, stog = None, None

def load_models(mode=['syntax', 'semantic']):
    # Lazy loader
    if 'syntax' in mode:
        global nlp
        if not nlp:
            model = spacy.load(spacy_model_name)
            try:
                model.add_pipe('benepar', config={'model': 'benepar_en3'})
            except:
                benepar.download('benepar_en3')
                model.add_pipe('benepar', config={'model': 'benepar_en3'})
            load_spacy(spacy_model_name)
            nlp = model
            return model
        else:
            return nlp

    if 'semantic' in mode:
        global stog
        if not stog:
            model = amrlib.load_stog_model()
            model = model.parse_sents
            stog = model
            return model
        else:
            return stog

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
            self.align_var = self.align_nodes(sents.sents)
        else:
            self.align_var = self.align_nodes()


    def align_nodes(self, spacy_sents=None):
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
        if spacy_sents:
            iter = zip(self.graphs, spacy_sents)
        else:
            iter = zip(self.graphs, [[]]*len(self.graphs))
            
        for idx, (graph_str, sent) in enumerate(iter):
            
            tokens = [t for t in sent] if spacy_sents else None
            penman_graph = add_lemmas(graph_str, snt_key='snt')
            penman_graph = annotate_penman(penman_graph, 
                                           tokens=tokens)
            aligned_graph = RBWAligner.from_penman_w_json(penman_graph)
            self.graphs[idx] = aligned_graph.get_penman_graph()
            sent = penman_graph.metadata['snt']
            
            nodes_alignment = {}
            
            # Nodes that can be aligned to the strings/tokens
            for i, (align, token) in \
                enumerate(zip(aligned_graph.alignments, tokens)):

                if align:
                    var = align.triple[0] if align.is_concept() else align.triple[1]
                    role = align.triple[1] if align.is_role() else None
                    concept = align.triple[2] if align.is_concept() else None

                    if var not in nodes_alignment:
                        nodes_alignment[var] = [
                            Node(i, var, concept, role, token, 
                                 token.idx, token.idx + len(token.text))
                        ]
                    else:
                        nodes_alignment[var].append(
                            Node(i, var, concept, role, token,
                                 token.idx, token.idx + len(token.text))
                        )

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
    
    
    def _get_concept(self, var):
        for v, role, concept in self.graphs[self.idx].instances():
            if v==var:
                return concept


    def tree(self, var, ori_var=None, depth=1):
        if depth > 1 and ori_var==var:
            return var, []
        edges = self.graphs[self.idx].edges(source=var)
        return var, [(e.role, self.tree(e.target, ori_var, depth+1)) for e in edges]
    

    def get_ancestor(self, tree, d=1, role=None):
        var, branch = tree
        ind = [(var, role)]
        for family in branch:
            role, child = family
            ind.extend(self.get_ancestor(child, d=d+1, role=role))
        return ind


    def get_actn_char(self):
        """
        Return:
            (list) candidates of action and character pairs
        """
        for idx, aligned in enumerate(self.align_var):
            self.idx = idx
            
            role = self._get_role_map(idx)
            cand = []
            for e in self.graphs[idx].edges():
                v1, v2 = e.source, e.target

                if e.role == ':ARG0':

                    if v1 in aligned:

                        if self._get_concept(v2) in ['and', 'person']:
                            ancestors = self.get_ancestor(self.tree(v2, v2))
                            vars = [v for v, r in ancestors \
                                     if r and (v in aligned) and (':ARG' not in r)]
                            for v in vars:
                                cand.append({'action': aligned[v1], 'character': aligned[v]})

                        elif v2 in aligned:
                            cand.append({'action': aligned[v1], 'character': aligned[v2]})
                        else:
                            # What could possibly goes here?
                            pass

                elif e.role == ':ARG0-of':
    
                    if v2 in aligned:
                        if v1 in aligned:
                            cand.append({'action': aligned[v2], 'character': aligned[v1]})
                    
            yield cand


# === Syntatic parsing happens here to analyse get the verb and subject of the sentence ===
class Sentence:
    def __init__(self, sentences, syntax_model=None, semantic_model=None):
        
        assert isinstance(sentences, str)

        if not syntax_model:
            syntax_model = load_models('syntax')

        self.doc = self.parse_syntax(syntax_model, sentences)
        self.structures = self.get_structures()

        if not semantic_model:
            semantic_model = load_models('semantic')

        self.amr = self.parse_semantic(semantic_model, [s.text for s in self.doc.sents])
        self.stories = list(self.amr.get_actn_char())


    def __len__(self):
        return len(self.doc)

    
    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        i = self.i
        if i == len(self.amr.graphs):
            raise StopIteration
        
        out = ''
        out += f"Sentence     : {self.amr.graphs[i].metadata['snt']}\n"
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
        return self.doc.sents


    def parse_syntax(self, syntax_model, sentences):
        return syntax_model(sentences)


    def parse_semantic(self, semantic_model, sentences):
        return GraphAligner(semantic_model(sentences), self.doc)
    

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
            'prt',
            'neg',
            'ccomp',
        ]
        for sent in self.doc.sents:
            
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
        for sent, verbs in zip(self.doc.sents, Verbs):
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
            'csubj',     # clausal subject
        ]
        subject = []
        for token in verb.children:
            if token.dep_ in dependents:
                for child in token.subtree:
                    if child.text in [',', '--', '—', '–']:
                         break
                    subject.append(child)

        subject = list(sorted(subject, key=lambda x: x.i))

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