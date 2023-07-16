import amrlib
from amrlib.graph_processing.annotator import add_lemmas, annotate_penman, load_spacy
from amrlib.alignments.rbw_aligner import RBWAligner
import spacy, benepar, nltk

spacy_model_name = 'en_core_web_md'
benepar.download('benepar_en3')

nlp = spacy.load(spacy_model_name)
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
load_spacy(spacy_model_name)

stog = amrlib.load_stog_model()

# === Semantic parsing happens here to determine the character and action of the sentence ===

class Node:
    def __init__(self, i=None, var=None, concept=None, role=None, text=''):
        self.i = i
        self.role = role
        self.text = text
        self.var = var
        self.concept = concept

    def __repr__(self):
        return self.text


class GraphAligner:
    def __init__(self, sents, graphs):
        if isinstance(sents, spacy.tokens.doc.Doc):
            self.align, self.graphs = self.align_nodes(sents.sents, graphs, True)
        else:
            assert isinstance(graphs, list)
            assert isinstance(graphs[0], str)
            self.align, self.graphs = self.align_nodes(sents, graphs)


    def align_nodes(self, sents, graphs, add_tokens=None):
        """
        Align the nodes of each AMR graphs to a token in the respective sentence string.
        Argument:
            graph_str: (str) AMR graph
        Return:
            a (dict) that contain the Penman graph and the collection nodes in the graph
            accompanied by information about the tokens, id of the tokens (token number)
            and each corresponding variable name
        """
        # assert isinstance(graphs, str)

        alignments, penman_graphs = [], []
        for sent, graph in zip(sents, graphs):
            tokens = [t for t in sent] if add_tokens else None
            penman_graph = add_lemmas(graph, snt_key='snt')
            penman_graph = annotate_penman(penman_graph, 
                                           tokens=tokens)
            aligned_graph = RBWAligner.from_penman_w_json(penman_graph)
            nodes_alignment = {}

            # Nodes that can be aligned to the strings/tokens
            for i, (align, token) in enumerate(zip(aligned_graph.alignments,
                                                   aligned_graph.tokens)):
                if align:
                    concept, var = align.triple[-1], align.triple[0]
                    if var not in nodes_alignment:
                        nodes_alignment[var] = Node(
                            i, var, concept, None, token
                        )
                    else:
                        nodes_alignment[var] = [
                            nodes_alignment[var],
                            Node(i, var, concept, None, token)
                        ]

            penman_graphs.append(aligned_graph.get_penman_graph())
            alignments.append(nodes_alignment)


        return alignments, penman_graphs


    def _get_concept_map(self, graph):
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


    def _get_role_map(self, graph):
        """
        Create a dictionary with tuples of two neighboring AMR graph nodes as the keys
        and their connection/role as the values.
        Argument:
            graph: (penman.graph)
        Return:
            dictionary: {(source node, target node): role}
        """
        return {(e.source, e.target): e.role for e in graph.edges()}


    def _get_children(self, graph, var):
        """
        Return the first child of a node.
        Arguments:
            graph: a Penman graph
            var: (str) the variable name of the node whose children wants to be returned
        Return:
            children: (list) of (child_target_node, role)
        """
        children = [(t, r) for (s, r, t) in graph.triples \
                    if s==var and r!=':instance']
        for child in children:
            yield child
        # return children


    def get_actn_char(self):
        """
        Argument:
            graph: (penman.graph)
        Return:
            (list) candidates of action and character pairs
        """
        for aligned, graph in zip(self.align, self.graphs):
            concept_map = self._get_concept_map(graph)
            role_map = self._get_role_map(graph)
            candidates = []

            for (source, target), role in role_map.items():
                pair = None

                if source in aligned:
                    source = aligned[source]

                    if role in [':ARG0']:

                        # When the node is an intermediete node
                        rels = ['person', 'country', 'and']
                        if concept_map[target] in rels:
                            depth = 0
                            children = self._get_children(graph, target)
                            parents = self._get_children(graph, target)
                            parent = next(parents)
                            parent_role = parent[1]
                            child = next(children, None)

                            while child:
                                var, role = child
                                # print('child', child, '| parent', parent)
                                
                                if (role in [':name', ':ARG0-of', ':ARG1-of']) \
                                or (':op' in role) \
                                or (role==':ARG2' and ':ARG0-of' in parent_role):

                                    # print('  IN')
                                    if var in aligned:

                                        # print('  ALIGNED', end='')
                                        if isinstance(aligned[var], list):
                                            target = aligned[var]
                                        else:
                                            target = [aligned[var]]
                                        cand = {
                                            'action': [source],
                                            'character': target
                                        }
                                        if not candidates:
                                            candidates.append(cand)
                                        elif cand!=candidates[-1]:
                                            candidates.append(cand)
                                        # print('     ', var, role)
                                    else:
                                        # print('  ELSE', var, role)
                                        parent_role = role
                                        children = self._get_children(graph, var)
                                
                                # print('   ', candidates)
                                child = next(children, None)
                                if child is None and parent:
                                    # print('    (((((CHANGE)))))')
                                    parent = next(parents, None)
                                    if parent:
                                        var, parent_role = parent
                                        children = self._get_children(graph, var)
                                        child = next(children, None)
                                                    
                        elif target in aligned:
                            target = aligned[target]
                            candidates.append(
                                {'action': [source], 'character': [target]}
                            )
                    elif role==':manner' and target in aligned:
                        candidates[-1]['action'].append(aligned[target])

            yield candidates


# === Syntatic parsing happens here to analyse get the verb and subject of the sentence ===
class Sentence:
    def __init__(self, sentences, syntax_model=nlp,
                 semantic_model=stog.parse_sents):

        self.docs = self.parse_syntax(syntax_model, sentences)
        assert len(list(self.docs.sents)) == len(sentences)
        self.structures = self.get_structures()

        self.amrs = self.parse_semantic(semantic_model, sentences)
        self.stories = list(self.amrs.get_actn_char())

        self.analyse = lambda: self.analyse()


    def __len__(self):
        return len(self.docs)


    def get_sentences(self):
        return self.docs.sents


    def parse_syntax(self, syntax_model, sentences):        
        if isinstance(sentences, list) and isinstance(sentences[0], str):
            sentences = " ".join(sentences)
        return syntax_model(sentences)


    def parse_semantic(self, semantic_model, sentences):
        return GraphAligner(self.docs, semantic_model(sentences))


    def annotation(self):
        return self.structures + self.stories
        

    def get_structures(self):

        def is_main(verb):
            for token in verb:
                if token.dep_=='ROOT':
                    return True
            return False
        
        Verbs = [v for v in self.get_verbs()]
        Subjects = [s for s in self.get_subjects(Verbs)]

        structures = []
        for verbs, subjects in zip(Verbs, Subjects):
            vs = []
            if verbs:
                if len(verbs)<len(subjects):
                    verbs = verbs * len(subjects)
                elif len(verbs)>len(subjects):
                    subjects = subjects * len(verbs)

                for v, s in zip(verbs, subjects):
                    if isinstance(s[0], list) and len(s)>1:
                        for s_ in s:
                            
                            vs.append({'verb': v, 
                                       'subject': s_, 
                                       'main': is_main(v)})
                    elif isinstance(v[0], list) and len(v)>1:
                        for v_ in v:
                            vs.append({'verb': v_, 
                                       'subject': s, 
                                       'main': is_main(v_)})
                    else:
                        vs.append({'verb': v, 
                                   'subject': s, 
                                   'main': is_main(v)})
                        
            vs = sorted(vs, key=lambda x: x['main'], reverse=True)
            structures.append(vs)
        return structures


    def get_verbs(self):
        """ Get the verb phrase which includes the root (main verb). """
        dependents = [
            'aux',
            'auxpass',
            'advmod',
            'prt',
            'neg',
            'ccomp',
        ]
        for sent in list(self.docs.sents):
            if sent.root.pos_ not in ['VERB', 'AUX']:
                yield []

            verbs = [sent.root]
            for child in sent.root.children:
                if child.dep_ in dependents and child.pos_!='NOUN':
                    verbs.append(child)
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


    def _get_noun(self, verb):
        dependents = [
            'nsubj',     # active sentence
            'nsubjpass', # passive sentence
            'expl',      # captures an existential *there* or *it* in extraposition constructions
            'npadvmod',
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


    def _group_consecutive_tokens(self, seq):
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
