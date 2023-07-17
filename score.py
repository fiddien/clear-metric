from parse import Sentence

class ClearMetric:
    def __init__(self, sentences):
        assert isinstance(sentences, Sentence)
        self.sents = sentences

    def score_character(self, structure, story):
        if structure and story:
            for scene in story:
                for x in scene['character']:
                    for idx in self.subjects_start:
                        if x.i==idx:
                            return 0
        return 1


    def score_action(self, structure, story):
        if structure and story:
            for scene in story:
                for x in scene['action']:
                    for idx in self.verbs_start:
                        if x.i==idx:
                            return 0
        return 1


    def score_long_abstract_subject(self, structure):
        if structure:
            s = structure[0]
            length = s['subject'][-1].i - s['subject'][0].i + 1
            return int(length > 8)
        return 0


    def score_long_intro_phrases_clauses(self, structure):
        if structure:
            s = structure[0]
            if len(structure) > 1 and 'said' in [t.text for t in s['verb']]:
                s = structure[1]
            word = s['subject'][0]
            # Raw calculation
            start = word.i - word.sent.start
            # Count the number of punctiations
            num_puncts = len([t for t in word.sent[:start] \
                            if t.pos_=='PUNCT'])
            return int((start - num_puncts) > 8)
        return 0


    def score_subj_verb_connection(self, structure):
        if structure:
            s = structure[0]
            start = s['subject'][-1].i
            end = s['verb'][0].i
            # Raw calculation
            interruption = end - start - 1
            # Count the number of entities
            ents = [ent for ent in s['verb'][0].sent.ents \
                    if ent.start<end and ent.end>start]
            len_ents_span = len([t for ent in ents for t in ent])
            # COunt the number of punctiations
            num_puncts = len([t for t in s['verb'][0].sent[start+1:end] \
                            if t.pos_=='PUNCT'])
            return int((interruption - len_ents_span + len(ents) - num_puncts) > 3)
        return 0


    def score_sents(self):
        scores = []
        for structure, story in zip(self.sents.structures, self.sents.stories):
            
            if structure:
                subjects = [su for st in structure for su in st['subject']]
                self.subjects_start = [su.i-su.sent.start for su in subjects]
                verbs = [ve for st in structure for ve in st['verb']]
                self.verbs_start = [ve.i-ve.sent.start for ve in verbs]
            
            scores.append([
                self.score_character(structure, story),
                self.score_action(structure, story),
                self.score_long_abstract_subject(structure),
                self.score_long_intro_phrases_clauses(structure),
                self.score_subj_verb_connection(structure),
            ])
        return scores