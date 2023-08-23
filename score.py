from parse import Sentence

class ClearMetric:
    def __init__(self, sentences):
        assert isinstance(sentences, Sentence)
        self.sents = sentences

    def score_character(self, structure, story):
        if structure and story:
                # if self.characters_idx & self.subjects_idx:
                #     return 0
                for ch_idx in self.characters_idx:
                    for su_idx in self.subjects_idx:
                        if ch_idx.issubset(su_idx):
                            return 0
        elif not structure or not story:
            return 0
        return 1


    def score_action(self, structure, story):
        if structure and story:
                # if self.actions_idx & self.verbs_idx:
                #     return 0
                for ac_idx in self.actions_idx:
                    for ve_idx in self.verbs_idx:
                        if ac_idx.issubset(ve_idx):
                            return 0
        elif not structure or not story:
            return 0
        return 1


    def score_long_abstract_subject(self, structure):
        if structure:
            s = structure[0]
            word = s['subject'][0]
            # Raw calculation first
            length = s['subject'][-1].i - s['subject'][0].i + 1
            # Count the number of entities
            subj_start = word.i - word.sent.start
            ents = [ent for ent in s['verb'][0].sent.ents \
                    if ent.end<subj_start]
            len_ents_span = len([t for ent in ents for t in ent])
            # Count the number of punctuations
            num_puncts = len([t for t in word.sent[:subj_start] \
                            if t.pos_=='PUNCT'])
            # Discount the puctuations
            length = length - num_puncts - len_ents_span + len(ents)
            return int(length > 8)
        return 0


    def score_long_intro_phrases_clauses(self, structure):
        if structure:
            is_long_intro = False
            for s in structure:
                if len(structure) > 1 and 'said' in [t.text for t in s['verb']]:
                    s = structure[1]
                word = s['subject'][0]
                # Raw calculation
                subj_start = word.i - word.sent.start
                # Count the number of entities
                ents = [ent for ent in s['verb'][0].sent.ents \
                        if ent.end<subj_start]
                len_ents_span = len([t for ent in ents for t in ent])
                # Count the number of punctuations
                num_puncts = len([t for t in word.sent[:subj_start] \
                                  if t.pos_=='PUNCT'])
                is_long_intro = (subj_start - num_puncts - len_ents_span + len(ents)) > 8
                # Return 1 as long as there's a subject
                # that appears early
            return int(is_long_intro)
        return 0


    def score_subj_verb_connection(self, structure):
        if structure:
            is_disconnected = False
            for s in structure:
                start = s['subject'][-1].i
                end = s['verb'][0].i
                # Raw calculation
                interruption = end - start - 1
                # Count the number of entities
                ents = [ent for ent in s['verb'][0].sent.ents \
                        if ent.start<end and ent.end>start]
                len_ents_span = len([t for ent in ents for t in ent])
                # Count the number of punctiations
                num_puncts = len([t for t in s['verb'][0].sent[start+1:end] \
                                  if t.pos_=='PUNCT'])
                is_disconnected = (interruption - len_ents_span + len(ents) - num_puncts) > 4
                # Return 1 as long as there's one example
                # whom subject-verb connection is disrupted
            return int(is_disconnected)
        return 0


    def score_sents(self):
        scores = []
        for structure, story in zip(self.sents.structures, self.sents.stories):
            
            if structure:
                # subjects = [su for st in structure for su in st['subject']]
                # self.subjects_idx = set([su.idx for su in subjects])
                self.subjects_idx = [{x.i for x in st['subject']} for st in structure]
                # verbs = [ve for st in structure for ve in st['verb']]
                # self.verbs_idx = set([ve.idx for ve in verbs])
                self.verbs_idx = [{x.i for x in st['verb']} for st in structure]
            
            if story:
                # characters = [ch for st in story for ch in st['character']]
                # self.characters_idx = set([ch.start for ch in characters])
                self.characters_idx = [{x.i for x in st['character']} for st in story]
                # actions = [ac for st in story for ac in st['action']]
                # self.actions_idx = set([ac.start for ac in actions])
                self.actions_idx = [{x.i for x in st['action']} for st in story]

            scores.append([
                self.score_character(structure, story),
                self.score_action(structure, story),
                self.score_long_abstract_subject(structure),
                self.score_long_intro_phrases_clauses(structure),
                self.score_subj_verb_connection(structure),
            ])
        return scores