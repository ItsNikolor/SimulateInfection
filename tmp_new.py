import pandas as pd
import numpy as np
import collections


def get_skill_to_skill_group_mapping():
    pass


class UseEmbedding:
    def vectors(self, _):
        return np.array(0)


def normalize(_):
    return np.array(0)


class Faiss:
    def IndexFlatIP(self, _):
        pass


faiss = Faiss()


def intent_to_skill_group(_, __):
    pass


class KnnModel:
    def __init__(self):
        self.skill_to_skills_group = get_skill_to_skill_group_mapping()

        self.k_neighbors = 3
        self.power = 8
        self.th_scope = 0.6
        self.th_final = 0.6

        self.embedding = UseEmbedding()

    def build_index(self, path):
        df = pd.read_csv(path)

        def index_for_intent(phrases):
            index_embeddings = np.stack(self.embedding.vectors(phrases)).astype(np.float32)
            index_embeddings = normalize(index_embeddings)

            index = faiss.IndexFlatIP(index_embeddings.shape[-1])
            index.add(index_embeddings)

            return index

        df = df.append({'phrase': '', 'subintent': 'unknown'}, ignore_index=True)
        df_index = df.groupby('subintent').phrase.apply(index_for_intent)

        self.intents = df_index.index.values
        self.index = df_index.values
        self.unk_ind = list(self.intents).index('unknown')

        self.group_mask = collections.defaultdict(lambda: np.zeros(len(self.intents), dtype=bool))
        for ind, intent in enumerate(self.intents):
            self.group_mask[intent_to_skill_group(intent, self.skill_to_skills_group)][ind] = True
        self.group_mask = collections.defaultdict(lambda: np.ones(len(self.intents), dtype=bool), self.group_mask)

    def predict(self, phrases, groups=None, topk=1):
        embeddings = np.stack(self.embedding.vectors(phrases)).astype(np.float32)
        embeddings = normalize(embeddings)

        all_distances = np.zeros((len(embeddings), self.k_neighbors, len(self.intents)))
        for i in range(len(self.intents)):
            if i == self.unk_ind:
                continue
            all_distances[:, :, i] = self.index[i].search(embeddings, self.k_neighbors)[0]



        if groups is not None:
            for ind, group in enumerate(groups):
                all_distances[ind] *= self.group_mask[group]

        prediction = self._choice_mean(all_distances, topk, self.unk_ind)

        return [[self.intents[ind] for ind in top_k_prediction] for top_k_prediction in prediction]

    def _choice_mean(self, dist, topk, unk_ind=-1):
        mask_scope = dist >= self.th_scope
        dist *= mask_scope

        mask_valid = dist != 0
        intents_count = mask_valid.sum(axis=1)
        intents_count += intents_count == 0

        score = dist ** self.power
        score = score.sum(axis=1) / intents_count

        best = np.argpartition(score, -min(score.shape[1], topk), axis=-1)[:, -topk:]
        best = np.take_along_axis(best,
                                  np.argsort(np.take_along_axis(score, best, -1))[:, ::-1], -1)

        score = np.take_along_axis(score, best, -1)

        final_mask = score < (self.th_final ** self.power)
        out_classes = best * (~final_mask) + final_mask * unk_ind

        return out_classes
