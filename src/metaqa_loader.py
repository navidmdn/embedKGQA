from complex_graph_embedding import config as kb_config
from transformers import RobertaTokenizer, TFRobertaModel

import random
import re
import os
import numpy as np


class QA:
    def __init__(self, question, answers, question_entity):
        self.question = question
        self.answers = answers
        self.question_entity = question_entity


class MetaQADataLoader:
    """
    loads data from txt file and creates QA instances
    receives multiple paths (for combining n-hop datasets)
    """
    def __init__(self, kb_mgr, paths, kb_embedding_model, contextual_embedder='roberta'):
        self.kb_mgr = kb_mgr
        self.paths = paths
        self.raw_dataset = self.combine_datasets(paths)
        self.tokenizer = None
        self.contextual_embedding_model = None
        self.set_contextual_embedder(contextual_embedder)
        self.kb_embedding_model = self.load_kb_model_weights(kb_embedding_model)

    def load_kb_model_weights(self, model):
        model.load_weights(os.path.join(kb_config.SAVED_MODELS_PATH, 'complex'))
        return model

    def set_contextual_embedder(self, contextual_embedder):
        if contextual_embedder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.contextual_embedding_model = TFRobertaModel.from_pretrained('roberta-base')
        else:
            raise NotImplementedError()

    @staticmethod
    def read_dataset(path):
        dataset = []
        with open(path, 'r') as f:
            data_lines = f.read().strip().split('\n')
            for line in data_lines:
                question_raw, ans_raw = line.split('\t')
                q_entity = re.search(r'\[.+\]', question_raw).group().strip('[]')
                question = question_raw.replace(']', '').replace('[', '')
                answers = ans_raw.strip().split('|')
                dataset.append(QA(
                    question=question,
                    answers=answers,
                    question_entity=q_entity
                ))
        return dataset

    def combine_datasets(self, paths):
        combined_ds = []
        for path in paths:
            ds = self.read_dataset(path)
            combined_ds.extend(ds)
        random.shuffle(combined_ds)
        return combined_ds

    def get_batch(self, target_dim, batch_size):
        batch_idx = 0
        entity_map = self.kb_mgr.entity_map

        remainder = len(self.raw_dataset) % batch_size
        if remainder != 0:
            dataset = self.raw_dataset[:-remainder]

        while batch_idx < len(dataset):
            batch = dataset[batch_idx:batch_idx + batch_size]

            qs = list(map(lambda qa: qa.question, batch))
            tokens = self.tokenizer(
                qs,
                return_tensors="tf",
                padding=True
            )

            q_embedds = self.contextual_embedding_model(tokens)
            q_embedds = q_embedds[0].numpy()[:, 0, :]

            entities = map(lambda qa: qa.question_entity, batch)
            entity_ids = list(map(lambda e: entity_map[e], entities))
            entity_embedds = self.kb_embedding_model.entity_encoder(np.array(entity_ids)).numpy()

            labels_list = map(lambda qa: qa.answers, batch)

            Xs = [q_embedds, entity_embedds]
            Ys = np.zeros((batch_size, target_dim))
            for idx, labels in enumerate(labels_list):
                targets = list(map(lambda ans: entity_map[ans], labels))
                Ys[idx][targets] = 1

            yield Xs, Ys
            batch_idx += batch_size
