
import pickle
from complex_graph_embedding import config
import numpy as np

from sklearn.model_selection import train_test_split


class KBManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_map = {}
        self.relation_map = {}
        self._build_vocabs()

    @staticmethod
    def extend_vocab(word, vocab):
        if word not in vocab:
            vocab[word] = len(vocab)

    def _build_vocabs(self):
        with open(self.data_dir, 'r') as f:
            data = f.read()
            lines = data.strip().split('\n')
            for line in lines:
                subj, rel, obj = line.split('|')
                self.extend_vocab(subj, self.entity_map)
                self.extend_vocab(obj, self.entity_map)
                self.extend_vocab(rel, self.relation_map)

        with open(config.ENTITY_VOCABULARY_PATH, 'wb') as f:
            pickle.dump(self.entity_map, f)

        with open(config.RELATION_VOCABULARY_PATH, 'wb') as f:
            pickle.dump(self.relation_map, f)

        # print("relation vocabulary size: {}".format(len(self.relation_map)))
        # print("entity vocabulary size: {}".format(len(self.entity_map)))

    def load_er_vocab(self):
        """"
        This function manages to build a dictionary outlined as: {(subject, relation): [object1, object2, ...]}
        each of subject, relation and objects are assigned an id in this class
        """
        result = {}
        with open(self.data_dir, 'r') as f:
            data = f.read()
            lines = data.strip().split('\n')
            for line in lines:
                subj, rel, obj = line.split('|')
                subj_idx = self.entity_map[subj]
                rel_idx = self.relation_map[rel]
                obj_idx = self.entity_map[obj]

                er_tuple = (subj_idx, rel_idx)
                if er_tuple in result:
                    result[er_tuple].append(obj_idx)
                else:
                    result[er_tuple] = [obj_idx]
        return result


class DataLoader:
    def __init__(self, kb_mgr):
        self.kb_mgr = kb_mgr
        raw_train_ds, raw_test_ds, raw_validation_ds = self.build_train_test_validation_dataset()
        self.target_dim = len(self.kb_mgr.entity_map)
        self.entity_dim = len(self.kb_mgr.entity_map)
        self.relation_dim = len(self.kb_mgr.relation_map)

        self.datasets = {
            'train': raw_train_ds,
            'test': raw_test_ds,
            'validation': raw_validation_ds
        }

    def build_train_test_validation_dataset(self):
        raw_ds = list(self.kb_mgr.load_er_vocab().items())
        train_raw_ds, test_raw_ds = train_test_split(raw_ds, test_size=config.TEST_SPLIT, shuffle=True)
        train_raw_ds, validation_raw_ds = train_test_split(train_raw_ds, test_size=config.VALIDATION_SPLIT)

        # print("train_size:{} test_size:{} validation_size:{}".format(
        #     len(train_raw_ds),
        #     len(test_raw_ds),
        #     len(validation_raw_ds)
        # ))

        return np.array(train_raw_ds), np.array(test_raw_ds), np.array(validation_raw_ds)

    def get_batch(self, dataset_type, batch_size):
        dataset = self.datasets[dataset_type]
        batch_idx = 0

        remainder = len(dataset) % batch_size
        if remainder != 0:
            dataset = dataset[:-remainder]

        while batch_idx < len(dataset):
            batch = dataset[batch_idx:batch_idx + batch_size, :]
            Xs = np.array(list(zip(*batch[:, 0])))
            Ys = np.zeros((batch_size, self.target_dim))
            for idx, targets in enumerate(batch[:, 1]):
                Ys[idx][targets] = 1
            yield Xs, Ys
            batch_idx += batch_size


kb_manager = KBManager(config.KB_PATH)
data_loader = DataLoader(kb_manager)
