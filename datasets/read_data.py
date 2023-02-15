import pickle
import random
import json, os
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    return tokenizer

class Load_all_data(object):
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.id2rel, self.rel2id = self._read_relations(file = '/home/truongpdd/L2P-DualPrompt/datasets/id2rel.json')
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(file = '/home/truongpdd/L2P-DualPrompt/CRL/datasets/data_with_marker.json')
        """
            args.num_of_relation = 80
            args.num_of_train = 420
            args.num_of_val = 140
            args.num_of_test = 140
        """


    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def _read_data(self, file):
        if os.path.isfile('./data.pkl'):
            with open('./data.pkl', 'rb') as f:
                datas = pickle.load(f)
            train_dataset, val_dataset, test_dataset = datas
            return train_dataset, val_dataset, test_dataset
        else:
            data = json.load(open(file, 'r', encoding='utf-8'))
            # """
            #     args.num_of_relation = 80
            #     args.num_of_train = 420
            #     args.num_of_val = 140
            #     args.num_of_test = 140
            # """
            train_dataset = [[] for i in range(80)]
            val_dataset = [[] for i in range(80)]
            test_dataset = [[] for i in range(80)]
            for relation in data.keys():
                rel_samples = data[relation]

                random.shuffle(rel_samples)
                count = 0
                count1 = 0
                for i, sample in tqdm(enumerate(rel_samples)):
                    tokenized_sample = {}
                    tokenized_sample['relation'] = self.rel2id[sample['relation']]
                    tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                    padding='max_length',
                                                                    truncation=True,
                                                                    max_length=256)
                    tokenized_sample['text'] = ' '.join(sample['tokens'])

                    if i < 420:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < 420 + 140:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
            print('--------------------------start save---------------------------')
            with open('./data.pkl', 'wb') as f:
                pickle.dump((train_dataset, val_dataset, test_dataset, self.id2rel, self.rel2id), f)
            print('----------------------------end save------------------------------')
            return train_dataset, val_dataset, test_dataset


