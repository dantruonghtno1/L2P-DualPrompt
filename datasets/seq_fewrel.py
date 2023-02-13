import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple 
from transformers import AutoModelForSequenceClassification
import pickle 

class FewRel:
    def __init__(
        self,
        train = True, 

    ):  
        with open('/home/truongpdd/L2P-DualPrompt/train_test_flatten.pkl', 'rb') as fr:
            train_dataset, _, test_dataset,self.id2rel, self.rel2id  = pickle.load(fr)
        if train:
            self.data = train_dataset 
            ids_list = [self.data[i]['tokens'] for i in range(len(self.data))]
            targets_list = [self.data[i]['tokens'] for i in range(len(self.data))]
        else:
            self.data = test_dataset 
        ids_list = [self.data[i]['tokens'] for i in range(len(self.data))]
        targets_list = [self.data[i]['relation'] for i in range(len(self.data))]
        import numpy as np
        self.data = ids_list
        self.targets = targets_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index : int):
        """
            get the requested element from dataset
            input: 
                + index : index of the element to be return:
            return:
                + tokens: list 
                + relation: int
        """

        tokens, relation = self.data[index], self.targets[index]
        return tokens, relation
    def collate_fn(self, data):
        # print(data)
        tokens = torch.tensor(
            [item[0] for item in data]
        )
        labels = torch.tensor(
            [item[1] for item in data]
        )

        return (
            tokens,
            labels
        )
# class SequentialFewRel(ContinualDataset):
#     NAME = 'seq-fewrel'
#     SETTING = 'class-il'
#     N_CLASSES_PER_TASK = 8
#     N_TASKS = 10

#     def get_data_loaders(self):
#         train_dataset = FewRel(train = True)
#         test_dataset = FewRel(train = False)

#         train, test = store_masked_loaders(
#             train_dataset, 
#             test_dataset, 
#             self,
#         )

class SequentialFewREl(ContinualDataset):
    NAME = 'seq-fewrel80'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 8
    N_TASKS = 10 

    def get_data_loaders(self):
        train_dataset = FewRel(train = True)
        test_dataset = FewRel(train = False)
        train, test = store_masked_loaders(
            train_dataset,
            test_dataset, 
            self
        )
        return train, test 
    
    @staticmethod 
    def get_backbone() -> nn.Module:
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        model.classifier = torch.nn.Linear(
            768, 
            80
        )
        return model.cuda()
        
    @staticmethod
    def get_loss() -> nn.functional:
        return nn.CrossEntropyLoss()
    