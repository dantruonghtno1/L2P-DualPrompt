# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_fewrel import SequentialFewREl
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

# string : instance
NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialFewREl.NAME : SequentialFewREl

}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
