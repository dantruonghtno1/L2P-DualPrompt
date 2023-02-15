import torch 
import time 
import numpy as np 
from utils.status import progress_bar, create_stash
from utils.tb_logger import * 
from utils.loggers import * 
from utils.loggers import CsvLogger 
from argparse import Namespace 
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset 
import sys
def mask_classes(
    outputs: torch.Tensor, 
    dataset: ContinualDataset, 
    k:int
)->None:
    """
    Give the output tensor, dataset at hand, and the current task 
    masks the former by setting the responses for the other tasks at -inf
    it is use to obtain the result for the task-il setting
    + output: the output of tensor 
    + dataset: the continual dataset 
    + k : the task index     
    """

    outputs[:, 0:k*dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k+1)*dataset.N_CLASSES_PER_TASK:] = -float('inf')

def evaluate_fewrel(
    model : ContinualModel, 
    dataset: ContinualDataset, 
    last = False
)-> Tuple[list, list]:
    """
    evaluate accuracy of model for each past task 
    + model: model to be evaluate 
    + dataset : continual dataset at hand
    + return: a tuple lists, containing the class-il and task-il accuracy for each task
    """
    
    status = model.net.training 
    model.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue 
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0 
        for data in test_loader:
            inputs, labels, text = data 

            input_ids = inputs.cuda()

            labels = torch.tensor(labels).cuda()
            # sửa phần đem input vào model forward
            outputs = model.forward_model(
                input_ids = input_ids, text = text
            )
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            mask_classes(outputs, dataset, k)
            _, pred = torch.max(outputs.data, 1)
            correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    model.net.train(status)
    return accs, accs_mask_classes

def train_fewrel(
    model: ContinualModel, 
    dataset: ContinualDataset, 
    args: Namespace
)-> None:
    """
    training process, includiong evaluations and loggers
    + model : the module to be trained 
    + dataset : the continual dataset at hand
    + args : the argments of the current execution 
    """

    model.net.to('cuda')
    results, result_mask_classes = [], [] 

    model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        csv_logger = CsvLogger(
            dataset.SETTING, dataset.NAME,model.NAME
        )
    if args.tensorboard:
        tb_logger = TensorboardLogger(
            args, 
            dataset.SETTING, 
            model_stash
        )
        model_stash['tensorboard_name'] = tb_logger.get_name()
    print(file = sys.stderr)
    start_time = time.time()
    
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        
        model.init_opt(args)
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                inputs, labels, text = data 
                input_ids = inputs.cuda()
                # loss = model.observe(inputs, labels,dataset,t)

                loss = model.observe(
                    input_ids = input_ids, 
                    labels = labels, 
                    dataset = dataset,
                    text = text,
                    t = t
                )

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['task_idx'] = t + 1 
            model_stash['epoch_idx'] = epoch + 1 
            model_stash['batch_idx'] = 0 

        model_stash['task_idx'] = t + 1 
        model_stash['epoch_idx'] = 0 

        accs = evaluate_fewrel(model, dataset) 
        results.append(accs[0])
        result_mask_classes.append(accs[1])
        print()
        print(accs[0])
        print(accs[1])

        mean_acc = np.mean(accs, axis = 1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t) 
        
    running_time = time.time() - start_time
    if args.csv_log:
        csv_logger.add_bwt(results, result_mask_classes)
        csv_logger.app_running_time(running_time)
        csv_logger.add_forgetting(results, result_mask_classes)
    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))










    