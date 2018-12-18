#build data loader for instruction data

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial
from collections import Counter, defaultdict

class IntructionDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, tags_list, max_sent_len):
        """
        
        @param data_list: list of recipie tokens 
        @param target_list: list of single tag, i.e. 'tag_cuisine_american'

        """
        self.data_list = data_list
        self.tags_list = tags_list
        self.max_sent_len = max_sent_len
        assert (len(self.data_list) == len(self.tags_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call recipie[i]
        """
        recipie = self.data_list[key]
        step1_idx = recipie[0][:self.max_sent_len[0]]
        step2_idx = recipie[1][:self.max_sent_len[1]]
        step3_idx = recipie[2][:self.max_sent_len[2]]
        step4_idx = recipie[3][:self.max_sent_len[3]]       
        step5_idx = recipie[4][:self.max_sent_len[4]]
        step6_idx = recipie[5][:self.max_sent_len[5]]
        label = self.tags_list[key]
        return [[step1_idx, step2_idx, step3_idx, step4_idx, step5_idx, step6_idx], 
                [len(step1_idx),len(step2_idx), len(step3_idx),len(step4_idx), len(step5_idx),len(step6_idx)], 
                label]

def collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    steps_dict = defaultdict(list)
    label_dict = defaultdict(list)
    length_dict = defaultdict(list)
    max_sent_len = []
    for datum in batch:
        for idx, task_label in enumerate(datum[-1]):
            label_dict[idx].append(task_label)
        for i in range(6):
            length_dict[i].append(datum[1][i])
    
    # padding
    for i in range(6):
        max_sent_len.append(max(length_dict[i]))
    
    for datum in batch:
        for i, step in enumerate(datum[0]):
            padded_vec = np.pad(np.array(step), 
                                pad_width=((0, max_sent_len[i]-datum[1][i])), 
                                mode="constant", constant_values=0)
            steps_dict[i].append(padded_vec)
    
    for key in length_dict.keys():
        length_dict[key] = torch.LongTensor(length_dict[key])
        steps_dict[key] = torch.from_numpy(np.array(steps_dict[key]).astype(np.int))
    for key in label_dict.keys():
        label_dict[key] = torch.LongTensor(label_dict[key])
        
    return [steps_dict, length_dict, label_dict]

# Build train, valid and test dataloaders
def create_dataset_obj(train,val,test,train_targets,val_targets,test_targets,
                       BATCH_SIZE,max_sent_len,collate_func):
    collate_func=partial(collate_func)
    train_dataset = IntructionDataset(train, train_targets, max_sent_len)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=collate_func,
                                               shuffle=True)

    val_dataset = IntructionDataset(val, val_targets, max_sent_len)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=collate_func,
                                               shuffle=False)

    test_dataset = IntructionDataset(test, test_targets, max_sent_len)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=collate_func,
                                               shuffle=False)
    return train_loader, val_loader, test_loader

