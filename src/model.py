import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def create_emb_layer(weights_matrix, trainable=False):
    vocab_size, emb_dim = weights_matrix.size()
    emb_layer = nn.Embedding(vocab_size, emb_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if trainable == False:
        emb_layer.weight.requires_grad = False
    return emb_layer, vocab_size, emb_dim

class two_stage_RNN(nn.Module):
    def __init__(self, rnn_1, hidden_dim1, bi, rnn_2, hidden_dim2, batch_size, cuda_on, weights_matrix, num_tasks, num_classes):
        
        super(two_stage_RNN, self).__init__()
        
        self.num_tasks = num_tasks
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.bi = bi	

        self.embedding, vocab_size, emb_dim = create_emb_layer(weights_matrix, trainable=False)
        
        # module for steps in the fisrt stage
#         self.hidden_stage1, self.hidden_stage2 = self.init_hidden(batch_size, cuda_on)
        rnn_common = rnn_1(emb_dim, hidden_dim1, num_layers=1, 
                           batch_first=True, bidirectional=bi)
        self.rnn_each_step = nn.ModuleList([])
        for i in range(6):
            self.rnn_each_step.append(rnn_common)
        
        # module for the second stage
        if self.bi:
            self.steps_rnn = rnn_2(hidden_dim1*2, hidden_dim2, num_layers=1, batch_first=False)
        else:
            self.steps_rnn = rnn_2(hidden_dim1, hidden_dim2, num_layers=1, batch_first=False)
        
        # module for interaction
        self.classifiers_mlt = nn.ModuleList([])
        for i in range(self.num_tasks):
            self.classifiers_mlt.append(nn.Linear(hidden_dim2, num_classes))
        #self.linear = nn.Linear(hidden_dim2, num_classes)

    def forward(self, steps, lengths):
        # first stage
        output_each_step = []
        for i in range(6):
            rnn_input = steps[i]
            emb = self.embedding(rnn_input) # embedding
            output, _ = self.rnn_each_step[i](emb) #, self.hidden_stage1[str(i)]
            if self.bi:
                output_size = output.size()
                output = output.view(output_size[0], output_size[1], 2, self.hidden_dim1)
            if self.bi:
                output_each_step.append(torch.cat((output[:,-1,0,:],output[:,0,1,:]),1))
            else:
                output_each_step.append(output[:,-1,:])
        
        #second stage
        output1 = torch.stack(output_each_step, 0)
        output, _ = self.steps_rnn(output1) #, self.hidden_stage2
        output = output[-1,:,:]
        logits = {}
        for i in range(self.num_tasks):
            logits[i] = self.classifiers_mlt[i](output)
        return logits

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    logits_all_dict = defaultdict(list)
    labels_all_dict = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    for steps_batch, lengths_batch, labels_batch in loader:
        for step_id in range(6):
            lengths_batch[step_id] = lengths_batch[step_id].to(device)
            steps_batch[step_id] = steps_batch[step_id].to(device)
        logits = model(steps_batch, lengths_batch)
        for i in labels_batch.keys():
            logits_all_dict[i].extend(list(logits[i].cpu().detach().numpy()))
            labels_all_dict[i].extend(list(labels_batch[i].numpy()))
    auc = {}
    acc = {}
    for i in labels_all_dict.keys():
        logits_all_dict[i] = np.array(logits_all_dict[i])
        labels_all_dict[i] = np.array(labels_all_dict[i])
        auc[i] = roc_auc_score(labels_all_dict[i], logits_all_dict[i])
        predicts = (logits_all_dict[i] > 0.5).astype(int)
        acc[i] = np.mean(predicts==labels_all_dict[i])
    return auc, acc
