# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from collections import defaultdict

from embedding import load_emb_vectors, build_emb_weight
from loadData import create_dataset_obj, collate_func
from model import create_emb_layer, two_stage_RNN, test_model
from preprocess import tokenize_dataset, all_tokens_list, build_vocab, token2index_dataset 

#build model
def train_model(params, emb_weight, train_loader, val_loader, test_loader, loss_weights):
    rnn1_type = params['rnn1_type'] 
    rnn_1 = rnn_types[rnn1_type]
    rnn2_type = params['rnn2_type']
    rnn_2 = rnn_types[rnn2_type]
    bi = params['bi']
    tags_predicted = params['tags_predicted']
    num_tasks = len(tags_predicted)

    hidden_dim1 = params['hidden_dim1']
    hidden_dim2 = params['hidden_dim2']
    
    multi_task_train = params['multi_task_train'] 
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    cuda_on = params['cuda_on']

    weights_matrix = torch.from_numpy(emb_weight)
    model = two_stage_RNN(rnn_1, hidden_dim1, bi, rnn_2, hidden_dim2, batch_size, 
                          cuda_on, weights_matrix, num_tasks, num_classes)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('The number of train parameters', sum([np.prod(p.size()) for p in model_parameters]))
    model = model.to(device)

    #parameter for training
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs'] # number epoch to train

    # Criterion and Optimizer
    #pos_weight=torch.Tensor([40,]).cuda()
    criterion = {}
    if loss_weights is None:
        for i in range(num_tasks):
            criterion[i] = nn.BCEWithLogitsLoss() #torch.nn.BCELoss(); torch.nn.CrossEntropyLoss()
    else:
        for i in range(num_tasks):
            criterion[i] = nn.BCEWithLogitsLoss(pos_weight=loss_weights[i])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    # train_AUC_list = defaultdict(float)
    # train_ACC_list = defaultdict(float)
    val_AUC_dict = defaultdict(list)
    val_ACC_dict= defaultdict(list)
    max_val_auc = defaultdict(float)
    step_max_descent = params['step_max_descent']

    val_auc_mean = defaultdict(float)
    val_acc_mean = defaultdict(float)

    for epoch in range(num_epochs):
        for i, (steps_batch, lengths_batch, labels_batch) in enumerate(train_loader):
            for step_id in range(6):
                lengths_batch[step_id] = lengths_batch[step_id].to(device)
                steps_batch[step_id] = steps_batch[step_id].to(device)
            model.train()
            optimizer.zero_grad()
            logits = model(steps_batch, lengths_batch)
            if multi_task_train == 'mean_loss':
                loss_list = []
                for task_id in range(num_tasks):
                    loss_list.append(criterion[task_id](logits[task_id], 
                                          labels_batch[task_id].view(-1,1).float().to(device)))
                loss= torch.mean(torch.stack(loss_list))
            elif multi_task_train == 'random_selection':
                task_id = np.random.randint(0, num_tasks)
                loss = criterion[task_id](logits[task_id], labels_batch[task_id].view(-1,1).float().to(device))
            else:
                print('multi-task-train-method Error')
            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if i % 10 == 0:
                val_auc, val_acc = test_model(val_loader, model)
                for key in val_auc.keys():
                    val_ACC_dict[key].append(val_acc[key])
                    val_AUC_dict[key].append(val_auc[key])

                print('{}/{}, Step:{}/{}, TrainLoss:{:.6f}, ValAUC:{} ValAcc:{}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), loss, val_auc, val_acc))
                
                # train_auc, train_acc = test_model(train_loader, model)
                # train_AUC_list.append(train_auc)
                # train_ACC_list.append(train_acc)
                
                # early stop
                flag_increase = False
                for key in val_auc.keys():
                    if max_val_auc[key] < val_auc[key]:
                        max_val_auc[key] = val_auc[key]
                        flag_increase = True
                if flag_increase == True:
                    step_num_descent = 0
                else:
                    step_num_descent += 1

                if step_max_descent == step_num_descent:
                    print('early stop!')
                    break
        val_auc, val_acc = test_model(val_loader, model)
        train_auc, train_acc = test_model(train_loader, model)
        print('Epoch: [{}/{}], trainAUC: {}, trainAcc: {}'.format(epoch+1, num_epochs, train_auc.values(), train_acc.values()))
        print('Epoch: [{}/{}], ValAUC: {}, ValAcc: {}'.format(epoch+1, num_epochs, val_auc.values(), val_acc.values()))
        if step_max_descent == step_num_descent:
            break
    for key in val_AUC_dict.keys():
        val_auc_mean[key] = np.mean(val_AUC_dict[key][-step_max_descent*2-1:])
        val_acc_mean[key] = np.mean(val_ACC_dict[key][-step_max_descent*2-1:])
    return val_auc_mean, val_acc_mean, model

RANDOM_STATE = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_path='/scratch/tx443/CapstonePlated/Plated_Recipe_Tags_Predict/data/'
# # Get Data 

fname = '/scratch/tx443/CapstonePlated/data/glove.6B.50d.txt'
words_emb_dict = load_emb_vectors(fname)


steps = ['step_one','step_two', 'step_three', 'step_four', 'step_five', 'step_six']
steps_aug = ['step_one_sp', 'step_two_sp', 'step_three_sp',
             'step_four_sp', 'step_five_sp', 'step_six_sp']
tags = ['tag_cuisine_indian', 'tag_cuisine_nordic', 'tag_cuisine_european',
        'tag_cuisine_asian', 'tag_cuisine_mexican',
        'tag_cuisine_latin-american', 'tag_cuisine_french',
        'tag_cuisine_italian', 'tag_cuisine_african',
        'tag_cuisine_mediterranean', 'tag_cuisine_american',
        'tag_cuisine_middle-eastern']


# In[5]:


data_with_aug = pd.read_csv(data_path+'recipe_data_with_aug.csv', index_col=0)
data_with_aug_tags = data_with_aug[steps+steps_aug+tags]
print(data_with_aug_tags.columns)


# Tokenization

print('Processing original instruction data')
# tokenize each steps on original datasets
steps_token = []
for step in steps:
    steps_token.append(step+'_token')
    data_with_aug_tags[step+'_token'] = tokenize_dataset(data_with_aug_tags[step])
    print(step, 'has been tokenized.')

# tokenize each steps on augmented datasets
print('Processing augmented instruction data')
steps_aug_token = []
for step in steps_aug:
    steps_aug_token.append(step+'_token')
    data_with_aug_tags[step+'_token'] = tokenize_dataset(data_with_aug_tags[step])
    print(step, 'has been tokenized.')


data_with_aug_tags = data_with_aug_tags[steps_token+steps_aug_token+tags]
data_with_aug_tags.columns


# # Split train, validation, test sets
train_val_data, test_data = train_test_split(data_with_aug_tags, test_size=0.1, random_state=RANDOM_STATE)
test_data = test_data[steps_token+tags]
#train_data, val_data, train_tags, val_tags = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

aug2ori_colname = dict(zip(steps_aug_token+tags, steps_token+tags))
tags_predicted = ['tag_cuisine_american', 'tag_cuisine_italian', 'tag_cuisine_asian', 'tag_cuisine_latin-american', 'tag_cuisine_french', 
    'tag_cuisine_mediterranean', 'tag_cuisine_middle-eastern', 'tag_cuisine_indian', 'tag_cuisine_mexican']


test_targets = []
for row in test_data[tags_predicted].iterrows():
    test_targets.append(list(row[1].values))

rnn_types = {
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU
    }

params = dict(
    rnn1_type = 'gru',
    rnn2_type = 'gru',
    bi = False,
    tags_predicted = tags_predicted,
    
    hidden_dim1 = 30,
    hidden_dim2 = 30,
    num_classes = 1,
    
    multi_task_train = 'mean_loss', #{'mean_loss', 'random_selection'}
    num_epochs = 20,
    batch_size = 50,
    learning_rate = 0.01,
    step_max_descent = 3,
    
    add_data_aug = True,
    cuda_on = True,
    loss_weight_on = True
    )

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
k = 1 
model_candidate_kf = []
val_auc_kf = defaultdict(list)
val_acc_kf = defaultdict(list)

for train_index, val_index in kf.split(train_val_data):
    print('===================== This is the Kfold {} ====================='.format(k))
    k += 1
    val_data = train_val_data[steps_token+tags].iloc[val_index]
    train_data = train_val_data.iloc[train_index]
    
    if params['add_data_aug']:
        ##### add augmentation to training set by index #####
        train_org = train_data[steps_token+tags]
        train_aug = train_data[steps_aug_token+tags]
        train_aug.rename(index=str, columns=aug2ori_colname, inplace=True)
        train_data = pd.concat([train_org, train_aug], axis=0, ignore_index=False)
        ##### add augmentation to training set by index #####
    else:
        train_data = train_data[steps_token+tags]

    pos_num_tags = np.zeros(shape=(len(tags_predicted),))
    train_targets = []
    for row in train_data[tags_predicted].iterrows():
        pos_num_tags += row[1].values
        train_targets.append(list(row[1].values))
    val_targets = []
    for row in val_data[tags_predicted].iterrows():
        val_targets.append(list(row[1].values))
    
    train_sample_num = len(train_tags)
    if params['loss_weight_on']:
        loss_weights = {}
        for idx in range(len(tags_predicted)):
            loss_weights[i] = torch.Tensor([(train_sample_num-pos_num_tags[idx])/pos_num_tags[idx]]).to(device)
    else:
        loss_weights = None
    train_X = train_data[steps_token]
    val_X = val_data[steps_token]
    test_X = test_data[steps_token]
    all_train_tokens = all_tokens_list(train_X)
    max_vocab_size = len(list(set(all_train_tokens)))
    token2id, id2token = build_vocab(all_train_tokens, max_vocab_size)
    emb_weight = build_emb_weight(words_emb_dict, id2token)
    train_data_indices = token2index_dataset(train_X, token2id)
    val_data_indices = token2index_dataset(val_X, token2id)
    test_data_indices = token2index_dataset(test_X, token2id)

    # batchify datasets: 
    batch_size = params['batch_size']
    max_sent_len = np.array([94, 86, 87, 90, 98, 91])
    train_loader, val_loader, test_loader = create_dataset_obj(train_data_indices, val_data_indices,
                                                           test_data_indices, train_targets,
                                                           val_targets, test_targets,
                                                           batch_size, max_sent_len, 
                                                           collate_func)
    
    val_auc, val_acc, model_to_test = train_model(params, emb_weight, train_loader, val_loader, test_loader. loss_weights)
    model_candidate_kf.append(model_to_test)
    for key in val_auc.keys():
        val_auc_kf[key].append(val_auc[key])
        val_acc_kf[key].append(val_acc[key])
 
