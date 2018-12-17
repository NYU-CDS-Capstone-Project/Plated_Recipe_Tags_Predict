# coding: utf-8
import pandas as pd
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import pickle as pkl
import random
import re
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import os 

import matplotlib.pyplot as plt
from functools import partial
from PIL import Image


def load_image(image_path):
    try:
        image = Image.open(image_data_path+'/'+image_path).convert('RGB')
    except:
        print('empty image')
    return image

normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_image = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(224),
    #tv.transforms.Resize((224,224)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    normalize,])

class ImageDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, imgs_add, tags, load_image, transform_image):
        """
        @param data_list: list of recipie tokens 
        @param target_list: list of single tag, i.e. 'tag_cuisine_american'
        """
        self.data = imgs_add
        self.tags = tags
        self.load = load_image
        self.transform = transform_image
        assert (len(self.data) == len(self.tags))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        """
        Triggered when  you call recipie[i]
        """
        image = self.transform(self.load(self.data[key]))
        label = self.tags[key]
        return (image, label)

def collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    images = []
    label_dict = defaultdict(list)
    for datum in batch:
        images.append(datum[0])
        for task_id, task_label in enumerate(datum[1]):
            label_dict[task_id].append(task_label)
    for task_id in label_dict.keys():
        label_dict[task_id] = torch.LongTensor(label_dict[task_id]).to(device)
    images = np.stack(images, axis=0)
    return [torch.from_numpy(images).to(device), label_dict]


# Build train, valid and test dataloaders
def create_dataset_obj(train,val,test,train_targets,val_targets,test_targets,
                       load_image,transform_image,batch_size,collate_func):
    train_dataset = ImageDataset(train, train_targets, load_image, transform_image)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=collate_func,
                                               shuffle=True)

    val_dataset = ImageDataset(val, val_targets, load_image, transform_image)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             collate_fn=collate_func,
                                             shuffle=False)

    test_dataset = ImageDataset(test, test_targets, load_image, transform_image)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               collate_fn=collate_func,
                                               shuffle=False)
    return train_loader, val_loader, test_loader


class Image_CNN(nn.Module):
    def __init__(self, num_tasks, hidden_dim, num_classes, train_resnet=False):
        super(Image_CNN, self).__init__()
        self.num_tasks = num_tasks
        resnet = tv.models.resnet50(pretrained=True)
        resnet_modules = list(resnet.children())[:-1]
        self.resnet50 = nn.Sequential(*resnet_modules)
        if train_resnet is False:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        self.linear1 = nn.Linear(2048, hidden_dim)
        linear2 = nn.ModuleList([])
        for i in range(num_tasks):
            linear2.append(nn.Linear(hidden_dim, num_classes))    
        self.linear2 = linear2

    def forward(self, img):
        hidden = self.resnet50(img)
        hidden = torch.squeeze(hidden)
        #print(hidden.size())
        hidden = self.linear1(hidden)
        hidden = F.relu(hidden)
        logits = {}
        for task_id in range(self.num_tasks):
            logits[task_id] = self.linear2[task_id](hidden)
        return logits

from sklearn.metrics import roc_auc_score
def test_model(loader, model, threshold=0.5):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    logits_all_dict = defaultdict(list)
    labels_all_dict = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    for image_batch, labels_batch in loader:
        logits = model(image_batch)
        for i in labels_batch.keys():
            logits_all_dict[i].extend(list(F.sigmoid(logits[i]).cpu().detach().numpy()))
            labels_all_dict[i].extend(list(labels_batch[i].cpu().numpy()))
    auc = {}
    acc = {}
    for i in labels_all_dict.keys():
        logits_all_dict[i] = np.array(logits_all_dict[i])
        labels_all_dict[i] = np.array(labels_all_dict[i])
        auc[i] = roc_auc_score(labels_all_dict[i], logits_all_dict[i])
        predicts = (logits_all_dict[i] > threshold).astype(int)
        acc[i] = np.mean(predicts==labels_all_dict[i])
    return auc, acc


def train_model(params, train_loader, val_loader, test_loader, loss_weights):
    num_classes = params['num_classes']
    hidden_dim = params['hidden_dim']
    batch_size = params['batch_size']
    train_resnet = params['train_resnet']
    
    multi_task_train = params['multi_task_train'] 
    tags_predicted = params['tags_predicted']
    num_tasks = len(tags_predicted)
    model = Image_CNN(num_tasks, hidden_dim, num_classes, train_resnet)
    # load pretrained data
    im2recipe_resnet_pretrained = torch.load('model/model_e500_v-8.950.resnet.pth')
    model_dict = model.state_dict()
    model_dict.update(im2recipe_resnet_pretrained['resnet50_im2recipe'])
    model.load_state_dict(model_dict)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('The number of train parameters', sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)

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
        for i, (images_batch, labels_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            logits = model(images_batch)
            if multi_task_train == 'mean_loss':
                loss_list = []
                for task_id in range(num_tasks):
                    loss_list.append(criterion[task_id](logits[task_id], 
                                          labels_batch[task_id].view(-1,1).float()))
                loss= torch.mean(torch.stack(loss_list))
            elif multi_task_train == 'random_selection':
                task_id = np.random.randint(0, num_tasks)
                loss = criterion[task_id](logits[task_id], labels_batch[task_id].view(-1,1).float())
            else:
                print('multi-task-train-method Error')
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
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
                   # break
        val_auc, val_acc = test_model(val_loader, model)
        train_auc, train_acc = test_model(train_loader, model)
        print('Epoch: [{}/{}], trainAUC: {}, trainAcc: {}'.format(epoch+1, num_epochs, train_auc.values(), train_acc.values()))
        print('Epoch: [{}/{}], ValAUC: {}, ValAcc: {}'.format(epoch+1, num_epochs, val_auc.values(), val_acc.values()))
        check_point_save = {
                        'model': model.state_dict()
			}
        torch.save(check_point_save, model_path+'epoch_{}.pth'.format(epoch))
        #if step_max_descent == step_num_descent:
           # break
    for key in val_AUC_dict.keys():
        val_auc_mean[key] = np.mean(val_AUC_dict[key][-step_max_descent*2-1:])
        val_acc_mean[key] = np.mean(val_ACC_dict[key][-step_max_descent*2-1:])
    return val_auc_mean, val_acc_mean, model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
image_data_path = 'data/images'

data_recipe_image = pd.read_csv('Plated_Recipe_Tags_Predict/data/recipe_image_data_with_cuisineTags.csv', index_col=0)

#RANDOM_STATE = 42
train, test_data= train_test_split(data_recipe_image, test_size=0.1)#, random_state=RANDOM_STATE)
train_data, val_data = train_test_split(train, test_size=0.2) #, random_state=RANDOM_STATE)

tags_predicted = ['tag_cuisine_mediterranean',] 
#['tag_cuisine_american', 'tag_cuisine_italian', 'tag_cuisine_asian', 
#                  'tag_cuisine_latin-american', 'tag_cuisine_french', 
#                  'tag_cuisine_mediterranean', 'tag_cuisine_middle-eastern', 
#                  'tag_cuisine_indian', 'tag_cuisine_mexican']
test_targets = []
for row in test_data[tags_predicted].iterrows():
    test_targets.append(list(row[1].values))

model_path = './{}_model/'.format(tags_predicted[0])
if not os.path.exists(model_path):
    os.makedirs(model_path)

params = dict(
    tags_predicted = tags_predicted,
    hidden_dim = 30,
    num_classes = 1,
    
    multi_task_train = 'mean_loss', #{'mean_loss', 'random_selection'}
    num_epochs = 50,
    batch_size = 50,
    learning_rate = 5e-4,
    train_resnet = False,
    
    step_max_descent = 10,
    loss_weight_on = True
)


# kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# k = 1 
model_candidate_kf = []
val_auc_kf = defaultdict(list)
val_acc_kf = defaultdict(list)

# for train_index, val_index in kf.split(train_val_data):
if True:
#     print('===================== This is the Kfold {} ====================='.format(k))
#     k += 1
#     val_data = train_val_data[steps_token+tags].iloc[val_index]
#     train_data = train_val_data.iloc[train_index]

    pos_num_tags = np.zeros(shape=(len(tags_predicted),))
    train_targets = []
    for row in train_data[tags_predicted].iterrows():
        pos_num_tags += row[1].values
        train_targets.append(list(row[1].values))
    val_targets = []
    for row in val_data[tags_predicted].iterrows():
        val_targets.append(list(row[1].values))
    
    train_sample_num = len(train_targets)
    if params['loss_weight_on']:
        loss_weights = {}
        for idx in range(len(tags_predicted)):
            loss_weights[idx] = torch.Tensor([(train_sample_num-pos_num_tags[idx])/pos_num_tags[idx]]).to(device)
    else:
        loss_weights = None
    
    train_img = list(train_data['image_filename'].values)
    val_img = list(val_data['image_filename'].values)
    test_img = list(test_data['image_filename'].values)

    # batchify datasets: 
    batch_size = params['batch_size']
    max_sent_len = np.array([94, 86, 87, 90, 98, 91])
    train_loader, val_loader, test_loader = create_dataset_obj(train_img, val_img, test_img, 
                                                               train_targets, val_targets, 
                                                               test_targets, load_image,
                                                               transform_image, batch_size, 
                                                               collate_func)
    
    val_auc, val_acc, model_to_test = train_model(params, train_loader, val_loader, test_loader, loss_weights)
    model_candidate_kf.append(model_to_test)
    for key in val_auc.keys():
        val_auc_kf[key].append(val_auc[key])
        val_acc_kf[key].append(val_acc[key])

print(val_auc_kf, val_acc_kf)

