import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from scipy.signal import cwt
from scipy.signal import morlet
from scipy import signal
from tqdm import tqdm_notebook
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from ddim_utils import funcx_z,funcz_x
from HP_config import HP_EEGNET_FeelingEmotions,HP_DEEP_FeelingEmotions,HP_Shallow_FeelingEmotions,\
    HP_EEGNET_Deap,HP_DEEP_Deap,HP_Shallow_Deap,HP_EEGNET_BCICIV2a,HP_DEEP_BCICIV2a,HP_Shallow_BCICIV2a

class EEGDataset(Dataset):
    def __init__(self, inputs, labels):
        self.x_train = torch.tensor(inputs)
        self.y_train = torch.tensor(labels)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
def select_HP(datatype, modeltype):

    if datatype == 'FeelingEmotions':
        if modeltype=='EEGNet':
            HP = HP_EEGNET_FeelingEmotions
        elif modeltype == 'DeepConvNet':
            HP = HP_DEEP_FeelingEmotions
        elif modeltype == 'ShallowConvNet':
            HP = HP_Shallow_FeelingEmotions
    elif datatype == 'BCICIV2a':
        if modeltype == 'EEGNet':
            HP = HP_EEGNET_BCICIV2a
        elif modeltype == 'DeepConvNet':
            HP = HP_DEEP_BCICIV2a
        elif modeltype == 'ShallowConvNet':
            HP = HP_Shallow_BCICIV2a
    elif datatype == 'DEAParo' or datatype == 'DEAPval':
        if modeltype == 'EEGNet':
            HP = HP_EEGNET_Deap
        elif modeltype == 'DeepConvNet':
            HP = HP_DEEP_Deap
        elif modeltype == 'ShallowConvNet':
            HP = HP_Shallow_Deap
    return HP



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = "/data/pycharm_sshcjf/diffusion_eeg/DDIM_datacondensation/data_split/"
datachoice = "BCICIV2a"
epochs = 150 # 1 for test
pretrained_modelpath =("/data/DDIM_datacondensation/model_best/"+ \
                       datachoice+"withchannelstandardTransformer_Diffusionbestv1")
save_syn_path_main  = '/data/DDIM_datacondensation/data_result/'
train_sam = np.load(save_path+datachoice+"train_sample_fold01.npy")
train_lab = np.load(save_path+datachoice+"train_labfold01.npy")

N, channel,num_features = train_sam.shape[0],train_sam.shape[1],train_sam.shape[2]
print(datachoice)
train_lab = train_lab.astype(np.int32)
num_classes = np.unique(train_lab).shape[0]

print("num_Class", num_classes)
from diffusion import create_diffusion
import model_tr

model = model_tr.DiffT(input_size=num_features,in_channels = channel,hidden_size=1024).to(device) if datachoice == "BCICIV2a" \
    else model_tr.DiffT(input_size=num_features,in_channels = channel,hidden_size=512).to(device)
state_dict = torch.load(pretrained_modelpath,map_location=device)
model.load_state_dict(state_dict)
for param in model.parameters():
    param.requires_grad = False

num_sampling_steps = 1000
diffusion = create_diffusion(str(num_sampling_steps))
indices_class = [[] for c in range(num_classes)]

for i, lab in enumerate(train_lab):
    indices_class[lab].append(i)

seed = 32
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed) 

def inint_select(x_start, N):
    torch.manual_seed(seed)
    ALL_N = x_start.shape[0]
    choice = torch.randint(low=0, high=ALL_N, size=(N,))
    return x_start[choice]

net_distribution = 'random'
is_standard = "withchannelstandard"
match_type = 'feat'
x_real2z_split = []
optm_split = []
z_split = []
lab_split = []
minloss_split = []
save_tag = True
TRdata_, TRlab_ = train_sam, train_lab
print(TRdata_.shape)
learning_rate = 0.005
for ratio_div in [0.02]:
    for (N_initial_sample,batchnum_fortrain) in [(50,10)]:
        dataloader_list = []
        for i in range(num_classes):
            dataset_classi = EEGDataset(TRdata_[indices_class[i]], TRlab_[indices_class[i]])
            dataloader_classi = DataLoader(dataset_classi, batch_size=batchnum_fortrain, shuffle=True)
            dataloader_list.append(dataloader_classi)
        for c in range(num_classes):

            x_c = inint_select(TRdata_[indices_class[c]], N_initial_sample)
            x_c = torch.tensor(x_c, dtype=torch.float, requires_grad=False, device=device)
            x_real2z_split.append(x_c)
            z_c__ = funcx_z(x_c, N_initial_sample，diffusion)
            z_c_ = z_c__.clone().detach().cpu().numpy()
            z_c = torch.tensor(z_c_, dtype=torch.float, requires_grad=True, device=device)  # z_c这边是对应上的
            z_split.append(z_c)
            lab_c = torch.tensor(c * np.ones((N_initial_sample,)), dtype=torch.long, requires_grad=False, device=device)
            lab_split.append(lab_c)
            optm_c =  torch.optim.SGD([z_c, ], lr=learning_rate, momentum=0.5)

            optm_split.append(optm_c)
            minloss_split.append(100000)
        save_x_real = np.zeros((0, TRdata_.shape[1], TRdata_.shape[2]))
        for i in range(num_classes):
            save_x_real = np.concatenate([save_x_real, x_real2z_split[i].cpu().numpy()], axis=0)
        if save_tag:
            np.save(save_syn_path_main + str(datachoice) + '/' + str(
                N_initial_sample) + 'sample' + is_standard + 'real_initial.npy', save_x_real)
        print('initial Z and save SUCCESS')
        criterion = nn.CrossEntropyLoss().to(device)


        from model_pool import EEGNet , ShllowCNN, DeepCNN

        modelpool_list = ['EEGNet','Shllow']
        for epoch in range(epochs):
            random_integer = random.randint(0, 2)
            if random_integer==0:
                HP_ = select_HP(datachoice, 'EEGNet')
                net = EEGNet(HP_)
            elif random_integer==1:
                HP_ = select_HP(datachoice, 'ShllowCNN')
                net = ShllowCNN(HP_)
            elif random_integer==2:
                HP_ = select_HP(datachoice, 'DeepCNN')
                net = DeepCNN(HP_)
            
            print('epoch:' + str(epoch) + '/' + str(epochs))
            loss_divs_all = []
            loss_cond_all = []
            loss_all = []

            for c in range(num_classes):
                net = net.to(device)
                net.train()  

                if match_type == 'feat':  
                    for param in net.parameters():
                        param.requires_grad = False
                for step, (x, y) in enumerate(
                        dataloader_list[c]):  

                    z_syn = z_split[c]
                    lab_syn = lab_split[c].detach()
                    img_syn = funcz_x(model, z_syn, lab_syn)

                    feat_syn, _ = net(img_syn)
    
                    if ratio_div > 0.00001:
                        img_real = x_real2z_split[c].detach()
                        feat_real, _ = net(img_real)
                        feat_real = feat_real.detach()
                        loss_divs = torch.mean(torch.sum((feat_syn - feat_real) ** 2, dim=-1))
                    else:
                        loss_divs = torch.tensor(0.0).to(device)
                    img_cond = x.to(torch.float).to(device)
                    y = y.cpu().numpy()
                    lab_cond = torch.tensor((y), dtype=torch.long, requires_grad=False, device=device)
                    if match_type == 'feat':  
                        feat_cond_, _ = net(img_cond)
                        feat_cond_ = feat_cond_.detach()
                        feat_cond = torch.mean(feat_cond_, dim=0)  
                        loss_cond = torch.sum((torch.mean(feat_syn, dim=0) - feat_cond) ** 2)

                    loss = ratio_div * loss_divs + (1 - ratio_div) * loss_cond
                    if step < 10:
                        print("class:" + str(c) + "this epoch " + str(epoch) + "step" + "loss_cond:", loss_cond)
                        print("loss_divs:", loss_divs)
                        print("loss", loss)
                        print("----------------------------")
                    optm_split[c].zero_grad()
                    loss.backward()
                    optm_split[c].step()
                    loss_divs_all.append(loss_divs.item())
                    loss_cond_all.append(loss_cond.item())
                    loss_all.append(loss.item())
                avg_totalloss = sum(loss_all) / len(loss_all)
                avg_divsloss = sum(loss_divs_all) / len(loss_divs_all)
                avg_condloss = sum(loss_cond_all) / len(loss_cond_all)
                print("class:" + str(c) + " epoch: ", epoch + 1, " avg_totalloss: ", avg_totalloss)
                print("class:" + str(c) + " epoch: ", epoch + 1, " avg_divsloss: ", avg_divsloss)
                print("class:" + str(c) + " epoch: ", epoch + 1, " avg_condloss: ", avg_condloss)
                if avg_totalloss < minloss_split[c]:
                    minloss_split[c] = avg_totalloss
                    print("current best totalloss:", avg_totalloss)
                    savez = z_split[c].clone().detach()
                    savesyn = img_syn.clone().detach()
                    if save_tag:
                        np.save(
                            save_syn_path_main + str(datachoice) + '/EEGCiDlambda' + str(ratio_div) + str(
                                N_initial_sample) + 'sample' + is_standard + 'class' + str(c) + 'savezbestv1.npy',
                            savez.cpu().numpy())
                        np.save(
                            save_syn_path_main + str(datachoice) + '/EEGCiDlambda' + str(ratio_div) + str(
                                N_initial_sample) + 'sample' + is_standard + 'class' + str(c) + 'savesynbestv1.npy',
                            savesyn.cpu().numpy())
                        print("savebest" + 'class' + str(c) + 'success')
