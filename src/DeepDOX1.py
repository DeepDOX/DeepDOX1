import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import csv
import metrics as metrics
from torch.utils.data import DataLoader
from math import sqrt
import random
from scipy.stats import spearmanr

CHAR_SMI_SET_LEN = 64
PT_FEATURE_SIZE = 40
PT_NEW_FEATURE_SIZE = 65

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze(3)

class Squeeze2(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze(1)

class Squeeze3(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze(2)

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DeepDOXA(nn.Module):
    def __init__(self):
        super().__init__()
        smi_oc = 128
        fet_oc = 128

        Feat_size = 128
        Fea_size = 128

        self.smi_embed = nn.Linear(CHAR_SMI_SET_LEN, smi_oc)
        self.fet_embed = nn.Linear(PT_NEW_FEATURE_SIZE, fet_oc)  

        conv_fev = []
        cont_fev = []
        conv_fev.append(nn.Conv2d(100,100,[1,64],stride=[1,64],bias=True))
        conv_fev.append(nn.BatchNorm2d(100))
        conv_fev.append(nn.PReLU())
        conv_fev.append(Squeeze())
        conv_fev.append(Squeeze3())
        self.conv_fev = nn.Sequential(*conv_fev)
        cont_fev.append(nn.Conv2d(100,100,[1,1],stride=[1,1],bias=False))
        cont_fev.append(nn.BatchNorm2d(100))
        cont_fev.append(nn.PReLU())
        cont_fev.append(Squeeze())
        cont_fev.append(Squeeze3())
        self.cont_fev = nn.Sequential(*cont_fev)
        conv_small = []
        conv_small.append(nn.Conv2d(150,150,[1,64],stride=[1,64],bias=False))
        conv_small.append(nn.BatchNorm2d(150))
        conv_small.append(nn.PReLU())
        conv_small.append(Squeeze())
        conv_small.append(Squeeze3())
        conv_small.append(nn.Linear(150,100))
        conv_small.append(nn.PReLU())
        self.conv_small = nn.Sequential(*conv_small)
        self.cat_dropout = nn.Dropout(0.1)
        self.att1 = nn.MultiheadAttention(100,4,batch_first=True)
        self.att2 = nn.MultiheadAttention(100,5,batch_first=True)
        self.prolin = nn.Sequential(
            nn.Linear(100,256),
            nn.PReLU(),
            nn.Linear(256,64),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU()
        )
        self.Lin = nn.Sequential()
        self.Tan = nn.Sequential(nn.Linear(1,1),
                                 nn.PReLU())
        self.ProLine2 = nn.Sequential(
            nn.Linear(100,128),
            nn.PReLU(),
            nn.Linear(128,100),
            nn.PReLU(),
            nn.Linear(100,16),
            nn.PReLU(),
            nn.Linear(16,1),
            nn.PReLU())
        self.SmaLine2 = nn.Sequential(
            nn.Linear(100,128),
            nn.PReLU(),
            nn.Linear(128,64),
            nn.PReLU(),
            nn.Linear(64,16),
            nn.PReLU(),
            nn.Linear(16,1),
            nn.PReLU()
        )
        self.Line1 = nn.Sequential(
            nn.Linear(100,100),
            nn.PReLU())
        self.classifier2 = nn.Sequential(
            nn.Linear(2,1),
            nn.PReLU()
        )

        self.dealener = nn.Sequential(nn.Linear(128,16),
                                      nn.PReLU(),
                                        nn.Linear(16,1),
                                        nn.PReLU()
        )
        self.test = nn.Sequential(nn.BatchNorm1d(2),
                                nn.Linear(2,1),
                                nn.PReLU()
        )
        self.rdkitclassify = nn.Sequential(
            nn.Linear(15,8),
            nn.PReLU(),
            nn.Linear(8,4),
            nn.PReLU(),
            nn.Linear(4,1),
            nn.PReLU()
        )
        self.classfy = nn.Sequential(
            nn.Linear(79,64),
            nn.PReLU(),
            nn.Linear(64,16),
            nn.PReLU(),
            nn.Linear(16,1),
            nn.PReLU()
        )
        self.dealRdkit = nn.Sequential(
            nn.Linear(12,8),
            nn.PReLU(),
            nn.Linear(8,1),
            nn.PReLU()
        )
        self.Embading1 = nn.Sequential(nn.Linear(64,64),
                                       nn.PReLU())
        self.Embading2 = nn.Sequential(nn.Linear(64,64),
                                        nn.PReLU())
        
    
    def forward(self, small,info,fet,proinfo,ligandfinger,ligandentropy):

        fetsize = fet.shape[0]
        fet = fet.reshape(fetsize,100,1,64)
        fev = self.conv_fev(fet)
        small1 = small.reshape(fetsize,150,1,64)
        small1 = self.conv_small(small1)
        small = torch.squeeze(small,1)
        small = torch.sum(small,1)
        proinfo = proinfo.reshape(fetsize,100,1,1)
        proener = self.cont_fev(proinfo)

        proener,proenerwe = self.att1(proener,proener,proener)
        fev1,fevwe = self.att1(fev,fev,fev)

        fev = fev1 + fev
        proener = proener + proener

        fev = self.Line1(fev)
        proener = self.Line1(proener)
        
        fev2,fev1we = self.att1(fev,small1,small1)

        fev = fev + fev2 + proener
        fev = self.Line1(fev)

        fev3,fev2we = self.att2(fev,fev,fev)
        fev = fev + fev3
        fev = self.Line1(fev)
        
        fev = fev + proener 
        
        fev = self.ProLine2(fev)

        ligandfinger = Squeeze2()(ligandfinger)
        if small.ndim != 1:
            info = info.reshape(small.shape[0], -1)
            ligandentropy = ligandentropy.reshape(small.shape[0], -1)
            cat = torch.cat([ligandfinger,info,ligandentropy,fev,small],1)
        else:
            cat = torch.cat([ligandfinger,info,ligandentropy,fev,small],0)

        x = self.classfy(cat)
        return x
    
def test(model: nn.Module, data, loss_function, device, show, testppid,isTest=False, test_file_path = r"E:\PHD\TS_G_ml\datatest\test.csv"):
    model.eval()
    test_loss = 0
    output_list = []
    target_list = []
    test_dict = {}
    with torch.no_grad():
        m = 0
        n = 0
        testdata = tqdm(enumerate(data), disable=not show,total=len(data))
        for i,(protein, proener, ligand, energy,label,ligandfinger,lignadentropy) in testdata:
            protein = protein.to(device).to(torch.float32)
            ligand = ligand.to(device).to(torch.float32)
            proener = proener.to(device).to(torch.float32)
            energy = energy.to(device).to(torch.float32)
            label = label.to(device).to(torch.float32)
            ligandfinger = ligandfinger.to(device).to(torch.float32)
            ligandentropy = lignadentropy.to(device).to(torch.float32)
            predict = model(ligand,energy,protein,proener,ligandfinger,ligandentropy)
            test_loss += loss_function(predict.view(-1),label.view(-1)).item()
            testdata.set_description(f' * Test Loss={test_loss / len(data):.3f}')
            if isTest == False:
                for i in range(len(label)):
                    pdbid = str(testppid[n])
                    predict_result = predict[i].item()
                    dict_key = pdbid
                    test_dict[dict_key] = predict_result
                    n += 1
            if isTest:
                with open(test_file_path, 'a') as f1:
                    for i in range(len(label)):
                        pdbid = str(testppid[m])
                        m += 1
                        f1.write('{} \t {} \t {}\n'.format(pdbid, predict[i].item(),label[i].item()))
                        test_dict[pdbid] = predict[i].item()
            output_list.append(predict.detach().cpu().numpy().reshape(-1))
            target_list.append(label.detach().cpu().numpy().reshape(-1))
    
    test_loss /= len(data.dataset)
    
    target = np.concatenate(target_list).reshape(-1)
    output = np.concatenate(output_list).reshape(-1)

    SpearmanR = spearmanr(target, output)
    Spearman = SpearmanR.statistic
    
    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(target, output),
        'RMSE': metrics.RMSE(target, output),
        'MAE': metrics.MAE(target, output),
        'SD': metrics.SD(target, output),
        'CORR': metrics.CORR(target, output),
        'R2': r2_score(target, output),
        'SP' : Spearman
    }


    return evaluation, test_dict

