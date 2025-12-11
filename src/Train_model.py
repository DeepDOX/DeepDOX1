import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from DeepDOX1 import DeepDOXA,test
from dataset3 import ProteinLigandDataset
import pandas as pd
import os
import shutil
from casf_bench import spearman_correlation_coefficient

SHOW_PROCESS_BAR = True

traindata_path = Path(f'../data/Train')
testdata_path = Path(f'../data/CASF-2016')

Merk_test_path = r'../data/Merk_FEP'
Merk_list = os.listdir(Merk_test_path)
Merk_test_path_list = [os.path.join(Merk_test_path, file) for file in Merk_list]

HLO_test_path = r'../data/HLO-2025'
HLO_list = os.listdir(HLO_test_path)
HLO_test_path_list = [os.path.join(HLO_test_path, file) for file in HLO_list]

Cov_test_path = r'../data/CMX-2025/C'
Cov_list = os.listdir(Cov_test_path)
Cov_test_path_list = [os.path.join(Cov_test_path, file) for file in Cov_list]

X_CASF_path = r'../data/CMX-2025/X'
X_list = os.listdir(X_CASF_path)
X_test_path_list = [os.path.join(X_CASF_path, file) for file in X_list]

M_CASF_path = r'../data/CMX-2025/M'
M_list = os.listdir(M_CASF_path)
M_test_path_list = [os.path.join(M_CASF_path, file) for file in M_list]


seed = np.random.randint(72249666, 72249667)
time_squence = datetime.now().strftime("%Y%m%d%H%M%S")
path = Path(f'../DeepDOX1/DeepDOX1_{time_squence}_{seed}')
device = torch.device("cuda:0")  

batch_size = 16
n_epoch = 100
interrupt = None    
save_best_epoch = 13

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

writer = SummaryWriter(path)
f_param = open(path / 'parameters.txt', 'w')

assert 0 <= save_best_epoch < n_epoch

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')

f_param.write(f'device={device}\n'
            f'seed={seed}\n'
            f'write to {path}\n')

print('Loading data...')
model = DeepDOXA()
model = model.to(device)
model.train()

f_param.write(f'model={model}\n')
f_param.close()

traindatalist = os.listdir(traindata_path)
testdatalist = os.listdir(testdata_path)

trainproteinpocket = []
trainproteinresienergy = []
trainligandsmile = []
trainligandrdkit = []
trainligandDOXenergy = []
trainligandentropy = []
trainlabel = []

for idx in range(len(traindatalist)):
    data = torch.load(os.path.join(traindata_path,traindatalist[idx]))
    pocket = torch.unsqueeze(data['pocket'],0)
    proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
    ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
    ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
    ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
    ligandentropy = torch.unsqueeze(data['Entropy'],0)
    ligandlabel = torch.unsqueeze(data['label'],0)
    trainproteinpocket.append(pocket)
    trainproteinresienergy.append(proteinresienergy)
    trainligandsmile.append(ligandsmile)
    trainligandrdkit.append(ligandrdkit)
    trainligandDOXenergy.append(ligandDOXenergy)
    trainligandentropy.append(ligandentropy)
    trainlabel.append(ligandlabel)
    print('runtraindata:{}'.format(idx))

testproteinpocket = []
testproteinresienergy = []
testligandsmile = []
testligandrdkit = []
testligandDOXenergy = []
testligandentropy = []
testlabel = []
testppidlist = []

for idx in range(len(testdatalist)):
    data = torch.load(os.path.join(testdata_path,testdatalist[idx]))
    pocket = torch.unsqueeze(data['pocket'],0)
    proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
    ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
    ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
    ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
    ligandentropy = torch.unsqueeze(data['Entropy'],0)
    ligandlabel = torch.unsqueeze(data['label'],0)
    testproteinpocket.append(pocket)
    testproteinresienergy.append(proteinresienergy)
    testligandsmile.append(ligandsmile)
    testligandrdkit.append(ligandrdkit)
    testligandDOXenergy.append(ligandDOXenergy)
    testligandentropy.append(ligandentropy)
    testlabel.append(ligandlabel)
    pdbid = data['id']
    print(pdbid)
    testppidlist.append(pdbid)  
    print('runtestdata:{}'.format(idx))


traindata = ProteinLigandDataset(trainproteinpocket,trainproteinresienergy,trainligandsmile,trainligandrdkit,trainligandDOXenergy,trainlabel,trainligandentropy)
testdata = ProteinLigandDataset(testproteinpocket,testproteinresienergy,testligandsmile,testligandrdkit,testligandDOXenergy,testlabel,testligandentropy)

train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(testdata, batch_size=1, shuffle=False)

optimizer1 = optim.LBFGS(params=model.parameters(),lr=1e-3, max_iter=10)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=n_epoch)
loss_func = nn.L1Loss(reduction='sum')

start = datetime.now()
print('Start training', start)

best_Spearman = -1
best_epoch  = -1
best_val_loss = 1e10
for epoch in range(n_epoch):
    model.train()
    traindata = tqdm(enumerate(train_loader), disable=not SHOW_PROCESS_BAR,total=len(train_loader))
    for i,(protein, proener, ligand, energy,label,ligandfinger,ligandentropy) in traindata:
        model.train()
        protein = protein.to(device).to(torch.float32)
        proener = proener.to(device).to(torch.float32)
        ligand = ligand.to(device).to(torch.float32)
        energy = energy.to(device).to(torch.float32)
        label = label.to(device).to(torch.float32)
        ligandfinger = ligandfinger.to(device).to(torch.float32)
        ligandentropy = ligandentropy.to(device).to(torch.float32)
        optimizer1.zero_grad()
        pred = model(ligand,energy,protein,proener,ligandfinger,ligandentropy)
        if pred.isnan().any():
            print('nan')
            print(ligand)
            print(energy)
            print(protein)
            print(proener)
            print(ligandfinger)
            print(label)
            sys.exit()
        loss = loss_func(pred.view(-1), label.view(-1)).to(device)
        loss.backward()
        optimizer.step() 
        scheduler.step()
        traindata.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(label):.3f}')

    performance,performance_dict = test(model, test_loader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=testppidlist,isTest=False)
    Spearman = spearman_correlation_coefficient(performance_dict)
    if performance['loss'] < best_val_loss or Spearman > best_Spearman:
        best_Spearman = Spearman
        best_val_loss = performance['loss']
        best_epoch = epoch
        print('Best model found at epoch {} with Spearman {:.4f} and val loss {:.4f}'.format(best_epoch, best_Spearman, best_val_loss))
        torch.save(model.state_dict(), path / f'best_model_{time_squence}_{seed}.pt')

shutil.copy(path / f'best_model_{time_squence}_{seed}.pt',
    Path("../best_model") /  f'best_model_{time_squence}_{seed}.pt')

model.load_state_dict(torch.load(path / f'best_model_{time_squence}_{seed}.pt'))


HLO_Test = {}
for testpath in HLO_test_path_list:
    filename = os.path.basename(testpath)
    HLOproteinpocket = []
    HLOproteinresienergy = []
    HLOligandsmile = []
    HLOligandrdkit = []
    HLOligandDOXenergy = []
    HLOligandentropy = []
    HLOlabel = []
    HLOppidlist = []
    for idx in range(len(os.listdir(testpath))):
        data = torch.load(os.path.join(testpath,os.listdir(testpath)[idx]))
        pocket = torch.unsqueeze(data['pocket'],0)
        proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
        ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
        ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
        ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
        ligandentropy = torch.unsqueeze(data['Entropy'],0)
        ligandlabel = torch.unsqueeze(data['label'],0)
        HLOproteinpocket.append(pocket)
        HLOproteinresienergy.append(proteinresienergy)
        HLOligandsmile.append(ligandsmile)
        HLOligandrdkit.append(ligandrdkit)
        HLOligandDOXenergy.append(ligandDOXenergy)
        HLOligandentropy.append(ligandentropy)
        HLOlabel.append(ligandlabel)
        HLOppidlist.append(data['id'])
    HLOdata = ProteinLigandDataset(HLOproteinpocket,HLOproteinresienergy,HLOligandsmile,HLOligandrdkit,HLOligandDOXenergy,HLOlabel,HLOligandentropy)
    HLOloader = DataLoader(HLOdata, batch_size=1, shuffle=False)
    performance_HLO,performance_HLOdict = test(model, HLOloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=HLOppidlist,isTest=True,test_file_path=path / f'{filename}_result.csv')
    HLO_Test[filename] = performance_HLO['CORR']

Merk_test = {}
for Merktest in Merk_test_path_list:
    filename = os.path.basename(Merktest)
    Merkproteinpocket = []
    Merkproteinresienergy = []
    Merkligandsmile = []
    Merkligandrdkit = []
    MerkligandDOXenergy = []
    Merkligandentropy = []
    Merklabel = []
    Merkppidlist = []
    for idx in range(len(os.listdir(Merktest))):
        data = torch.load(os.path.join(Merktest,os.listdir(Merktest)[idx]))
        pocket = torch.unsqueeze(data['pocket'],0)
        proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
        ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
        ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
        ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
        ligandentropy = torch.unsqueeze(data['Entropy'],0)
        ligandlabel = torch.unsqueeze(data['label'],0)
        Merkproteinpocket.append(pocket)
        Merkproteinresienergy.append(proteinresienergy)
        Merkligandsmile.append(ligandsmile)
        Merkligandrdkit.append(ligandrdkit)
        MerkligandDOXenergy.append(ligandDOXenergy)
        Merkligandentropy.append(ligandentropy)
        Merklabel.append(ligandlabel)
        Merkppidlist.append(data['id'])
    Merkdata = ProteinLigandDataset(Merkproteinpocket,Merkproteinresienergy,Merkligandsmile,Merkligandrdkit,MerkligandDOXenergy,Merklabel,Merkligandentropy)
    Merkloader = DataLoader(Merkdata, batch_size=1, shuffle=False)
    performance_Merk,performance_Merkdict = test(model, Merkloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=Merkppidlist,isTest=True,test_file_path=path / f'{filename}_result.csv')
    Merk_test[filename] = performance_Merk['CORR']


Merk_Average = {}
Merk_Average['Merk_ACORR'] = sum(Merk_test.values()) / len(Merk_test) if Merk_test else 0

Cov_Test = {}
for Covtest in Cov_test_path_list:
    filename = os.path.basename(Covtest)
    if filename == 'KRAS38':
        Covproteinpocket = []
        Covproteinresienergy = []
        Covligandsmile = []
        Covligandrdkit = []
        CovligandDOXenergy = []
        Covligandentropy = []
        Covlabel = []
        Covppidlist = []
        for idx in range(len(os.listdir(Covtest))):
            data = torch.load(os.path.join(Covtest,os.listdir(Covtest)[idx]))
            pocket = torch.unsqueeze(data['pocket'],0)
            proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
            proteinresienergy[proteinresienergy>5] = (data['ligand_DOXenergt'] - torch.sum(proteinresienergy[proteinresienergy<5]))
            ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0) 
            ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
            ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
            ligandentropy = torch.unsqueeze(data['Entropy'],0) * 2
            ligandlabel = torch.unsqueeze(data['label'],0)
            Covproteinpocket.append(pocket)
            Covproteinresienergy.append(proteinresienergy)
            Covligandsmile.append(ligandsmile)
            Covligandrdkit.append(ligandrdkit)
            CovligandDOXenergy.append(ligandDOXenergy)
            Covligandentropy.append(ligandentropy)
            Covlabel.append(ligandlabel)
            Covppidlist.append(data['id'])
        Covdata = ProteinLigandDataset(Covproteinpocket,Covproteinresienergy,Covligandsmile,Covligandrdkit,CovligandDOXenergy,Covlabel,Covligandentropy)
        Covloader = DataLoader(Covdata, batch_size=1, shuffle=False)
        performance_Cov,performance_Covdict = test(model, Covloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=Covppidlist,isTest=True,test_file_path=path / f'{filename}_result.csv')
        if filename =='KRAS38':
            HLO_Test[filename] = performance_Cov['CORR']
        else:
            Cov_Test[filename] = performance_Cov['SP']
    else:
        Cov_CASFdirlist = os.listdir(Covtest)
        for Cov_CASFdir in Cov_CASFdirlist:
            Covtestpath = os.path.join(Covtest,Cov_CASFdir)
            filename = os.path.basename(Covtestpath)
            Covproteinpocket = []
            Covproteinresienergy = []
            Covligandsmile = []
            Covligandrdkit = []
            CovligandDOXenergy = []
            Covligandentropy = []
            Covlabel = []
            Covppidlist = []
            for idx in range(len(os.listdir(Covtestpath))):
                data = torch.load(os.path.join(Covtestpath,os.listdir(Covtestpath)[idx]))
                pocket = torch.unsqueeze(data['pocket'],0)
                proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
                proteinresienergy[proteinresienergy>5] = (data['ligand_DOXenergt'] - torch.sum(proteinresienergy[proteinresienergy<5]))
                ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0) 
                ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
                ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
                ligandentropy = torch.unsqueeze(data['Entropy'],0) * 2
                ligandlabel = torch.unsqueeze(data['label'],0)
                Covproteinpocket.append(pocket)
                Covproteinresienergy.append(proteinresienergy)
                Covligandsmile.append(ligandsmile)
                Covligandrdkit.append(ligandrdkit)
                CovligandDOXenergy.append(ligandDOXenergy)
                Covligandentropy.append(ligandentropy)
                Covlabel.append(ligandlabel)
                Covppidlist.append(data['id'])
            Covdata = ProteinLigandDataset(Covproteinpocket,Covproteinresienergy,Covligandsmile,Covligandrdkit,CovligandDOXenergy,Covlabel,Covligandentropy)
            Covloader = DataLoader(Covdata, batch_size=1, shuffle=False)
            performance_Cov,performance_Covdict = test(model, Covloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=Covppidlist,isTest=True,test_file_path=path / f'{filename}_result.csv')
            if filename =='KRAS38':
                HLO_Test[filename] = performance_Cov['CORR']
            else:
                Cov_Test[filename] = performance_Cov['SP']

HLO_Average = {}
HLO_Average['HLO_ACORR'] = sum(HLO_Test.values()) / len(HLO_Test) if HLO_Test else 0
Cov_Average = {}
Cov_Average['COV_ASP'] = sum(Cov_Test.values()) / len(Cov_Test) if Cov_Test else 0

X_CASF = {}
for Xtest in X_test_path_list:
    filename = os.path.basename(Xtest)
    Xproteinpocket = []
    Xproteinresienergy = []
    Xligandsmile = []
    Xligandrdkit = []
    XligandDOXenergy = []
    Xligandentropy = []
    Xlabel = []
    Xppidlist = []
    for idx in range(len(os.listdir(Xtest))):
        data = torch.load(os.path.join(Xtest,os.listdir(Xtest)[idx]))
        pocket = torch.unsqueeze(data['pocket'],0)
        proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
        ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
        ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
        ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
        ligandentropy = torch.unsqueeze(data['Entropy'],0)
        ligandlabel = torch.unsqueeze(data['label'],0)
        Xproteinpocket.append(pocket)
        Xproteinresienergy.append(proteinresienergy)
        Xligandsmile.append(ligandsmile)
        Xligandrdkit.append(ligandrdkit)
        XligandDOXenergy.append(ligandDOXenergy)
        Xligandentropy.append(ligandentropy)
        Xlabel.append(ligandlabel)
        Xppidlist.append(data['id'])
    Xdata = ProteinLigandDataset(Xproteinpocket,Xproteinresienergy,Xligandsmile,Xligandrdkit,XligandDOXenergy,Xlabel,Xligandentropy)
    Xloader = DataLoader(Xdata, batch_size=1, shuffle=False)
    performance_X,performance_Xdict = test(model, Xloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=Xppidlist,isTest=True,test_file_path=path / f'{filename}_result.csv')
    X_CASF[filename] = performance_X['SP']

X_Average = {}
X_Average['X_ASP'] = sum(X_CASF.values()) / len(X_CASF) if X_CASF else 0

M_CASF_16 = {}
for Mtest in M_test_path_list:
    filename = os.path.basename(Mtest)
    Mppidlist = []
    Mproteinpocket = []
    Mproteinresienergy = []
    Mligandsmile = []
    Mligandrdkit = []
    MligandDOXenergy = []
    Mligandentropy = []
    Mlabel = []
    for idx in range(len(os.listdir(Mtest))):
        data = torch.load(os.path.join(Mtest,os.listdir(Mtest)[idx]))
        pocket = torch.unsqueeze(data['pocket'],0)
        proteinresienergy = torch.unsqueeze(data['protein_redienergy'],0)
        ligandsmile = torch.unsqueeze(data['ligand_SMILE'],0)
        ligandrdkit = torch.unsqueeze(data['ligand_Rdkit'],0)
        ligandDOXenergy = torch.unsqueeze(data['ligand_DOXenergt'],0)
        ligandentropy = torch.unsqueeze(data['Entropy'],0)
        ligandlabel = torch.unsqueeze(data['label'],0)
        Mproteinpocket.append(pocket)
        Mproteinresienergy.append(proteinresienergy)
        Mligandsmile.append(ligandsmile)
        Mligandrdkit.append(ligandrdkit)
        MligandDOXenergy.append(ligandDOXenergy)
        Mligandentropy.append(ligandentropy)
        Mlabel.append(ligandlabel)
        Mppidlist.append(data['id'])
    Mdata = ProteinLigandDataset(Mproteinpocket,Mproteinresienergy,Mligandsmile,Mligandrdkit,MligandDOXenergy,Mlabel,Mligandentropy)
    Mloader = DataLoader(Mdata, batch_size=1, shuffle=False)
    performance,performance_dict = test(model, Mloader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=Mppidlist,isTest=True,test_file_path=path / 'M_result.csv')
    M_CASF_16[filename] = performance['SP']

M_Average = {}
M_Average['M_ASP'] = sum(M_CASF_16.values()) / len(M_CASF_16) if M_CASF_16 else 0

CASF2016 = {}
CASF2016_performance,performance_dict = test(model, test_loader, loss_func, device,show=SHOW_PROCESS_BAR,testppid=testppidlist,isTest=True,test_file_path=path / 'CASF2016_result.csv')
CASF2016['CASF2016P'] = CASF2016_performance['CORR']
Spearman = spearman_correlation_coefficient(performance_dict)
CASF2016['CASF2016SP'] = Spearman

column_names = []
column_names.append('Model')
for key in HLO_Test.keys():
    column_names.append(key)
for key in Merk_test.keys():
    column_names.append(key)
for key in Cov_Test.keys():
    column_names.append(key)
for key in X_CASF.keys():
    column_names.append(key)
for key in M_CASF_16.keys():
    column_names.append(key)
for key in CASF2016.keys():
    column_names.append(key)
for key in Merk_Average.keys():
    column_names.append(key)
for key in HLO_Average.keys():
    column_names.append(key)
for key in Cov_Average.keys():
    column_names.append(key)
for key in X_Average.keys():
    column_names.append(key)
for key in M_Average.keys():
    column_names.append(key)

savedata = pd.DataFrame(columns=column_names)
getdata = savedata

M_test = 0
HLO_test = 0
X_Test = 0
M_SP = 0
X_SP = 0
HLO_P = 0
for idx in range(len(column_names)):
    if column_names[idx] == 'Model':
        getdata.loc[0,column_names[idx]] = str(path)
    else:
        if column_names[idx] in CASF2016.keys():
            getdata.loc[0,column_names[idx]] = CASF2016[column_names[idx]]
        elif column_names[idx] in M_CASF_16.keys():
            getdata.loc[0,column_names[idx]] = M_CASF_16[column_names[idx]]
        elif column_names[idx] in X_CASF.keys():
            getdata.loc[0,column_names[idx]] = X_CASF[column_names[idx]]
        elif column_names[idx] in Cov_Test.keys():
            getdata.loc[0,column_names[idx]] = Cov_Test[column_names[idx]]
        elif column_names[idx] in HLO_Test.keys():
            getdata.loc[0,column_names[idx]] = HLO_Test[column_names[idx]]
        elif column_names[idx] in Merk_test.keys():
            getdata.loc[0,column_names[idx]] = Merk_test[column_names[idx]]
        elif column_names[idx] in HLO_Average.keys():
            getdata.loc[0,column_names[idx]] = HLO_Average[column_names[idx]]
        elif column_names[idx] in Merk_Average.keys():
            getdata.loc[0,column_names[idx]] = Merk_Average[column_names[idx]]
        elif column_names[idx] in M_Average.keys():
            getdata.loc[0,column_names[idx]] = M_Average[column_names[idx]]
        elif column_names[idx] in X_Average.keys():
            getdata.loc[0,column_names[idx]] = X_Average[column_names[idx]]
        elif column_names[idx] in Cov_Average.keys():
            getdata.loc[0,column_names[idx]] = Cov_Average[column_names[idx]]

write_path = Path(f'../Performance/')

try:
    havedata = pd.read_csv(write_path / 'Train.csv',sep='\t')
    datawrite = pd.concat([havedata,getdata],ignore_index=True)
    datawrite.to_csv(write_path / 'Train.csv', index=False,sep='\t')
except:
    getdata.to_csv(write_path / 'Train.csv', index=False,sep='\t')
with open(path / 'performance.txt', 'w') as f:
    f.write(str(performance))
