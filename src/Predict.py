import os
import torch as th
import sys
from torch.utils.data import DataLoader
from DeepDOX1 import DeepDOXA,test
from dataset4 import ProteinLigandDataset
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
import pandas as pd
from tqdm.auto import tqdm


class CaculateFinger:
    def __init__(self,file):
        m = Chem.MolFromMol2File(file)
        if m is None:
            print('Fail to convert mol2 file: ' + file)
            raise RuntimeError("Fail to convert mol2 file: " + file)
        self.Wtcal = Descriptors.MolWt(m)
        self.Logp = Descriptors.MolLogP(m)
        self.numHBD = Descriptors.NumHDonors(m)
        self.numHBA = Descriptors.NumHAcceptors(m)
        self.QED = QED.qed(m)
        self.TBSA = Descriptors.TPSA(m)
        self.NOCount = float(Lipinski.NOCount(m))
        self.CSP3 = float(Lipinski.FractionCSP3(m))
        self.Rotatable = float(Lipinski.NumRotatableBonds(m))
        self.NumCarbocycle = float(Lipinski.NumAliphaticRings(m))
        self.NumSaring = float(Lipinski.NumSaturatedRings(m))
        self.Numring = float(Lipinski.RingCount(m))

    def getall(self):
        return {
            'MolWt':self.Wtcal,
            'LogP':self.Logp,
            'NumHBD':self.numHBD,
            'NumHBA':self.numHBA,
            'QED':self.QED,
            'TPSA':self.TBSA,
            'NOCount':self.NOCount,
            'CSP3':self.CSP3,
            'Rotatable':self.Rotatable,
            'NumCarbocycle':self.NumCarbocycle,
            'NumSaring':self.NumSaring,
            'Numring':self.Numring
        }

class Predict:
    def __init__(self, model_path, test_data_path,Cov = False):
        self.model = DeepDOXA()
        self.Cov = Cov
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.load_state_dict(th.load(model_path, map_location=device))
        self.model.to(device)
        self.test_data_path = test_data_path

    def load_test_data(self):
        test_path = self.test_data_path
        protein_pocket = []
        protein_resienergy = []
        ligand_Atomic = []
        ligand_DOXenergy = []
        ligand_rdkit = []
        entropy_data = []
        DOX_energy = None
        entropy = None
        P_pocket = None
        P_resienergy = None
        L_SMILE = None
        IsSkip = False

        for file in os.listdir(test_path):
            if file == 'DOX.log':        
                with open(os.path.join(test_path, file), 'r') as f:
                    lines = f.readlines()
                    for idy in range(len(lines)):
                        if lines[idy].startswith('   Binding Enthalpy          ='):
                            DOX_lines = lines[idy].split()
                            DOX_energy = DOX_lines[-2]
                            if DOX_energy == '********' or DOX_energy == 0.00 or DOX_energy == '0.00' or DOX_energy == '0.0':
                                IsSkip = True
                        if lines[idy].startswith(' Ligand Entropy Contribution ='):
                            entropy_lines = lines[idy].split()
                            entropy = entropy_lines[-2]
                        if lines[idy].startswith('Interaction Descriptor: Begin'):
                            pocket_lines_begin = idy + 1
                            pocket_lines_end = idy + 101
                            P_pocket = lines[pocket_lines_begin:pocket_lines_end]
                            P_pocket = [i.replace(',',' ') for i in P_pocket]
                            ligandS_lines_begin = idy + 101
                            ligandS_lines_end = idy + 251
                            L_SMILE = lines[ligandS_lines_begin:ligandS_lines_end]
                            L_SMILE = [i.replace(',',' ') for i in L_SMILE]
                        if lines[idy].startswith('Binding Affinity Decomposition:'):
                            P_resienergy_begin = idy + 1
                        if lines[idy].startswith('Interaction Descriptor: End'):
                            P_resienergy_end = idy
                            P_resienergy = lines[P_resienergy_begin:P_resienergy_end]
                            for idk in range(len(P_resienergy)):
                                P_resienergy[idk] = P_resienergy[idk].split()[0]
                            P_resienergy = [i.replace(',',' ') for i in P_resienergy]
                    if IsSkip:
                        print('Skipping due to invalid DOX energy.')
                    else:
                        protein_pocket.append(P_pocket)
                        protein_resienergy.append(P_resienergy)
                        ligand_Atomic.append(L_SMILE)
                        ligand_DOXenergy.append(DOX_energy)
                        entropy_data.append(entropy)
                        protein_pocket_T = th.tensor([])

                        for i in range(len(protein_pocket)):
                            protein_procket_tori = str(protein_pocket[i]).replace('[','').replace(']','')
                            protein_procket_torilist = protein_procket_tori.split(',')
                            protein_procket_toripd = pd.DataFrame()
                            for idx in range(len(protein_procket_torilist)):
                                protein_pocket_data = protein_procket_torilist[idx].replace('\'','')
                                protein_pocket_data = protein_pocket_data.replace('\\n','')
                                protein_pocket_data = pd.DataFrame([float(x) for x in protein_pocket_data.split()])
                                protein_pocket_data = protein_pocket_data.T
                                protein_procket_toripd = pd.concat([protein_procket_toripd,protein_pocket_data],axis=0)
                        protein_procket_toripd = th.tensor(protein_procket_toripd.values)
                        protein_pocket_T = th.cat((protein_pocket_T,protein_procket_toripd),0)

                        protein_resienergy_T = th.tensor([])
                        protein_resienergy_torpd = pd.DataFrame()
                        protein_resienergy_ori = str(protein_resienergy[i])
                        protein_resienergy_ori = protein_resienergy_ori.replace('[','').replace(']','')
                        protein_resienergy_orilist = protein_resienergy_ori.split(',') 
                        for idx in range(len(protein_resienergy_orilist)):
                            protein_resienergy_data = protein_resienergy_orilist[idx].replace('\'','')
                            protein_resienergy_data = protein_resienergy_data.replace('\\n','')
                            protein_resienergy_data = pd.DataFrame([float(x) for x in protein_resienergy_data.split()])
                            protein_resienergy_data = protein_resienergy_data.T
                            protein_resienergy_torpd = pd.concat([protein_resienergy_torpd,protein_resienergy_data],axis=0)
                        protein_resienergy_torpd = th.tensor(protein_resienergy_torpd.values)
                        if protein_resienergy_torpd.shape[0] < 100:
                            protein_resienergy_torpd = th.cat((protein_resienergy_torpd,th.zeros(100-protein_resienergy_torpd.shape[0],protein_resienergy_torpd.shape[1])),0)
                        protein_resienergy_T = th.cat((protein_resienergy_T,protein_resienergy_torpd),0)

                        ligand_SMILE_T = th.tensor([])
                        ligand_SMILE_torpd = pd.DataFrame()
                        ligand_SMILE_ori = str(ligand_Atomic[i])
                        ligand_SMILE_ori = ligand_SMILE_ori.replace('[','').replace(']','')
                        ligand_SMILE_orilist = ligand_SMILE_ori.split(',')
                        for idx in range(len(ligand_SMILE_orilist)):
                            ligand_SMILE_data = ligand_SMILE_orilist[idx].replace('\'','')
                            ligand_SMILE_data = ligand_SMILE_data.replace('\\n','')
                            ligand_SMILE_data = pd.DataFrame([float(x) for x in ligand_SMILE_data.split()])
                            ligand_SMILE_data = ligand_SMILE_data.T
                            ligand_SMILE_torpd = pd.concat([ligand_SMILE_torpd,ligand_SMILE_data],axis=0)
                        ligand_SMILE_torpd = th.tensor(ligand_SMILE_torpd.values)
                        ligand_SMILE_T = th.cat((ligand_SMILE_T,ligand_SMILE_torpd),0)

                        ligand_DOXenergy_T = th.tensor([])
                        ligand_DOXenergy_ori = ligand_DOXenergy[i]
                        ligand_DOXenergy_torpd = pd.DataFrame([float(ligand_DOXenergy_ori)])
                        ligand_DOXenergy_torpd = th.tensor(ligand_DOXenergy_torpd.values)
                        ligand_DOXenergy_T = th.cat((ligand_DOXenergy_T,ligand_DOXenergy_torpd),0)

                        Entropy_data_T = th.tensor([])
                        Entropy_data_ori = entropy_data[i]
                        Entropy_data_torpd = pd.DataFrame([float(Entropy_data_ori)])
                        Entropy_data_torpd = th.tensor(Entropy_data_torpd.values)
                        Entropy_data_T = th.cat((Entropy_data_T,Entropy_data_torpd),0)
                        pocket = protein_pocket_T
                        Atomic = ligand_SMILE_T
                        DOXenergy = ligand_DOXenergy_T                                
                        entropy = Entropy_data_T
                        residue_energy = protein_resienergy_T
            if file == 'ligand.mol2':
                rdkit = th.tensor([])
                mol2file = os.path.join(test_path, file)
                calc_finger = CaculateFinger(mol2file)
                rdkit_features = calc_finger.getall()
                rdkit_features_list = [rdkit_features['MolWt'], rdkit_features['LogP'], rdkit_features['NumHBD'], rdkit_features['NumHBA'], rdkit_features['QED'], rdkit_features['TPSA'], rdkit_features['NOCount'], rdkit_features['CSP3'], rdkit_features['Rotatable'], rdkit_features['NumCarbocycle'], rdkit_features['NumSaring'], rdkit_features['Numring']]
                rdkit_features_T = th.tensor(rdkit_features_list)
                rdkit = th.cat((rdkit, rdkit_features_T),0)
        test_pocket = th.tensor([])
        test_Atomic = th.tensor([])
        test_rdkit = th.tensor([])
        test_residue_energy = th.tensor([])
        test_DOXenergy = th.tensor([])
        test_entropy = th.tensor([])
        if self.Cov == True:
            test_pocket = th.cat((test_pocket, th.unsqueeze(pocket,0)),0)
            test_Atomic = th.cat((test_Atomic, th.unsqueeze(Atomic,0)),0)
            test_rdkit = th.cat((test_rdkit, th.unsqueeze(rdkit,0)),0)
            test_residue = th.unsqueeze(residue_energy,0)
            test_residue[test_residue>5] = (test_DOXenergy- th.sum(test_residue[test_residue<5]))
            test_residue_energy = th.cat((test_residue_energy, test_residue),0)
            test_DOXenergy = th.cat((test_DOXenergy, th.unsqueeze(DOXenergy,0)),0)
            test_entropy = th.cat((test_entropy, th.unsqueeze(entropy,0)),0)
        else:
            test_pocket = th.cat((test_pocket, th.unsqueeze(pocket,0)),0)
            test_Atomic = th.cat((test_Atomic, th.unsqueeze(Atomic,0)),0)
            test_rdkit = th.cat((test_rdkit, th.unsqueeze(rdkit,0)),0)
            test_residue_energy = th.cat((test_residue_energy, th.unsqueeze(residue_energy,0)),0)
            test_DOXenergy = th.cat((test_DOXenergy, th.unsqueeze(DOXenergy,0)),0)
            test_entropy = th.cat((test_entropy, th.unsqueeze(entropy,0)*2),0)
        return test_pocket, test_Atomic, test_residue_energy, test_DOXenergy, test_rdkit, test_entropy
    
    def predict(self):
        protein,ligand,residueenergy,DOXenergy,rdkitfeat,ligandentropy = self.load_test_data()
        model = self.model
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        model.eval()
        protein = protein.to(device).to(th.float32)
        ligand = ligand.to(device).to(th.float32)
        residueenergy = residueenergy.to(device).to(th.float32)
        DOXenergy = DOXenergy.to(device).to(th.float32)
        rdkitfeat = rdkitfeat.to(device).to(th.float32)
        ligandentropy = ligandentropy.to(device).to(th.float32)
        predict = model(ligand,DOXenergy,protein,residueenergy,rdkitfeat,ligandentropy)
        print('DeepDOX1 Result: {}'.format(predict.item()))
    
if __name__ == "__main__":
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    Cov = sys.argv[3]
    predictor = Predict(model_path, test_data_path)
    predictor.predict()