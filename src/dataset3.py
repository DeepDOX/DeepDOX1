from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import re
from sklearn.metrics import r2_score
from typing import List
import torch

class ProteinLigandDataset(Dataset):
    def __init__(self, proteinpocket: torch.Tensor, proteinenergy: torch.Tensor, ligandSMILE: torch.Tensor, ligandRdkit: torch.Tensor, ligandEnergy: torch.Tensor, label: torch.Tensor,ligandentropy:torch.Tensor):
        self.protein = proteinpocket    
        self.proenergy = proteinenergy
        self.ligand = ligandSMILE
        self.ligandfinger = ligandRdkit
        self.energy = ligandEnergy
        self.label = label
        self.ligandentropy = ligandentropy

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.protein[idx], self.proenergy[idx], self.ligand[idx], self.energy[idx], self.label[idx], self.ligandfinger[idx], self.ligandentropy[idx]