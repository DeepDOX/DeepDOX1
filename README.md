

## Introduction
This repository contains the implementation of the DeepDOX1 model for predicting protein-ligand binding affinities. The model leverages deep learning techniques to analyze molecular structures and predict their interactions.

## Requirements
- Python 3.8+
- cuda 12.4
- PyTorch 2.4.1+cu124
- RDKit
- NumPy
- Pandas

The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment.yml
conda activate DeepDOX
```

## Training & Testing
To train the DeepDOX1 model, run the following command:
```bash
cd ./src
python Train.py
```
To test the DeepDOX1 model, run the following command:
```bash
cd ./src
python Test_all.py
```
To Use the DeepDOX1 model, run the following command:
```bash
cd ./Example
python Predict.py
```
## Citation


## Contact
For any questions or issues, please contact : 
