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
cd ./src
python Predict.py [model].pt [Datadir] [True/False]
```
## Citation
> Zheng Liu, Hao Sun, Yuliang Wang, Yanliang Ren, Li Rao, Zeyue Huang, Hongxuan Cao, Xiuqi Hu, Xinyue Zhu, Meng Li, Jian Wan; DeepDOX1: A Dual-Drive Framework Integrating Deep Learning and First-Principles Quantum Chemistry for Drug–Protein Affinity Prediction. *JACS Au* 22 June 2026; 6 (6): 3190–3202. https://doi.org/10.1021/jacsau.6c00177

## Contact
For any questions or issues, please contact : 
