COVDOX280 and BEDOX21 are first-principles computational methods developed by our group for predicting protein–ligand binding affinities. Built upon the XO method from Prof. Xin Xu's team, these tools are capable of predicting binding poses and estimating binding energies for both covalent and non-covalent complexes.

📦 Software Dependencies
To run COVDOX280/BEDOX21, the following software packages are required:

AutoDock

SYBYL

MOPAC2016

XO (from Prof. Xin Xu’s group)

Users are responsible for obtaining and installing these packages, including compliance with any relevant licensing terms.

⚙️ Usage Instructions
For Non‑covalent Binding Prediction
Required input files:

Protein structure file (e.g., .pdb)

Ligand structure file (e.g., .mol2)

DSSP‑derived protein secondary structure file

Parameter configuration file (DOX_config)

Run command:

bash
COVDOX280 <protein.pdb> <ligand.mol2>
For Covalent Binding Prediction
Required input files (the four files above, plus):
5. Pre‑reactive ligand structure file (must be named ligand.mol2)

Run command: Same as for non‑covalent prediction.
```
#DOX_config Example 
CovDOX = True/False  # Whether to use covalent DOX restraints
Initial_Stage = 0    # Initial stage number
Final_Stage = 3      # Final stage number
Num_Core = 64        # Number of CPU cores to use
Lig_Cterm = 33       # Ligand Covalent C Atom number
Lig_Sterm = 31       # Ligand Covalent S Atom number
Pointer_ChainID = A   # Pointer Chain ID
Pointer_Resname = UNK  # Pointer Residue Name
Pointer_Resnum = 1    # Pointer Residue Number
COV_ChainID = A     # Covalent Chain ID
COV_Resname = CYS   # Covalent Residue Name
COV_Resnum = 96    # Covalent Residue Number
Fast_DOX = True   # Whether to use fast docking mode(FastDOX will caculate 5 poses with XO method)
```
