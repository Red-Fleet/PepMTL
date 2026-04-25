# PepMTL
PepMTL: Integrated Multi-Task Learning Framework for the Prediction of Multifunctional Bioactive Peptides


**PepMTL** is an integrated Multi-Task Learning framework designed for the high-fidelity prediction of multifunctional bioactive peptides. By utilizing a dual-encoder protein language model (ESM-2), a 1D-CNN local motif extractor, and a Dual Sequence-Level Cross-Attention mechanism, PepMTL effectively captures both broad evolutionary context and highly specific local chemical gradients. 

To ensure realistic generalization and prevent sequence homology leakage, the primary dataset is rigorously processed using a cluster-based splitting strategy (CD-HIT at an 80% identity threshold).

## 📂 Repository Structure

```text
├── benchmark/                        # Benchmarking datasets and evaluation scripts
│   ├── MCMEPP/                       # Files for MCMFPP state-of-the-art comparison
│   │   ├── MCMFPP/                   # MFTP benchmark datasets
│   │   └── MCMFPP.ipynb              # PepMTL evaluation on MFTP dataset
│   ├── ToxMSRC/                      # Files for ToxMSRC toxicity benchmark
│   │   ├── ToxMSRC/                  # Highly imbalanced toxicity datasets
│   │   └── ToxMsrc.ipynb             # 10-fold CV evaluation script
│   └── ToxPre/                       # Files for ToxPre-2L toxicity benchmark
│       ├── ToxPre/                   # Balanced toxicity datasets
│       └── ToxPre.ipynb              # 10-fold CV evaluation script
├── classification_model.ipynb        # Main PepMTL architecture, training, and evaluation pipeline
├── cleaned_clustered_dataset.json    # Primary multi-label dataset (CD-HIT 80% clustered)
├── predict.py                        # Given peptides this file loads the model and predict the peptide classes 
├── test_results.py                   # file to generate test results present in paper
├── model.py                          # contains the model class
├── LICENSE                           # Open-source license
└── README.md                         # This file
```


## 🚀 Usage
All code is provided in easy-to-use Jupyter Notebooks.

1. Main Model (Multi-Label Prediction)
To view the core PepMTL architecture, data loading procedures, and the full multi-task training loop (including Asymmetric Loss, R-Drop, and Stochastic Weight Averaging), open:
classification_model.ipynb

This notebook processes cleaned_clustered_dataset.json, splits it by cluster, and trains the dual-encoder framework to predict the binary "bioactive vs. non-functional" state alongside 12 specific functional categories using our End-to-End Soft Gating mechanism.

2. Benchmarking (Single-Task & Multi-Task SOTA)
To reproduce the benchmarking results presented in the manuscript, navigate to the benchmark/ folder.

Open benchmark/ToxPre/ToxPre.ipynb or benchmark/ToxMSRC/ToxMsrc.ipynb to run the 10-fold cross-validation toxicity evaluations.

Open benchmark/MCMEPP/MCMFPP.ipynb to run the paper-style multi-label evaluation against the MCMFPP framework.

3. Predicting New Peptides
To predict the functional classes of new peptide sequences, use the `predict.py` script. This script loads the pre-trained model and evaluates the given peptides.
```bash
python predict.py
```

4. Generating Test Results
To reproduce the exact test set metrics and evaluate the model's performance as presented in the paper, run:
```bash
python test_results.py
```

💾 Pre-trained Weights
Pre-trained model weights for the full PepMTL architecture have been made publicly available. You can download the .pt files 
🔗 https://drive.google.com/drive/folders/16eGJxFZI1x3kYpHA7byTRqxfa0NIJFsJ?usp=sharing


📊 Data Availability
The primary dataset (cleaned_clustered_dataset.json) is included in this repository. This file contains the peptide sequences and their corresponding functional classes, pre-clustered using CD-HIT at an 80% identity threshold to eliminate sequence homology leakage between training and testing partitions.

The benchmark datasets are sourced from their respective original publications and included in the benchmark/ subdirectories for ease of reproducibility.

zenodo: 🔗https://doi.org/10.5281/zenodo.19749666
