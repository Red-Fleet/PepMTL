# PepMTL: A Debiased Multi-Task Learning Framework and Web Server for Multi-Functional Bioactive Peptide Screening

**PepMTL** is a homology-aware, multi-representation multi-task deep learning framework designed for the robust screening and multi-label profiling of bioactive peptides. It addresses fundamental dataset artifacts—specifically sequence homology data leakage, length-distribution bias, and uncurated negative background sampling.

The framework combines a dual-encoder ESM-2 tower (trainable domain-adapted encoder + frozen evolutionary anchor) with a 1D-CNN local motif extractor, integrated via dual cross-attention and orthogonal attention pooling into a hierarchical Task Query Decoder.

---

## 📂 Repository Structure

```text
├── ablation/                                    # Systematic ablation studies
│   ├── cdhit/                                   # Homology-reduction stringency ablations
│   │   ├── cd_hit_0.9.json                      # Metrics for CD-HIT 90% threshold
│   │   ├── cd_hit.ipynb                         # CD-HIT clustering pipeline
│   │   ├── cd_hit_no.json                       # Metrics without CD-HIT (random split)
│   │   ├── classification_cdhit_0.9.ipynb       # Model training with 90% sequence identity
│   │   └── classification_cdhit_no.ipynb        # Model training on random split (no clustering)
│   ├── length_bias/                             # Length-bias audits & background evaluation
│   │   └── length_bias.ipynb                    # Evaluation across length-stratified sequences
│   ├── model_analyse/                           # Feature disentanglement & latent space analysis
│   │   ├── analyse.ipynb                        # PCA visualizations & cosine similarity analysis
│   │   └── metric_analyse.ipynb                 # Read training .json files and convert result to table
│   └── submodel/                                # Architectural component ablations
│       ├── cnn_removed.ipynb                    # Training without 1D-CNN branch
│       ├── cnn_removed.json                     # Evaluation metrics (CNN ablated)
│       ├── esm_removed.ipynb                    # Training without frozen ESM-2 anchor
│       └── esm_removed.json                     # Evaluation metrics (ESM anchor ablated)
├── benchmark/                                   # External & state-of-the-art comparative benchmarks
│   ├── MCMFPP.zip                               # Baseline MCMFPP re-training files & weights
│   └── PEPMTL_MFTP/                             # External MFTP multi-functional benchmark
│       ├── MCMFPP/                              # MFTP benchmark train/test partitions
│       │   ├── test.txt
│       │   └── train.txt
│       └── PEPMTL_MFTP_dataset.ipynb            # PepMTL evaluation on 21-class MFTP benchmark
├── cleaned_clustered_dataset.json               # Primary multi-label dataset (CD-HIT 80% clustered)
├── train_classification.csv                     # Training split sequences & multi-label annotations
├── validation_classification.csv                # Validation split sequences & multi-label annotations
├── test_classification.csv                      # Independent test split sequences & annotations
├── val_non_func_predict_subseqs_after_cd_hit.txt  # Length-matched pseudo-negative validation set
├── test_non_func_predict_subseqs_after_cd_hit.txt # Length-matched pseudo-negative test set
├── model.py                                     # PepMTL PyTorch neural network architecture
├── pepmtl_training.ipynb                        # Main two-phase training curriculum
├── phase_1_model.pt                             # Pre-trained weights (Phase 1 Baseline)
├── phase_1_model.json                           # Configuration & hyperparameters (Phase 1)
├── phase_2_model.pt                             # Pre-trained weights (Phase 2 Length-Debiased)
├── phase_2_model.json                           # Configuration & hyperparameters (Phase 2)
├── predict.py                                   # Inference script for custom peptide sequences
├── test_results.py                              # Independent test-set evaluation script
├── server/                                      # Web server deployment module
│   └── server.ipynb                             # PyTorch web server backend pipeline
├── LICENSE                                      # Open-source license (MIT)
└── README.md                                    # Repository documentation

```

---

## ⚡ Key Features

* **Homology-Aware Splitting:** Cluster-based partitioning via CD-HIT at an 80% sequence identity threshold ensures zero sequence overlap between training, validation, and test splits.


* **Two-Phase Length Debiasing:** A curriculum training strategy that introduces length-matched pseudo-negative substrings to eliminate length-distribution classification shortcuts.


* **Multi-Representation Dual-Encoder:** Integrates a fine-tuned ESM-2 (8M) encoder with a frozen ESM-2 (8M) evolutionary anchor via sequence-level cross-attention and a 1D-CNN motif extractor.


* **Hierarchical Classification:** A binary gatekeeper informs a Task Query Decoder to profile 13 distinct therapeutic functional categories (e.g., Anti-Bacterial, Anti-Viral, Anti-Cancer, Signal Peptide, Metabolic).


* **Interpretability:** Built-in gradient saliency mapping, CNN sequence logos, and cross-attention map extraction for lead optimization.



---

## 🚀 Getting Started

### Prerequisites

Clone the repository and install the dependencies:

```bash
git clone https://github.com/neurocare-iiitd/PepMTL.git
cd PepMTL
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Peptide Bioactivity & Function Prediction

To screen and profile new peptide sequences, run `predict.py`:

```bash
python predict.py

```


### 2. Reproduce Manuscript Test Results

To evaluate the pre-trained Phase 1 and Phase 2 models on the independent, homology-reduced test set:

```bash
python test_results.py

```

### 3. Model Training & Two-Phase Curriculum

The full end-to-end training pipeline is contained in `pepmtl_training.ipynb`:

* **Phase 1:** Trains the multi-task architecture on the CD-HIT 80% clustered dataset with boolean-masked negative sequences.


* **Phase 2:** Extracts length-matched pseudo-negative substrings from background proteins, filters them through the Phase 1 model and MMseqs2, and retrains the network without masking.



### 4. Ablation Studies & Benchmarking

* **Homology Ablation:** Run `ablation/cdhit/classification_cdhit_no.ipynb` or `classification_cdhit_0.9.ipynb` to evaluate the effect of sequence clustering.


* **Submodel Ablation:** Run `ablation/submodel/cnn_removed.ipynb` and `ablation/submodel/esm_removed.ipynb` to assess individual representation streams.


* **Length-Bias Audit:** Execute `ablation/length_bias/length_bias.ipynb` to reproduce length-stratified background evaluations against MCMFPP and Macrel.


* **External MFTP Benchmark:** Execute `benchmark/PEPMTL_MFTP/PEPMTL_MFTP_dataset.ipynb` for the 21-class MFTP benchmark comparison.



---

## 🌐 Web Server

An interactive web server offering bioactivity screening, multi-label classification, and residue-level interpretability (saliency maps, CNN motifs, and cross-attention maps) is available at:

👉 **[https://neurocare.iiitd.edu.in/pepmtl]

---

## 📊 Data & Model Checkpoints

* Pre-trained model weights (`phase_1_model.pt` and `phase_2_model.pt`) and configuration files are available directly in this repository.
* Zenodo: **[https://doi.org/10.5281/zenodo.19749666](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.19749666)**.




