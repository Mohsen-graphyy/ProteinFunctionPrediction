# ProteinFunctionPrediction
Protein function prediction using ESM2 protein language model embeddings and a JAX/Flax multilabel classifier trained on Gene Ontology molecular function annotations.
Protein function prediction using ESM2 protein language model embeddings and a JAX/Flax multilabel classifier trained on Gene Ontology molecular function annotations.

## Overview

This project builds an end-to-end protein function prediction pipeline using:

- ESM2 (Evolutionary Scale Modeling) protein language model embeddings
- Gene Ontology (GO) molecular function annotations
- JAX/Flax neural network training framework
- Human protein sequences (Taxonomy ID: 9606)
- Multi-label classification for predicting protein functions

The workflow demonstrates how pretrained protein foundation models can be leveraged to generate biologically meaningful sequence representations and predict molecular functions directly from protein sequences.

---

## Project Pipeline

```text
Protein Sequences (FASTA)
            │
            ▼
Data Cleaning & Filtering
            │
            ▼
Gene Ontology Annotation Processing
            │
            ▼
ESM2 Embedding Extraction
            │
            ▼
Train / Validation / Test Split
            │
            ▼
Multi-Label Neural Network
            │
            ▼
Function Prediction
            │
            ▼
Evaluation (AUROC, AUPRC, F1)
```

---

## Dataset

The notebook uses:

- Protein amino acid sequences
- Gene Ontology (GO) annotations
- GO term descriptions
- Taxonomy metadata

### Filtering

The workflow focuses on:

- Human proteins (`taxonomyID = 9606`)
- Molecular Function Ontology (MFO) annotations
- Multi-label protein function prediction

---

## Model Architecture

### Feature Extraction

Protein embeddings are generated using:

- **ESM2**
  - Model: `facebook/esm2_t30_150M_UR50D`
  - Transformer-based protein language model
  - Mean-pooled sequence embeddings

### Prediction Model

A neural network classifier is trained on top of ESM2 embeddings to predict multiple Gene Ontology molecular function labels simultaneously.

---

## Evaluation Metrics

The notebook evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC
- AUPRC (Area Under Precision-Recall Curve)

### Baseline Comparisons

- Random (coin-flip) predictor
- Proportional label frequency predictor
- Learned neural network model

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- BioPython
- Transformers (Hugging Face)
- ESM2
- JAX
- Flax
- Optax
- Matplotlib
- Seaborn
- Gene Ontology (GO)

---

## Results

The notebook compares learned predictions against simple baselines and identifies the molecular functions that are most accurately predicted by the model.

### Visualizations Include

- Function-level performance analysis
- Top-performing GO terms
- Validation metrics across prediction methods

---

## Future Improvements

- Fine-tune ESM2 on protein function prediction tasks
- Extend prediction to Biological Process and Cellular Component ontologies
- Experiment with larger protein language models
- Hyperparameter optimization
- Improved handling of class imbalance
- Cross-species generalization

---
