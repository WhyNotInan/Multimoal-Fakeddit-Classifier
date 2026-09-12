# Explainable Fake News Detection Using CLIP-Based Text-Image Consistency

This repository contains the code and resources for a lightweight, multimodal fake news detection pipeline. Instead of relying on computationally expensive Optical Character Recognition (OCR) or unimodal fine-tuning, this architecture leverages pre-trained Contrastive Language-Image Pretraining (CLIP) embeddings and an explicitly engineered cosine similarity metric to classify deceptive content.

## Architecture & Pipeline

* **Feature Extraction:** Processes raw headlines and images through OpenAI’s frozen CLIP model (`vit-base-patch32`), generating mathematically aligned 512-dimensional text and image vectors.
* **Semantic Fusion:** Computes a 1D cosine similarity score between the text and image embeddings to explicitly measure multimodal agreement. This is concatenated with the base vectors to form a **1025-dimensional feature tensor**.
* **Hyperparameter Tuning:** A Genetic Algorithm was executed on a 5,000-sample subset to dynamically evolve the optimal network topology (layer sizing, dropout rates) without exhausting computational resources.
* **Classification:** A custom Multilayer Perceptron (MLP) structured as `1025-512-128-6`, utilizing Batch Normalization and Dropout layers (0.4 and 0.1) to combat overfitting.

## Dataset

Trained and evaluated on a cleaned multimodal subset of the **Fakeddit** benchmark dataset, consisting of **490,038 posts**. The model classifies content into six distinct categories:
* True
* Satire/Parody
* Misleading Content
* Manipulated Content
* False Connection
* Completely Fabricated

## Performance Metrics

The model was rigorously evaluated using **5-Fold Stratified Cross-Validation** to ensure statistical stability across the entire dataset.

| Metric | Score |
| :--- | :--- |
| **Mean CV Accuracy** | 92.01% (± 0.08%) |
| **Mean Macro F1** | 0.8779 (± 0.0020) |
| **Best Fold (Fold 5) Accuracy** | 92.12% |
| **Best Fold (Fold 5) Macro-AUC** | 0.9911 |

## Ablation Study

To isolate and quantify the contribution of each modality and our engineered fusion strategy, a comprehensive ablation study was conducted.

| Configuration | Mean Accuracy |
| :--- | :--- |
| Text Only | 79.77% |
| Image Only | 84.06% |
| Text + Image (No Cosine) | 91.50% |
| **Baseline (Multimodal Full)** | **91.97%** |

## Getting Started

* **`train_kfold.py`**: The primary training script. Executes the 5-fold cross-validation across the full `fakeddit_features_full.pt` dataset using the AdamW optimizer. Generates and saves high-resolution ROC-AUC, Confusion Matrix, and Learning Curve plots directly to the `/plots` directory.
* **`ablation.py`**: Executes the independent modality testing to output the comparative baseline metrics.

## Future Scope

* **Explainable Inference:** Transitioning from rigid 6-way classification to probabilistic reasoning, utilizing attention maps to highlight the specific semantic mismatches driving the model's predictions.
* **Bias Mitigation:** Investigating the impact of inherited upstream biases baked into the foundational CLIP training corpus and implementing debiasing techniques within the MLP layers to improve cross-cultural accuracy.
