# 🏠 London House Price Prediction: Machine Learning vs. Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-blueviolet?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 📖 Overview

Predicting house prices accurately in dynamic markets like London poses significant challenges
for traditional modelling techniques due to the non-linearity of influencing factors. This
project investigates the extent to which machine learning (ML) and deep learning (DL) models
improve short-term prediction accuracy for freehold houses in London. By merging HM Land
Registry Price Paid Data (PPD) with Energy Performance Certificate (EPC) datasets, the study
overcomes the feature deficiency of transactional records alone. Five models — Linear
Regression, Random Forest, XGBoost, Multi-Layer Perceptron (MLP), and Long Short-Term Memory
(LSTM) — were evaluated on a 2024 hold-out set to test temporal generalisation and
real-world applicability.

---

## 📊 Model Performance Results

Models were trained on historical data (2015–2023) and validated against a final
**2024 Hold-Out Set** to test robustness against temporal drift.

| Model | Test R² | 2024 Hold-Out R² | Status |
|---|---|---|---|
| **XGBoost Regressor** | **0.82** | **0.83** | 🏆 Best Model |
| Random Forest | 0.78 | 0.77 | ✅ Strong |
| MLP (Neural Network) | 0.75 | 0.74 | ⚡ Moderate |
| LSTM (Deep Learning) | 0.63 | 0.64 | ⚠️ Weak |
| Linear Regression | 0.55 | 0.59 | 📏 Baseline |

---

## 🗄️ Dataset

The project uses an integrated dataset formed by merging two public sources under the
**Open Government Licence v3.0**:

- **HM Land Registry PPD** — Transactional data: sale price, date, property type, and address
- **EPC Data (MHCLG)** — Physical characteristics: floor area, habitable rooms, energy
  efficiency rating, and built form

> **Data Engineering Challenge:** High-cardinality features (postcodes, streets) and memory
> constraints (>32GB RAM) were resolved using Label Encoding and ZRAM compression, enabling
> training on over a decade of London freehold property records.

---

## 📂 Project Structure

```text
london-house-price-prediction/
├── README.md              # Project documentation
├── index.html             # GitHub Pages portfolio site
├── main.py                # Gradio visualization tool & model logic
├── FPR.pdf                # Final Project Report (Full Thesis)
├── assets/                # Screenshots and visualisation images
└── notebooks/             # Data cleaning & model training scripts
