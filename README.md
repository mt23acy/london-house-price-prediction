# 🏠 London House Price Prediction: Machine Learning vs. Deep Learning



***

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

***

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

***

## 🗄️ Dataset

The project uses an integrated dataset formed by merging two public sources under the
**Open Government Licence v3.0**:

- **HM Land Registry PPD** — Transactional data: sale price, date, property type, and address
- **EPC Data (MHCLG)** — Physical characteristics: floor area, habitable rooms, energy
  efficiency rating, and built form

> **Data Engineering Challenge:** High-cardinality features (postcodes, streets) and memory
> constraints (>32GB RAM) were resolved using Label Encoding and ZRAM compression, enabling
> training on over a decade of London freehold property records.

### 📁 Input Dataset — `london_merged_df_filtered.csv`

The model and visualisation tool require a pre-processed CSV file named
`london_merged_df_filtered.csv` as input. This file is **not included in the repository**
due to its large size, but can be reproduced by following these steps:

1. Download the full **Price Paid Data (PPD)** complete file from
   [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
2. Download the full **EPC dataset** for England and Wales from
   [MHCLG Open Data](https://epc.opendatacommunities.org/)
3. Merge the two datasets by matching property address fields
4. Apply the following filters to produce the input CSV:
   - **Region:** London only (filter by London districts/postcodes)
   - **Property type:** Freehold houses only (exclude flats and leasehold)
   - **Price range:** Remove extreme outliers (min/max price thresholds applied)
   - **Date range:** 2015–2024 transactions only
   - **Records with missing critical fields** (floor area, postcode, price) are dropped
5. Save the resulting filtered dataframe as `london_merged_df_filtered.csv` in the
   root directory of the project

> This filtering process reduces the full national PPD+EPC dataset down to a
> London-specific, model-ready CSV that the training scripts and Gradio tool
> can load directly without memory issues.

***

## 📂 Project Structure

```text
london-house-price-prediction/
├── README.md                        # Project documentation
├── index.html                       # GitHub Pages portfolio site
├── main.py                          # Gradio visualization tool & model logic
├── FPR.pdf                          # Final Project Report (Full Thesis)
├── assets/                          # Screenshots and visualisation images
├── notebooks/                       # Data cleaning & model training scripts
└── london_merged_df_filtered.csv    # Input dataset (generate locally — see Dataset section)
```

***

## 🛠️ Installation & Requirements

Requires **Python 3.9+**

**1. Clone the repository:**
```bash
git clone https://github.com/mt23acy/london-house-price-prediction.git
cd london-house-price-prediction
```

**2. Install dependencies:**
```bash
pip install numpy pandas scikit-learn xgboost tensorflow gradio matplotlib
```

***

## 🚀 How to Run the Gradio Visualization Tool

The interactive tool allows address lookups to view historical sales and predict property
valuations using the optimised XGBoost model.

**1. Prepare the data** — Ensure `london_merged_df_filtered.csv` is in the root directory
(see the Dataset section above for how to generate this file).

**2. Run the script:**
```bash
python main.py
```

**3. Open the UI** — Gradio will provide a local URL (e.g., `http://127.0.0.1:7860`)

**4. Use the tool:**
- Type a **Street Name** to filter the database
- Select **House Number (PAON)** and **Postcode** from dynamic dropdowns
- Click **"Get Prices & Predict Current"** for the historical price trend and AI valuation

***

## 💡 Key Findings

- **XGBoost Superiority** — Achieved R² of 0.83, proving its effectiveness on complex
  structured tabular data
- **Feature Integration Value** — Merging EPC data (floor area, habitable rooms) significantly
  boosted predictive power over PPD alone
- **Deep Learning vs. Ensembles** — LSTM underperformed Gradient Boosting for this tabular
  task, showing sequence modelling is not a silver bullet for real estate without tuning
- **Encoding Strategy** — Label Encoding was the necessary compromise over One-Hot Encoding
  due to London's high-cardinality geographical data
- **Temporal Robustness** — Near-identical Test and 2024 Hold-Out R² scores confirm the
  XGBoost model generalises well to unseen future market data

***

## 🔮 Future Work

- **Macroeconomic Features** — Incorporate interest rates, inflation, and proximity to
  tube stations and local amenities
- **Advanced Encoding** — Target Encoding or CatBoost Encoding to capture geographical
  nuances without artificial ordering
- **Spatial Architectures** — Graph Neural Networks (GNNs) to model relationships between
  neighbouring properties and postcodes
- **Hyperparameter Optimisation** — Bayesian Optimisation for finer tuning of XGBoost
  and MLP architectures

***

## 👤 Author

**Muhammad Faizan Tariq**  
MSc Data Science and Analytics — University of Hertfordshire (2025)


***

## 📜 License

- **Data:** [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
  — Contains HM Land Registry & EPC data
- **Code:** MIT License — See `LICENSE` for details

***

*Muhammad Faizan Tariq | MSc Data Science and Analytics — University of Hertfordshire, 2025*
