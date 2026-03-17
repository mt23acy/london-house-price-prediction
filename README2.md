# 🏠 London House Price Prediction: Machine Learning vs. Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-blueviolet?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF8C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 📖 Overview

Predicting house prices accurately in dynamic markets like London poses significant challenges for traditional modelling techniques due to the non-linearity of influencing factors. This project investigates the extent to which machine learning (ML) and deep learning (DL) models improve short-term prediction accuracy for freehold houses in London. By merging HM Land Registry Price Paid Data (PPD) with Energy Performance Certificate (EPC) datasets, the study overcomes the feature deficiency of transactional records alone. Five models including Linear Regression, Random Forest, XGBoost, Multi-Layer Perceptron (MLP), and Long Short-Term Memory (LSTM) were evaluated on a 2024 hold-out set to test temporal generalisation and real-world applicability.

The work follows a complete data-science lifecycle: large-scale data acquisition from open government sources, iterative data cleaning and feature engineering, comparative model training, rigorous evaluation using standard regression metrics, and deployment of an interactive **Gradio-based address lookup tool** (implemented in `main.py`) to make the results accessible to end users, as detailed in the final project report (`FPR.pdf`).

---

## 📊 Model Performance Results

Models were trained on historical data (2015–2023) and validated against a final **2024 Hold-Out Set** to test robustness against temporal drift.

| Model | Test R² | 2024 Hold-Out R² | Status |
|---|---|---|---|
| **XGBoost Regressor** | **0.82** | **0.83** | 🏆 Best Model |
| Random Forest | 0.78 | 0.77 | ✅ Strong |
| MLP (Neural Network) | 0.75 | 0.74 | ⚡ Moderate |
| LSTM (Deep Learning) | 0.63 | 0.64 | ⚠️ Weak |
| Linear Regression | 0.55 | 0.59 | 📏 Baseline |

XGBoost clearly outperforms both traditional linear models and the tested deep learning architectures on this tabular, structured data, offering the best balance of accuracy and computational efficiency for short-term London house price prediction.

---

## 🗄️ Dataset

The project uses an integrated dataset formed by merging two public sources under the **Open Government Licence v3.0**:

- **HM Land Registry PPD** — Transactional data: sale price, date, property type, and address.  
- **EPC Data (MHCLG)** — Physical characteristics: floor area, habitable rooms, energy efficiency rating, and built form.

> **Data Engineering Challenge:** High-cardinality features (postcodes, streets) and memory constraints (>32GB RAM) were resolved using Label Encoding and ZRAM compression, enabling training on over a decade of London freehold property records.

### 📁 Model-Ready Datasets (Zipped CSVs)

The core model-ready datasets are provided in compressed form in this repository:

- `london_merged_df_filtered_2015.zip`  
  Contains the **10-year filtered dataset (2015–2024)** for London freehold houses. This is the main dataset used for model training, validation, and the 2024 hold-out evaluation.

- `london_merged_df_filtered.zip` plus split parts `london_merged_df_filtered.z01` and `london_merged_df_filtered.z02`  
  Together these form the **full merged London dataset** (broader time range) before restricting to the 10-year analysis window. This can be used for extended experiments or alternative time-window choices.

Because GitHub imposes a file size limit on direct uploads, the large CSVs have been compressed and, for the full dataset, **split into parts**.

To reconstruct locally:

1. Download `london_merged_df_filtered_2015.zip` and extract it to obtain `london_merged_df_filtered_2015.csv`.  
2. Download `london_merged_df_filtered.zip`, `london_merged_df_filtered.z01`, and `london_merged_df_filtered.z02` into the same folder.  
3. Use a tool that supports split archives (7-Zip, WinRAR, or macOS Archive Utility) to extract `london_merged_df_filtered.zip`. This will reconstruct `london_merged_df_filtered.csv`.

These CSVs are the inputs used by the notebooks and the Gradio app in `main.py`.

### 🔁 Reproducing the Filtered CSVs from Raw Data

If you prefer to regenerate the CSVs from raw PPD and EPC data rather than using the ready-made archives:

1. Download the full **Price Paid Data (PPD)** complete file from the official HM Land Registry source.  
2. Download the full **EPC dataset** for England and Wales from the official open data portal.  
3. Merge the two datasets by matching property address fields.  
4. Apply the following filters:  
   - **Region:** London only (filter by London districts/postcodes).  
   - **Property type:** Freehold houses only (exclude flats and leasehold).  
   - **Price range:** Remove extreme outliers (min/max price thresholds).  
   - **Date range:** 2015–2024 only for the 10-year dataset.  
   - Drop records with missing critical fields (e.g. floor area, postcode, price).  
5. Save the resulting filtered dataframes as:  
   - `london_merged_df_filtered_2015.csv` (2015–2024 subset).  
   - `london_merged_df_filtered.csv` (full merged London subset).

---

## 📂 Project Structure

```text
london-house-price-prediction/
├── README.md                          # Project documentation (this file)
├── index.html                         # GitHub Pages portfolio site
├── FPR.pdf                            # Final Project Report (full MSc thesis)
├── FPR_Code.txt                       # Full pipeline script (reference implementation)
├── main.py                            # Gradio address-lookup app + XGBoost model serving
├── LICENSE                            # MIT License for the code
├── notebooks/                         # Jupyter notebooks for data & model work
│   ├── label_london_2024p-merge-data-visualisation-final.ipynb
│   │   # Label encoding experiments + early visualisation
│   └── label_enc_london_2024p-merge-data_final2.ipynb
│       # Final label-encoding pipeline, model training & evaluation
├── london_merged_df_filtered_2015.zip # 10-year (2015–2024) filtered dataset (compressed)
├── london_merged_df_filtered.zip      # Full merged London dataset (compressed, split)
├── london_merged_df_filtered.z01      # Split archive part 1 for full dataset
└── london_merged_df_filtered.z02      # Split archive part 2 for full dataset
```

`main.py` loads `london_merged_df_filtered.csv`, performs preprocessing and label encoding, trains an XGBoost regressor (if needed), and exposes a Gradio interface for interactive address-based queries using the trained model.

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

## 📥 Preparing the Data Locally

**Option A – Use the ready-made zipped datasets (recommended)**

1. Extract `london_merged_df_filtered_2015.zip` to get:
   - `london_merged_df_filtered_2015.csv`
2. Ensure the following three files are in the same folder:
   - `london_merged_df_filtered.zip`
   - `london_merged_df_filtered.z01`
   - `london_merged_df_filtered.z02`
3. Open **`london_merged_df_filtered.zip`** with an archive tool that supports split archives (7‑Zip, WinRAR, macOS Archive Utility). This will reconstruct:
   - `london_merged_df_filtered.csv`
4. Move both CSV files into the project root (the same folder as `main.py` and `README.md`):
   - `london_merged_df_filtered_2015.csv`
   - `london_merged_df_filtered.csv`

**Option B – Regenerate from raw PPD + EPC**

If you prefer to recreate the CSVs from scratch:

1. Download the full **Price Paid Data (PPD)** complete file (England & Wales) from the official HM Land Registry source.  
2. Download the full **EPC dataset** for England and Wales from the official open data portal.  
3. Merge the two datasets on address fields (e.g. postcode, PAON, street, or official address keys).  
4. Filter the merged data:
   - Keep **London** records only (by region / postcode).
   - Keep **freehold houses only** (exclude flats and leasehold).
   - Apply **price range limits** to remove extreme outliers.
   - Restrict the **analysis window to 2015–2024** for the 10‑year dataset.
   - Drop rows with missing critical fields (price, floor area, postcode, etc.).
5. Save the resulting dataframes as:
   - `london_merged_df_filtered_2015.csv` — 2015–2024 subset.
   - `london_merged_df_filtered.csv` — full merged London subset.

---

## 🚀 How to Run the Gradio Visualization Tool (`main.py`)

The interactive Gradio app in `main.py` exposes the trained XGBoost model as an address‑level valuation tool.

**1. Ensure data is available**

- Confirm that `london_merged_df_filtered.csv` is present in the project root.

**2. Run the app from the repo root:**
```bash
python main.py
```

On startup, `main.py` will:

- Load `london_merged_df_filtered.csv` from the project root.
- Drop unused or helper columns (address keys and technical EPC fields that are not needed for prediction).
- Apply basic cleaning and **price range filtering** to remove extreme outliers.
- Engineer date-related features such as:
  - `year`
  - `month`
  - `day`
  - `weekday` (0–6)
  - `quarter` (1–4)
- Apply **label encoding** to all categorical features, including:
  - `postcode`
  - `PAON` / `SAON`
  - `street`
  - other object-type columns used as model inputs.
- Separate **2024 data** as a hold-out evaluation set using a target-year variable.
- Train an **XGBoost regressor** on the pre-2024 data (if a trained model is not already available), using:
  - Features: all engineered numeric and encoded categorical columns
  - Target: `price`
- Fit a `StandardScaler` on the training features to standardise them before modelling.
- Store:
  - The trained XGBoost model
  - The fitted scaler
  - The list and order of training features
  - Category mappings and default values  
  so that new predictions are consistent with the training setup.

3. Once the script has finished initialisation, Gradio will start and print a local URL such as:

```text
http://127.0.0.1:7860

Open this URL in your browser.

---
```
**4. Use the interface**

In the browser UI:

- Select a **Street** from the dropdown (populated from the cleaned dataset).
- Select the corresponding **House Number (PAON)** and **Postcode**.
- Click **“Get Prices & Predict Current”** to:
  - View historical sale prices for that property.
  - Get the current predicted price from the trained XGBoost model.

---

## 📓 Working with the Notebooks

To explore the full pipeline in notebooks:

1. Start Jupyter from the project root:
   ```bash
   jupyter notebook
2. Open:
- `notebooks/label_london_2024p-merge-data-visualisation-final.ipynb`  
  for label encoding experiments and early visualisations.
- `notebooks/label_enc_london_2024p-merge-data_final2.ipynb`  
  for the final label-encoding pipeline, model training, evaluation, and plots.

## 💡 Key Findings

- **XGBoost Superiority** — Achieved R² of 0.83, proving its effectiveness on complex structured tabular data and outperforming both traditional regression and tested deep learning models for this task.
- **Feature Integration Value** — Merging EPC data (floor area, habitable rooms, energy rating) with PPD significantly boosted predictive power over PPD alone.
- **Deep Learning vs. Ensembles** — The LSTM model underperformed tree-based ensembles (RF, XGBoost) on this tabular dataset, indicating that sequence models are not automatically superior for real estate forecasting without dedicated architecture and tuning.
- **Encoding Strategy** — Label Encoding was a necessary and effective compromise over One-Hot Encoding in the presence of extremely high-cardinality location features in London.
- **Temporal Robustness** — The small gap between test and 2024 hold-out R² scores shows that the XGBoost model generalises well to unseen future data, within the studied time window.

## 🔮 Future Work

- **Macroeconomic Features** — Incorporate interest rates, inflation, policy changes, and local economic indicators to capture broader market dynamics.
- **Advanced Encoding** — Explore Target Encoding or CatBoost-style encodings to better capture location effects without imposing artificial ordering.
- **Spatial Architectures** — Investigate Graph Neural Networks (GNNs) or spatial models to explicitly model relationships between neighbouring properties and postcodes.
- **Hyperparameter Optimisation** — Apply Bayesian optimisation or similar techniques for more systematic tuning of XGBoost, RF, and MLP architectures.
- **Uncertainty Quantification** — Extend the models to provide prediction intervals or uncertainty estimates rather than point estimates only.

## 👤 Author

**Muhammad Faizan Tariq**  
MSc Data Science and Analytics — University of Hertfordshire (2025)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-faizan-tariq/)

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github&logoColor=white)](https://github.com/mt23acy)

[![Email](https://img.shields.io/badge/Email-faizantariq44444%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:faizantariq44444@gmail.com)


## 📜 License

- **Data:** Open Government Licence v3.0 — Contains HM Land Registry & EPC data.
- **Code:** MIT License — See `LICENSE` for details.

*Muhammad Faizan Tariq | MSc Data Science and Analytics — University of Hertfordshire, 2025.*
