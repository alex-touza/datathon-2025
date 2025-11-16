# GTM (Go-To-Market) – Machine Learning Explainability  
*Datathon 2025 – Schneider Electric*

This repository contains our solution for the **“Explainability on Classifier Models”** challenge from the **2025 Datathon** organized with **Schneider Electric**.  
The goal is to build a **binary classifier** that predicts whether a sales opportunity will be **won (1)** or **lost (0)** and, more importantly, to **explain why** the model makes each prediction.

---

## 1. Problem Overview

Schneider Electric sells complex energy management and industrial automation solutions, generating millions of CRM records on past opportunities.  
The challenge is to:

1. **Train a classification model** to predict the opportunity outcome.
2. **Reach at least 0.70 F1-score** on the target variable.
3. **Focus on explainability** (global + local explanations) so that **non-technical business users** can understand and trust the model’s decisions.

### Dataset (high level)

Each row is an opportunity with:

- `id` – unique identifier (not used as a feature).
- `target_variable` – binary target: `1 = won`, `0 = lost`.
- Features describing:
  - **Past sales with the customer**  
    - `product_A_sold_in_the_past`, `product_B_sold_in_the_past`
    - `Product_A_recommended`
  - **Current opportunity**  
    - `product_A`, `product_C`, `product_D` (amounts in the current opportunity)
  - **Customer relationship**  
    - `cust_hitrate` (success rate), `cust_interactions`, `cust_contracts`
  - **Context**  
    - `opp_month`, `opp_old`
  - **Competition**  
    - `competitor_X`, `competitor_Y`, `competitor_Z`

The dataset is already **preprocessed** and **ready to train** (no heavy feature engineering required).
More detailed explanation about each paremeter can be found in `data/statement.pdf`

---

## 2. Repository Structure

```text
datathon-2025/
├── data/                # Dataset(s) provided for the Datathon (train / test / samples)
├── graphs/              # Plots and explainability visualizations (with SHAP)
├── models/              # Code to train and execute the binary classification models
├── catboost_info/       # CatBoost training logs/metadata
├── rootForest.pkl       # Example trained tree-based model used in our experiments
├── .ipynb_checkpoints/  # Jupyter auto-saved checkpoints
├── .idea/, .vscode/     # IDE / editor config
└── *.ipynb              # Jupyter notebooks with the full workflow (training + explainability)
```

---

## 3. Methodology

### 3.1 Modeling

1. **Train / validation split**  
   - Stratified split on `target_variable`.
   - F1-score used as main metric, with accuracy and ROC-AUC as secondary metrics.

2. **Baseline models**
   - Simple models for reference (e.g. Logistic Regression / Decision Tree).
   - Help understand whether more complex models are justified.

3. **Final model**
   - Tree-based model(s) (e.g. Random Forest / Gradient Boosting / CatBoost).
   - Tuned via cross-validation to reach **≥ 0.70 F1-score** (Datathon requirement).

4. **Model selection**
   - Choose the model with the best validation F1-score and stability.
   - Persist the chosen model to the `models/` folder (and optionally `rootForest.pkl`).

### 3.2 Explainability

The heart of this project is **explainable ML**. We focus on both **global** and **local** explanations:

- **Global insights**
  - **Feature importance** from the model.
  - **SHAP summary plots** to see which features most strongly drive **wins vs losses**.

- **Local insights**
  - **SHAP force / waterfall plots** for individual opportunities:
    - Highlight why a specific deal was predicted as *won* or *lost*.
    - Show which features “pushed” the prediction up or down.
  - Optionally, **LIME** to cross-check local explanations with another method.
---

## 4. How to Run

### 4.1 Prerequisites

- Python 3.10+  

### 4.2 Clone and open the project

```bash
git clone https://github.com/alex-touza/datathon-2025.git
cd datathon-2025
```

Open the notebooks with your preferred tool:

- **VS Code** (Python + Jupyter extensions), or  
- **Jupyter Lab / Notebook**:

```bash
jupyter lab
```

## 5. References

- Schneider Electric Datathon 2025 problem statement (GTM Machine Learning Explainability).

