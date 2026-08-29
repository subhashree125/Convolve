# Convolve - PAN IIT AI-ML Hackathon Projects

This repository contains machine learning solutions developed for the **Convolve PAN IIT AI-ML Hackathon**, conducted by **IDFC Bank**. The projects tackle two real-world financial and marketing challenges using deep learning and advanced data science techniques.

---

## 📋 Projects Overview

### 1. Credit Card Default Prediction

A deep learning-based risk assessment model that predicts the likelihood of credit card default. This project helps financial institutions identify at-risk customers and implement targeted interventions.

### 2. Email Engagement Optimization

A machine learning model that determines the optimal time slots for sending personalized emails. This project optimizes customer engagement by identifying when individual customers are most likely to open emails.

---

## 🏗️ Repository Structure

```text
Convolve/
├── Round 1/
│   ├── Doraemon_code.ipynb      # Credit card default prediction model implementation
│   ├── Doraemon_convolve.pdf    # Round 1 presentation/documentation
│   └── results.csv              # Initial results and model outputs
└── Round 2/
    ├── data_pre-processing1.ipynb  # Primary data cleaning and feature engineering
    ├── data_pre-processing2.ipynb  # Secondary preprocessing pipeline
    ├── training_final.ipynb        # Email timing model training and evaluation
    ├── testing.ipynb               # Model testing and validation
    ├── TEAM_doreamon_sinchan.pdf   # Round 2 final submission/presentation
    └── submission.csv              # Final predictions for evaluation

```

---

## 🛠️ Technology Stack

* **Language**: Python 3.12.7
* **Core Libraries**:
* `pandas` - Data manipulation and analysis
* `numpy` - Numerical computing
* `scikit-learn` - Machine learning algorithms and preprocessing
* `joblib` - Model serialization and persistence
* `matplotlib` & `seaborn` - Data visualization


* **Models**:
* Random Forest Classifiers
* Multi-output Classification
* Deep learning approaches (LSTM for sequential data)



---

## 📊 Key Approaches

### Data Processing

#### Cleaning & Preparation

* Removal of redundant/sparse features (columns with >60% missing values)
* Handling of mixed data types with proper dtype conversion
* Date-based customer record merging using forward and backward fill
* Strategic column elimination for irrelevant features

#### Categorical Variable Standardization

The preprocessing pipeline normalizes categorical variables with multiple representations:

* **Marital Status (`v27`)**:
* Consolidates: `UNMARRIED`, `S`, `Single`, `SINGLE` → `Unmarried`
* Consolidates: `WIDOW` → `Widow`
* Consolidates: `MARRIED`, `M` → `Married`


* **Occupation (`v29`)**:
* Standardizes 30+ occupation variations into meaningful categories
* Examples: `SELFEMPLOYEDBUSINESS`, `SELF EMPLOYED` → `Self Employed Business`
* Groups rare occupations as `"Others"` (threshold: 20 occurrences)


* **Gender (`v54`)**:
* Normalizes: `MALE`, `M` → `Male`
* Normalizes: `FEMALE`, `F` → `Female`
* Groups: `THIRD GENDER`, `U`, `O`, `C` → `OTHERS`


* **Email Providers (`v55`)**:
* Extracts domain from email addresses
* Consolidates Gmail variants (`GAMIL`, `GMAI`, `GMAL`, etc.) → `GMAIL`
* Consolidates Yahoo variants (`YOHOO`, `YHOO`) → `YAHOO`
* Groups rare providers as `"Others"` (threshold: 21 occurrences)



#### Missing Value Imputation

* **Categorical Features**: Mode-based imputation for all object-type columns
* **Numeric Features**: Linear interpolation with bidirectional fill
* **Special Cases**:
* `v30`: Converts to numeric with coercion for mixed types
* `v5`: Replaces `'ZZ'` with `NaN` before imputation
* `v65`: Direct fill with `0.0`
* `v71`, `v73`: Boolean columns filled with `True`



#### Feature Scaling

* **Integer Columns** (`int64`, `int32`): Applied `MinMaxScaler` to scale values to `[0, 1]`, preserving feature relationships while normalizing magnitude.
* **Float Columns** (`float64`): Applied `StandardScaler` to standardize to `mean=0, std=1`, removing outlier influence and improving model convergence.

#### Feature Encoding

* Binary encoding for specific flags (`v60`: `'N'` → `1`, else `0`)
* Boolean columns converted to integer representation
* Label encoding for all remaining categorical variables with `LabelEncoder`

---

### Feature Engineering

* **Customer Demographics Aggregation**: Per-customer feature vectors combining marital status, occupation, gender, and email provider (110 feature dimensions per customer after preprocessing).
* **Email Engagement Pattern Extraction**: Aggregation of customer email history by `CUSTOMER_CODE`, extracting `send_timestamp` and `open_timestamp` sequences to create multi-output binary target vectors.
* **Target Variable Construction**: 28-dimensional binary vector representing daily time slots. Slot activation is `1` if the customer opened an email in that slot, and `0` otherwise.

---

### Model Architecture

#### Email Timing Predictor - Multi-Output Classification

```python
Base Classifier: RandomForestClassifier(
    n_estimators=300,      # 300 decision trees for ensemble robustness
    max_depth=20,          # Controls tree complexity and overfitting
    min_samples_split=5,   # Minimum samples to split internal nodes
    n_jobs=-1,             # Parallel processing across all CPU cores
    random_state=42        # Reproducibility across runs
)

Wrapper: MultiOutputClassifier
- Trains independent classifier for each of 28 time slots
- Enables simultaneous prediction across all slots
- Provides slot-specific probability estimates

```

#### Model Training Pipeline

1. Load customer features (`CDNA.csv`) and email history (`HISTORY.csv`).
2. Aggregate email history and create multi-output targets.
3. Merge customer demographics with engagement targets.
4. Perform Train-Test Split (80-20) with `random_state=42`.
5. Train `MultiOutputClassifier` on training data.
6. Evaluate precision score for each of 28 time slots.
7. Save trained model to `joblib` format.

#### Fine-tuning Mechanism

* Load pre-trained model and apply to new test data.
* Re-train individual estimators on test set features to enable continuous model improvement while keeping structure intact.

#### Model Output & Prediction Format

* **Probability-based Ranking**: Generates prediction probabilities for all 28 time slots, extracts positive class probabilities, and ranks time slots in descending order.
* **Submission Output**:
```csv
customer_code,predicted_slots_order
[hash],['slot_10','slot_27','slot_7',...,'slot_28']

```


*(Output generated for 65,708 customers in Round 2)*

---

## 📈 Results & Performance

### Round 1 - Credit Card Default Prediction

* **Dataset**: Comprehensive customer transaction data with 664+ attributes.
* **Output**: Risk assessment scores saved in `results.csv`.

### Round 2 - Email Timing Optimization

* **Training Data**: 220,699 customers with 110 processed features.
* **Test Data**: 68,450 customers requiring predictions.
* **Sample Precision Scores (per time slot)**:
* **Slot 0**: `98.20%` *(Highest - optimal for morning emails)*
* **Slot 23**: `62.78%` *(Evening peak engagement)*
* **Slot 10**: `61.11%`
* **Slot 21**: `60.37%`
* **Slot 7**: `59.45%`
* **Slot 6**: `58.93%`
* **Slot 12**: `38.46%` *(Lowest - midday lull)*
* **Average Precision**: `~56%` across all slots



---

## 🚀 Quick Start Guide

### Prerequisites

* **Python Version**: `3.12.7`
* **Dependencies**:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn

```



### Directory Setup

Ensure your local data files match the expected structure:

```text
Convolve/
├── Round 2/
│   ├── CDNA.csv                 # Training customer features
│   ├── HISTORY.csv              # Training email history
│   ├── TEST_CDNA.csv            # Test customer features
│   ├── TEST_HISTORY.csv         # Test email history
│   ├── data_pre-processing1.ipynb
│   ├── training_final.ipynb
│   └── testing.ipynb
└── Round 1/
    ├── Doraemon_code.ipynb
    └── Dev_data_to_be_shared.csv

```

### Running the Complete Pipeline

#### Step 1: Data Preprocessing (Round 2)

```bash
jupyter notebook "Round 2/data_pre-processing1.ipynb"

```

* **Input**: `train_cdna_data.csv`
* **Output**: Cleaned and scaled feature matrix
* **Operations**: Column filtering, categorical standardization, missing value imputation, feature scaling, label encoding.

#### Step 2: Model Training

```bash
jupyter notebook "Round 2/training_final.ipynb"

```

* **Input**: Processed customer features + `HISTORY.csv`
* **Output**: Trained model saved as `doraemon.joblib`
* **Operations**: Multi-output target generation, 300-tree Random Forest per slot, test set fine-tuning.

#### Step 3: Test & Predict

```bash
jupyter notebook "Round 2/testing.ipynb"

```

* **Input**: Test customer data (`TEST_CDNA.csv` + `TEST_HISTORY.csv`)
* **Output**: `submission.csv` containing ranked slot predictions derived from `doraemon_sinchan.joblib`.

#### Round 1 Execution

```bash
jupyter notebook "Round 1/Doraemon_code.ipynb"

```

* **Input**: `Dev_data_to_be_shared.csv`
* **Output**: `results.csv` with default risk scores.

---

## 💾 Model Persistence

Models are serialized using the `joblib` binary format:

* **`doraemon.joblib`**: Base trained `RandomForestClassifier` trained on 220,699 customer samples and validated on an 80-20 split.
* **`doraemon_sinchan.joblib`**: Fine-tuned production model incorporating the additional 68,450 test set samples for improved generalization.

```python
import joblib

# Load production model
model = joblib.load('doraemon_sinchan.joblib')

# Predict classes and probabilities
predictions = model.predict(test_features)
probabilities = model.predict_proba(test_features)

```

---

## 🎯 Key Highlights

* **Comprehensive Data Pipeline**: Robust handling of missing data, variable scaling, and category standardization.
* **Multi-Output Strategy**: Simultaneous prediction across 28 distinct hourly time slots.
* **Ensemble Architecture**: 300-tree Random Forest engine ensuring high variance reduction.
* **Continuous Fine-Tuning**: Built-in strategy to adapt model weights as new test/production instances arrive.
* **Scalable**: Handles over 200,000 multi-attribute records efficiently.

---

## 📌 Project Context

* **Hackathon**: Convolve PAN IIT AI-ML Hackathon
* **Organizer**: IDFC Bank
* **Date**: January 2025
* **Team**: Doraemon & Sinchan

---

## 📜 License & Attribution

This project was developed as part of the **Convolve PAN IIT AI-ML Hackathon** initiative organized by **IDFC Bank**.
