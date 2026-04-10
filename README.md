# 🛒 ANN Retail Sales Forecasting & Markdown Impact Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Deployed-yellow?logo=amazonaws)](http://13.234.117.37:8501)

> A deep learning project that forecasts department-wide weekly retail sales across 45 stores using a PyTorch ANN, analyzes markdown impact during holiday weeks, and deploys the solution as a Streamlit dashboard on AWS EC2.

## 🌐 Live Demo
**Public URL:** [http://13.234.117.37:8501](http://13.234.117.37:8501)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [AWS Deployment](#aws-deployment)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)

---

## 🎯 Problem Statement

Develop a predictive ANN model to:
- Forecast **department-wide weekly sales** for each of 45 retail stores
- Analyze the **impact of markdowns** on sales during holiday weeks
- Provide **actionable insights** to optimize markdown strategies and inventory management

---

## 📁 Project Structure

```
retail_project/
├── data/
│   ├── sales_data_set.csv          # Raw sales data (421K+ records)
│   ├── Features_data_set.csv       # Store features & markdowns
│   ├── stores_data_set.csv         # Store type & size info
│   ├── cleaned_data.csv            # After Step 1
│   └── featured_data.csv           # After Step 3
├── models/
│   ├── best_ann_model.pt           # Trained PyTorch model
│   ├── scaler_X.pkl                # Feature scaler
│   └── scaler_y.pkl                # Target scaler
├── outputs/
│   ├── eda/                        # EDA charts (7 PNG files)
│   └── evaluation/
│       └── test_predictions.csv    # Model predictions on test set
├── pipeline.py                     # Steps 1-4: Clean → EDA → Features → Train
├── app.py                          # Streamlit dashboard
└── requirements.txt
```

---

## 📊 Dataset

| Source | 45 retail stores — historical sales data |
|---|---|
| Period | February 5, 2010 — November 1, 2012 |
| Total Records | 421,570 weekly sales entries |
| Format | 3 CSV files |

### Data Tables

| Table | Key Columns |
|---|---|
| **Stores** | Store, Type (A/B/C), Size |
| **Features** | Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday |
| **Sales** | Store, Dept, Date, Weekly_Sales, IsHoliday |

---

## 🔬 Approach

```
Step 1: Data Cleaning & Preprocessing
    Merge 3 CSVs, fix dates, fill NAs, remove negative sales
Step 2: Exploratory Data Analysis
    7 visualizations: distributions, trends, holiday impact, correlations
Step 3: Feature Engineering
    39 total features: lags, rolling stats, cyclical encoding, interactions
Step 4: ANN Training (PyTorch)
    4-layer network, HuberLoss, AdamW, early stopping
Step 5: Streamlit Dashboard
    3 tabs: EDA, Prediction, Batch Results
Step 6: AWS EC2 Deployment
    Public URL, nohup background process
```

---

## 🧠 Model Architecture

```
Input (39 features)
        ↓
Linear(512) → BatchNorm1d → ReLU → Dropout(0.3)
        ↓
Linear(256) → BatchNorm1d → ReLU → Dropout(0.3)
        ↓
Linear(128) → BatchNorm1d → ReLU → Dropout(0.2)
        ↓
Linear(64)  → BatchNorm1d → ReLU → Dropout(0.1)
        ↓
Linear(1)  →  Weekly Sales Prediction ($)
```

| Hyperparameter | Value |
|---|---|
| Loss Function | HuberLoss (delta=1.0) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Early Stopping | Patience = 10 epochs |
| Batch Size | 512 |
| Max Epochs | 80 |
| Train/Test Split | Time-based — before/after July 1, 2012 |

---

## ⚙️ Feature Engineering

| Category | Features | Count |
|---|---|---|
| Calendar | Week, Month, Quarter, Year, DayOfYear | 5 |
| Cyclical Encoding | Week_sin/cos, Month_sin/cos | 4 |
| Lag Features | Sales_Lag_1, 2, 4, 8, 13, 26, 52 | 7 |
| Rolling Mean | RollMean_4, 8, 13 weeks | 3 |
| Rolling Std | RollStd_4, 8, 13 weeks | 3 |
| Markdown | TotalMarkdown, AnyMarkdown, Markdown x Holiday | 3 |
| Store | Type, Size, SizeBucket | 3 |
| External | Temperature, Fuel_Price, CPI, Unemployment | 4 |
| Raw Markdowns | MarkDown1-5 | 5 |
| Base | Store, Dept, IsHoliday | 3 |
| **Total** | | **39** |

---

## 📈 Results

| Metric | Overall | Holiday Weeks | Non-Holiday Weeks |
|---|---|---|---|
| MAE | ~$1,800 | ~$2,200 | ~$1,600 |
| RMSE | ~$2,900 | ~$3,800 | ~$2,600 |
| R2 | ~0.97 | ~0.95 | ~0.98 |

> Model explains ~97% of variance in weekly sales.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/retail-sales-forecasting.git
cd retail-sales-forecasting

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Place CSV files in data/ folder
mkdir data
# Copy the 3 CSV files into data/
```

---

## 💻 Usage

### Run Full Pipeline

```bash
python pipeline.py
```

This runs all 4 steps sequentially:
1. Data cleaning → `data/cleaned_data.csv`
2. EDA → `outputs/eda/` (7 charts)
3. Feature engineering → `data/featured_data.csv`
4. ANN training → `models/best_ann_model.pt`

### Launch Streamlit Dashboard

```bash
streamlit run app.py
# OR on Windows:
py -m streamlit run app.py
```

Open: http://localhost:8501

---

## ☁️ AWS Deployment

Deployed on **AWS EC2 t3.micro** — Asia Pacific (Mumbai) region.

```bash
# Launch with nohup (stays alive after SSH closes)
nohup streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > streamlit.log 2>&1 &
```

- Security Group: Port 8501 open to 0.0.0.0/0
- Live URL: http://13.234.117.37:8501

---

## 💡 Key Insights

1. Holiday weeks generate ~30% higher average sales than regular weeks
2. Type A stores produce 3x more revenue than Type C stores
3. MarkDown1 has the strongest positive correlation with weekly sales
4. Departments 92 and 95 are consistently top revenue contributors
5. Lag features (1-week, 4-week) are the strongest predictors in the ANN
6. Applying markdowns 2 weeks before holidays maximizes sales uplift
7. MarkDown2 and MarkDown4 have minimal impact — budget better spent on MarkDown1 and MarkDown3

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.2 |
| Data Processing | Pandas, NumPy |
| ML Utilities | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Cloud | AWS EC2 (t3.micro, Ubuntu 22.04) |
| Model Persistence | Joblib |

---

## 🏫 About

**Domain:** Retail Analytics
**Institution:** GUVI | HCL
**Project:** ANN Predictive Modeling of Retail Sales and Markdown Impact

