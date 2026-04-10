import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# --- 1. Model Architecture (must match training) ---
class SalesANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# --- 2. Load Pre-trained Model and Scalers ---
@st.cache_resource
def load_model():
    model_path = "models/best_ann_model.pt"
    input_dim = 40 # Based on FEATURE_COLS length during training
    model = SalesANN(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_scalers():
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    return scaler_X, scaler_y

model = load_model()
scaler_X, scaler_y = load_scalers()

# --- 3. Feature Columns (must match training) ---
FEATURE_COLS = [
    "Store", "Dept", "Type", "Size", "SizeBucket",
    "IsHoliday", "Week", "Month", "Quarter", "Year", "DayOfYear",
    "Week_sin", "Week_cos", "Month_sin", "Month_cos",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "TotalMarkdown", "AnyMarkdown", "Markdown_x_Holiday",
    "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_4", "Sales_Lag_8",
    "Sales_Lag_13", "Sales_Lag_26", "Sales_Lag_52",
    "Sales_RollMean_4", "Sales_RollMean_8", "Sales_RollMean_13",
    "Sales_RollStd_4", "Sales_RollStd_8", "Sales_RollStd_13",
]

# --- 4. Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Retail Sales Forecast")
st.title("🛒 Retail Sales Forecasting with ANN")

st.markdown("Enter the features below to predict weekly sales for a specific store and department.")

# --- Input Features ---
with st.sidebar:
    st.header("Input Features")

    store = st.slider("Store ID", 1, 45, 1)
    dept = st.slider("Department ID", 1, 99, 1)
    date_input = st.date_input("Date", value=pd.to_datetime("2012-11-02")) # Example date after train split

    is_holiday = st.checkbox("Is Holiday?", False)
    temperature = st.slider("Temperature (°F)", -10.0, 100.0, 45.0)
    fuel_price = st.slider("Fuel Price ($)", 2.0, 5.0, 3.0)
    cpi = st.slider("CPI (Consumer Price Index)", 100.0, 230.0, 170.0)
    unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0)

    st.subheader("Markdown Information")
    markdown1 = st.number_input("MarkDown1", value=0.0)
    markdown2 = st.number_input("MarkDown2", value=0.0)
    markdown3 = st.number_input("MarkDown3", value=0.0)
    markdown4 = st.number_input("MarkDown4", value=0.0)
    markdown5 = st.number_input("MarkDown5", value=0.0)

    # Placeholder for Size, Type, and historical lags/rolling features
    st.subheader("Historical Data (Simulated for Demo)")
    sales_lag_1 = st.number_input("Sales Lag 1 (Previous Week)", value=20000.0)
    sales_lag_2 = st.number_input("Sales Lag 2", value=19000.0)
    sales_lag_4 = st.number_input("Sales Lag 4", value=18000.0)
    sales_lag_8 = st.number_input("Sales Lag 8", value=17000.0)
    sales_lag_13 = st.number_input("Sales Lag 13", value=16000.0)
    sales_lag_26 = st.number_input("Sales Lag 26", value=15000.0)
    sales_lag_52 = st.number_input("Sales Lag 52", value=21000.0)
    sales_roll_mean_4 = st.number_input("Sales Roll Mean 4", value=19500.0)
    sales_roll_std_4 = st.number_input("Sales Roll Std 4", value=500.0)
    sales_roll_mean_8 = st.number_input("Sales Roll Mean 8", value=18500.0)
    sales_roll_std_8 = st.number_input("Sales Roll Std 8", value=600.0)
    sales_roll_mean_13 = st.number_input("Sales Roll Mean 13", value=17500.0)
    sales_roll_std_13 = st.number_input("Sales Roll Std 13", value=700.0)

# Simulate other necessary features that were not part of direct input
# In a real app, these would come from a database or a more complex feature generation pipeline
# For simplicity, we'll use dummy values or derive them.

# For Type and Size, we need to load the original store data or pre-compute them.
# For this demo, let's assume we have a way to get them or use fixed values.

# Load original store data to get Type and Size
@st.cache_data
def get_store_info(store_id):
    # This assumes stores_data_set.csv is available. In a real app, you'd load this more robustly.
    # Try multiple possible locations
    stores_path = None
    for p in ["data/stores_data_set.csv", "stores_data_set.csv", "../data/stores_data_set.csv"]:
        if os.path.exists(p):
            stores_path = p
            break
    if stores_path is None:
        st.warning("stores_data_set.csv not found. Using default store values.")
        return 1, 151315
    stores_df = pd.read_csv(stores_path)
    store_info = stores_df[stores_df["Store"] == store_id]
    if not store_info.empty:
        store_type = {"A": 0, "B": 1, "C": 2}[store_info["Type"].iloc[0]]
        store_size = store_info["Size"].iloc[0]
        return store_type, store_size
    return None, None

store_type_encoded, store_size = get_store_info(store)

if store_type_encoded is None or store_size is None:
    st.warning(f"Store ID {store} not found in store data. Using default Type=1 (B) and Size=100000.")
    store_type_encoded = 1 # Default to B
    store_size = 100000 # Default size

# Derive Date features
pd_date = pd.to_datetime(date_input)
week = int(pd_date.isocalendar().week)
month = pd_date.month
quarter = pd_date.quarter
year = pd_date.year
day_of_year = pd_date.dayofyear
week_sin = np.sin(2 * np.pi * week / 52)
week_cos = np.cos(2 * np.pi * week / 52)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

total_markdown = markdown1 + markdown2 + markdown3 + markdown4 + markdown5
any_markdown = 1 if total_markdown > 0 else 0
markdown_x_holiday = total_markdown * (1 if is_holiday else 0)

# Re-create SizeBucket logic based on original data distribution or fixed bins
# For a demo, a simplified approach:
if store_size <= 93638: # Roughly 25th percentile of original data
    size_bucket = 0
elif store_size <= 140167: # Roughly 50th percentile
    size_bucket = 1
else:
    size_bucket = 2

# Create a dictionary with all features
input_data = {
    "Store": store,
    "Dept": dept,
    "Type": store_type_encoded,
    "Size": store_size,
    "SizeBucket": size_bucket,
    "IsHoliday": 1 if is_holiday else 0,
    "Week": week,
    "Month": month,
    "Quarter": quarter,
    "Year": year,
    "DayOfYear": day_of_year,
    "Week_sin": week_sin,
    "Week_cos": week_cos,
    "Month_sin": month_sin,
    "Month_cos": month_cos,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
    "MarkDown1": markdown1,
    "MarkDown2": markdown2,
    "MarkDown3": markdown3,
    "MarkDown4": markdown4,
    "MarkDown5": markdown5,
    "TotalMarkdown": total_markdown,
    "AnyMarkdown": any_markdown,
    "Markdown_x_Holiday": markdown_x_holiday,
    "Sales_Lag_1": sales_lag_1,
    "Sales_Lag_2": sales_lag_2,
    "Sales_Lag_4": sales_lag_4,
    "Sales_Lag_8": sales_lag_8,
    "Sales_Lag_13": sales_lag_13,
    "Sales_Lag_26": sales_lag_26,
    "Sales_Lag_52": sales_lag_52,
    "Sales_RollMean_4": sales_roll_mean_4,
    "Sales_RollMean_8": sales_roll_mean_8,
    "Sales_RollMean_13": sales_roll_mean_13,
    "Sales_RollStd_4": sales_roll_std_4,
    "Sales_RollStd_8": sales_roll_std_8,
    "Sales_RollStd_13": sales_roll_std_13,
}

# Ensure the order of columns matches FEATURE_COLS
input_df = pd.DataFrame([input_data], columns=FEATURE_COLS)

# --- Prediction ---
if st.button("Predict Weekly Sales"):
    try:
        # Scale input features
        X_scaled = scaler_X.transform(input_df.values.astype(np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(X_tensor).item()

        # Inverse transform the prediction
        predicted_sales = scaler_y.inverse_transform(np.array(prediction_scaled).reshape(-1, 1))[0][0]

        st.subheader("Predicted Weekly Sales:")
        st.success(f"Estimated Weekly Sales: ${predicted_sales:,.2f}")

        st.markdown("--- Other Inputs for Context ---")
        st.json(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
---
**Note**: This is a simplified demo. In a production system, historical data for lags and rolling averages would be dynamically fetched and calculated for the given Store and Department.
""")

st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="🛒",
    layout="wide",
)

class SalesANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

FEATURE_COLS = [
    "Store", "Dept", "Type", "Size", "SizeBucket",
    "IsHoliday", "Week", "Month", "Quarter", "Year", "DayOfYear",
    "Week_sin", "Week_cos", "Month_sin", "Month_cos",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "TotalMarkdown", "AnyMarkdown", "Markdown_x_Holiday",
    "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_4", "Sales_Lag_8",
    "Sales_Lag_13", "Sales_Lag_26", "Sales_Lag_52",
    "Sales_RollMean_4", "Sales_RollMean_8", "Sales_RollMean_13",
    "Sales_RollStd_4",  "Sales_RollStd_8",  "Sales_RollStd_13",
]

@st.cache_resource
def load_artifacts():
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    model = SalesANN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(
        torch.load("models/best_ann_model.pt", map_location="cpu")
    )
    model.eval()
    return model, scaler_X, scaler_y

@st.cache_data
def load_predictions():
    return pd.read_csv("outputs/evaluation/test_predictions.csv", parse_dates=["Date"])

st.title("🛒 Retail Sales Forecasting Dashboard")
st.markdown("**ANN-powered predictions using PyTorch * 45 Stores * 81 Departments**")

tab1, tab2, tab3 = st.tabs(["📊 EDA & Insights", "🔮 Single Prediction", "📈 Batch Results"])

# TAB 1 - EDA & Insights
with tab1:
    st.header("Exploratory Data Analysis")

    eda_dir = "outputs/eda"
    eda_images = {
        "Sales Distribution":               "01_sales_distribution.png",
        "Sales Over Time":                  "02_sales_over_time.png",
        "Holiday vs Non-Holiday":           "03_holiday_vs_nonholiday.png",
        "Sales by Store Type":              "04_sales_by_store_type.png",
        "Top 10 Departments":               "05_top10_departments.png",
        "Correlation Heatmap":              "06_correlation_heatmap.png",
        "Markdown vs Sales (Holiday)":      "07_markdown_vs_sales_holiday.png",
    }

    cols = st.columns(2)
    for i, (title, fname) in enumerate(eda_images.items()):
        path = os.path.join(eda_dir, fname)
        if os.path.exists(path):
            with cols[i % 2]:
                st.subheader(title)
                st.image(path)

    st.divider()
    st.subheader("💡 Key Insights")
    st.markdown("""
    - **Holiday weeks** generate significantly higher sales; markdowns amplify this effect.
    - **Type A stores** (largest) have the highest average weekly sales.
    - **Departments 92 & 95** consistently rank in the top revenue contributors.
    - **MarkDown1** has the strongest positive correlation with sales among markdown types.
    - **CPI and Unemployment** show mild inverse relationships with sales.
    - **Lag features** (especially 1-week and 4-week lags) are strong predictors.
    """)

# TAB 2 - Single Store/Dept Prediction
with tab2:
    st.header("Single Week Sales Prediction")

    try:
        model, scaler_X, scaler_y = load_artifacts()

        col1, col2, col3 = st.columns(3)
        with col1:
            store      = st.number_input("Store #", 1, 45, 1)
            dept       = st.number_input("Dept #",  1, 99, 1)
            store_type = st.selectbox("Store Type", [0, 1, 2],
                                      format_func=lambda x: ["A", "B", "C"][x])
            store_size = st.number_input("Store Size (sq ft)", 20000, 250000, 151315)
        with col2:
            is_holiday = st.selectbox("Is Holiday?", [0, 1],
                                      format_func=lambda x: ["No", "Yes"][x])
            pred_date  = st.date_input("Prediction Date")
            temperature = st.slider("Temperature (°F)", 0.0, 110.0, 60.0)
            fuel_price  = st.slider("Fuel Price ($)", 2.0, 5.0, 3.5)
        with col3:
            cpi          = st.number_input("CPI", 100.0, 250.0, 211.0)
            unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0)
            md1 = st.number_input("MarkDown1 ($)", 0.0, 100000.0, 0.0)
            md2 = st.number_input("MarkDown2 ($)", 0.0, 50000.0,  0.0)

        st.markdown("*(Lag / rolling features default to a typical value for demo purposes)*")

        if st.button("🔮 Predict Sales", type="primary"):
            import datetime
            d = pd.Timestamp(pred_date)
            week    = d.isocalendar()[1]
            month   = d.month
            quarter = d.quarter
            year    = d.year
            doy     = d.dayofyear
            total_md  = md1 + md2
            any_md    = int(total_md > 0)
            md_x_hol  = total_md * is_holiday
            size_bkt  = 0 if store_size < 100000 else (1 if store_size < 175000 else 2)

            # Use median lag values (typical store behaviour as defaults)
            typical_lag = 20000.0

            row = [
                store, dept, store_type, store_size, size_bkt,
                is_holiday, week, month, quarter, year, doy,
                np.sin(2*np.pi*week/52), np.cos(2*np.pi*week/52),
                np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                temperature, fuel_price, cpi, unemployment,
                md1, md2, 0.0, 0.0, 0.0,
                total_md, any_md, md_x_hol,
                *(typical_lag,) * 7,   # 7 lag features
                *(typical_lag,) * 3,   # 3 rolling means
                *(0.0,)        * 3,    # 3 rolling stds
            ]

            X_input = scaler_X.transform(np.array([row], dtype=np.float32))
            with torch.no_grad():
                pred_scaled = model(torch.tensor(X_input, dtype=torch.float32))
            pred = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]

            st.success(f"### 💰 Predicted Weekly Sales: **${pred:,.2f}**")

            context = "Holiday Week 🎉" if is_holiday else "Regular Week"
            st.info(f"Week: {week}  |  {context}  |  Store Type: {['A','B','C'][store_type]}")

    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please run **step4_ann_training.py** first.")

# TAB 3 - Batch Evaluation Results
with tab3:
    st.header("Model Evaluation on Test Set")

    try:
        results = load_predictions()
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_true = results["Weekly_Sales"]
        y_pred = results["Predicted_Sales"]

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"${mae:,.0f}")
        m2.metric("RMSE", f"${rmse:,.0f}")
        m3.metric("R²",   f"{r2:.4f}")

        st.divider()

        # Holiday breakdown
        st.subheader("Holiday vs Non-Holiday Performance")
        hol  = results[results["IsHoliday"] == 1]
        nhol = results[results["IsHoliday"] == 0]

        breakdown = pd.DataFrame({
            "Segment":        ["Holiday", "Non-Holiday"],
            "MAE ($)":        [mean_absolute_error(hol["Weekly_Sales"], hol["Predicted_Sales"]),
                               mean_absolute_error(nhol["Weekly_Sales"], nhol["Predicted_Sales"])],
            "RMSE ($)":       [np.sqrt(mean_squared_error(hol["Weekly_Sales"],  hol["Predicted_Sales"])),
                               np.sqrt(mean_squared_error(nhol["Weekly_Sales"], nhol["Predicted_Sales"]))],
            "R²":             [r2_score(hol["Weekly_Sales"],  hol["Predicted_Sales"]),
                               r2_score(nhol["Weekly_Sales"], nhol["Predicted_Sales"])],
        })
        st.dataframe(breakdown.style.format({"MAE ($)": "{:,.0f}",
                                              "RMSE ($)": "{:,.0f}",
                                              "R²": "{:.4f}"}), use_container_width=True)

        st.divider()
        # Actual vs Predicted chart
        st.subheader("Actual vs Predicted Sales (sample)")
        sample = results.sample(min(3000, len(results)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample["Weekly_Sales"], sample["Predicted_Sales"],
                   alpha=0.3, s=6, color="teal")
        mv = max(sample["Weekly_Sales"].max(), sample["Predicted_Sales"].max())
        ax.plot([0, mv], [0, mv], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Sales ($)")
        ax.set_ylabel("Predicted Sales ($)")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # Evaluation images
        for fname, title in [
            ("training_curve.png",    "Training Loss Curve"),
            ("residuals.png",         "Residual Distribution"),
        ]:
            path = f"outputs/evaluation/{fname}"
            if os.path.exists(path):
                st.subheader(title)
                st.image(path)

        st.divider()
        st.subheader("Raw Predictions (first 100 rows)")
        st.dataframe(results.head(100))

    except FileNotFoundError:
        st.warning("⚠️ Predictions not found. Please run **step4_ann_training.py** first.")

st.divider()
st.caption("ANN Retail Sales Forecasting * PyTorch * Streamlit * GUVI/HCL Project")