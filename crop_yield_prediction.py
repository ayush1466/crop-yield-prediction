"""
Crop Yield Prediction Using Machine Learning
============================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
df = pd.read_csv("crop_yield_dataset.csv")

# 🔥 FIX 1: Clean column names (removes spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

print("=" * 55)
print("  CROP YIELD PREDICTION — ML PROJECT")
print("=" * 55)

print("\nAvailable columns:", df.columns.tolist())

# 🔥 FIX 2: Auto-detect column names
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

yield_col = find_col(['yield_kg_per_ha'])
fert_col  = find_col(['fertilizer_kg_per_ha', 'fertilizer_kg_ha'])
rain_col  = find_col(['rainfall_mm'])
temp_col  = find_col(['temperature', 'temperature_c'])
humid_col = find_col(['humidity', 'humidity_pct'])

# Check required columns
if None in [yield_col, fert_col, rain_col, temp_col, humid_col]:
    print("\n❌ ERROR: Some required columns not found.")
    print("👉 Please check dataset column names.")
    exit()

# ─────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────
print("\n[STEP 2] Exploratory Data Analysis")
print(f"  Dataset shape : {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Yield range   : {df[yield_col].min():.0f} – {df[yield_col].max():.0f} kg/ha")
print(f"  Avg yield     : {df[yield_col].mean():.0f} kg/ha")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0,0].hist(df[yield_col], bins=40)
axes[0,0].set_title("Yield Distribution")

if 'crop' in df.columns:
    df.groupby('crop')[yield_col].mean().plot(kind='barh', ax=axes[0,1])
    axes[0,1].set_title("Avg Yield by Crop")

if 'season' in df.columns:
    df.groupby('season')[yield_col].mean().plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title("Avg Yield by Season")

axes[1,0].scatter(df[rain_col], df[yield_col], alpha=0.3)
axes[1,0].set_title("Rainfall vs Yield")

axes[1,1].scatter(df[fert_col], df[yield_col], alpha=0.3)
axes[1,1].set_title("Fertilizer vs Yield")

num_cols = [rain_col, temp_col, humid_col, fert_col, yield_col]
sns.heatmap(df[num_cols].corr(), annot=True, ax=axes[1,2])

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# STEP 3: Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Data Preprocessing")

le = LabelEncoder()

for col in ['crop', 'state', 'season', 'soil_type']:
    if col in df.columns:
        df[col + '_enc'] = le.fit_transform(df[col])
    else:
        df[col + '_enc'] = 0  # fallback

feature_cols = [
    'crop_enc', 'state_enc', 'season_enc', 'soil_type_enc',
    rain_col, temp_col, humid_col, fert_col
]

if 'area_hectares' in df.columns:
    feature_cols.append('area_hectares')
else:
    df['area_hectares'] = 1
    feature_cols.append('area_hectares')

X = df[feature_cols]
y = df[yield_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 4: Models
# ─────────────────────────────────────────────
print("\n[STEP 4] Training Models...")

def evaluate(name, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    print(f"{name} R2:", r2_score(y_test, preds))
    return model, preds

lr_model, lr_preds = evaluate("Linear", LinearRegression(), X_train_sc, X_test_sc)
rf_model, rf_preds = evaluate("Random Forest", RandomForestRegressor(), X_train, X_test)
gb_model, gb_preds = evaluate("Gradient Boost", GradientBoostingRegressor(), X_train, X_test)

# ─────────────────────────────────────────────
# STEP 5: Prediction
# ─────────────────────────────────────────────
print("\n[STEP 5] Sample Prediction")

def predict_yield(rainfall, temp, humidity, fertilizer, area=1):
    sample = np.array([[0,0,0,0, rainfall, temp, humidity, fertilizer, area]])
    pred = rf_model.predict(sample)[0]
    print(f"Predicted Yield: {pred:.2f} kg/ha")

predict_yield(1200, 28, 75, 150, 5)

print("\n✅ Project completed without errors!")