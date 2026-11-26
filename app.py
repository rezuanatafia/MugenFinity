"""Streamlit app for ward-level dengue forecasts."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_features, load  # noqa: E402

RAW = ROOT / "data" / "hackathon_preli_dataset_-_bangladesh_dengue_large_dataset.csv"
BUNDLE = ROOT / "data" / "models" / "dengue_forecast_bundle.pkl"

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE)


@st.cache_data
def load_features():
    return build_features(load(RAW))


def risk_tier(cases):
    if cases < 2:
        return "Low"
    if cases < 5:
        return "Medium"
    if cases < 12:
        return "High"
    return "Critical"


def forecast(models, X):
    point = max(float(models["point_model"].predict(X)[0]), 0.0)
    prob = float(models["clf_model"].predict_proba(X)[0, 1])
    low = min(max(float(models["q10_model"].predict(X)[0]), 0.0), point)
    high = max(float(models["q90_model"].predict(X)[0]), point)
    return point, prob, low, high


st.title("Dengue Early Warning: Bangladesh")
if not BUNDLE.exists():
    st.error("Model file not found. Run `python train.py` first.")
    st.stop()

bundle = load_bundle()
data = load_features()
cols = bundle["feature_cols"]
usable = data.dropna(subset=cols)

with st.sidebar:
    st.header("Choose a forecast")
    city = st.selectbox("City", sorted(usable["city"].unique()))
    ward = st.selectbox("Ward", sorted(usable.loc[usable["city"] == city, "ward_id"].unique()))
    rows = usable[(usable["ward_id"] == ward) & (usable["date"].dt.year >= 2025)]
    dates = list(rows["date"].dt.date)
    date = st.selectbox("Data available up to week of", dates, index=len(dates) - 1)
    weeks = st.radio("Weeks ahead", [2, 3, 4], horizontal=True)

row = rows[rows["date"].dt.date == date].iloc[[0]]
point, prob, low, high = forecast(bundle["horizons"][f"target_{weeks}w"], row[cols])
target_date = date + pd.Timedelta(weeks=weeks)

st.write(
    "Forecasts dengue cases 2 to 4 weeks ahead for a selected ward, "
    "using weather, ward conditions and recent cases."
)

st.subheader(f"{ward}: forecast for week of {target_date}")
a, b, c = st.columns(3)
a.metric("Outbreak probability (1+ case)", f"{prob:.1%}")
b.metric("Predicted cases", f"{point:.1f}")
c.metric("Risk tier", risk_tier(point))
st.write(f"80% prediction interval: **{low:.1f} to {high:.1f} cases**")

actual = row[f"target_{weeks}w"].iloc[0]
if np.isnan(actual):
    st.info("The actual case count for this week is not in the dataset.")
else:
    st.write(f"Actual recorded cases: **{int(actual)}**")

st.subheader("All forecast periods")
table = []
for w in (2, 3, 4):
    p, pr, lo, hi = forecast(bundle["horizons"][f"target_{w}w"], row[cols])
    table.append({
        "Weeks ahead": w,
        "Week of": str(date + pd.Timedelta(weeks=w)),
        "Outbreak probability": f"{pr:.1%}",
        "Predicted cases": round(p, 1),
        "80% interval": f"{lo:.1f} to {hi:.1f}",
        "Risk tier": risk_tier(p),
    })
st.dataframe(pd.DataFrame(table), hide_index=True)
st.caption("Risk tiers by predicted cases: Low < 2, Medium 2 to < 5, High 5 to < 12, Critical 12+.")

st.subheader(f"Conditions in the week of {date}")
now = row.iloc[0]
a, b, c = st.columns(3)
a.metric("Rainfall", f"{now['rainfall_mm']:.1f} mm")
b.metric("Average temperature", f"{now['avg_temp_c']:.1f} °C")
c.metric("Humidity", f"{now['relative_humidity_pct']:.0f}%")
a, b, c = st.columns(3)
a.metric("Population per sq km", f"{now['population_density_sqkm']:,.0f}")
b.metric("Water-logging risk index", f"{now['water_logging_risk_index']:.2f}")
c.metric("Sanitation score", f"{now['sanitation_score']:.2f}")

st.subheader("Weekly cases in the year up to the selected week")
history = data[(data["ward_id"] == ward) & (data["date"].dt.date <= date)].tail(52)
st.line_chart(history.set_index("date")["dengue_cases"])

st.subheader("Weekly rainfall (mm) over the same period")
st.bar_chart(history.set_index("date")["rainfall_mm"])

st.caption(
    "Student project based on historical data. Not confirmed case counts, "
    "medical guidance or an official warning."
)
