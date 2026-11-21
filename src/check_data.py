"""Sanity checks. Run after features.py. Every line should say OK."""
import pandas as pd
from features import RAW, OUT, load

def check(name, ok, info=""):
    print(("OK   " if ok else "FAIL ") + name, info)

raw = load(RAW)
check("raw rows = 31320", len(raw) == 31320, f"(got {len(raw)})")
check("raw columns = 12", raw.shape[1] == 12, f"(got {raw.shape[1]})")
check("60 wards", raw.ward_id.nunique() == 60, f"(got {raw.ward_id.nunique()})")
check("no missing values (raw)", raw.isna().sum().sum() == 0)
check("dates 2016-01-04 to 2025-12-29",
      str(raw.date.min().date()) == "2016-01-04" and str(raw.date.max().date()) == "2025-12-29",
      f"(got {raw.date.min().date()} to {raw.date.max().date()})")
check("522 weeks per ward", (raw.groupby("ward_id").size() == 522).all())

f = pd.read_csv(OUT, parse_dates=["date"])
check("features rows = 30600", len(f) == 30600, f"(got {len(f)})")
check("features columns = 45", f.shape[1] == 45, f"(got {f.shape[1]})")
check("no missing values (features)", f.isna().sum().sum() == 0)

# lag/target correctness inside one ward
w = f[f.ward_id == f.ward_id.iloc[0]].reset_index(drop=True)
check("rain_lag3 = rain 3 weeks earlier",
      (w.rain_lag3.iloc[3:] == w.rainfall_mm.shift(3).iloc[3:]).all())
check("target_3w = cases 3 weeks later",
      (w.target_3w.iloc[:-3] == w.dengue_cases.shift(-3).iloc[:-3]).all())
print("\nrain lag vs cases correlation (lag3 should be highest):")
print(f[[f"rain_lag{k}" for k in range(1, 7)] + ["dengue_cases"]].corr()["dengue_cases"].round(2).to_string())