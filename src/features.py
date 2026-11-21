"""Load raw data and build lag features + future targets, per ward."""
import pandas as pd

RAW = "data/hackathon_preli_dataset_-_bangladesh_dengue_large_dataset.csv"
OUT = "data/features.csv"

RAIN_LAGS = range(1, 7)    # 1-6 weeks back
CLIM_LAGS = range(1, 9)    # 1-8 weeks back
HORIZONS = (2, 3, 4)       # predict cases 2/3/4 weeks ahead


def load(path=RAW):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df):
    df = df.copy()
    g = df.groupby("ward_id")  # always group by ward

    new = {}
    for k in RAIN_LAGS:
        new[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    for k in CLIM_LAGS:
        new[f"temp_lag{k}"] = g["avg_temp_c"].shift(k)
        new[f"hum_lag{k}"] = g["relative_humidity_pct"].shift(k)

    for w in (2, 3, 4):
        new[f"rain_sum{w}w"] = g["rainfall_mm"].transform(lambda s: s.rolling(w).sum())

    new["cases_lag1"] = g["dengue_cases"].shift(1)
    new["cases_lag2"] = g["dengue_cases"].shift(2)

    for h in HORIZONS:
        new[f"target_{h}w"] = g["dengue_cases"].shift(-h)

    df = pd.concat([df, pd.DataFrame(new)], axis=1)

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["city_code"] = df["city"].astype("category").cat.codes
    return df


if __name__ == "__main__":
    df = build_features(load())
    n0 = len(df)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"rows {n0} -> {len(df)} | columns {df.shape[1]} | saved {OUT}")